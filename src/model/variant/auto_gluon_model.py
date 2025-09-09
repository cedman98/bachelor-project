from src.model.variant.model_interface import ModelInterface
import pandas as pd
import numpy as np
from typing import Dict
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


class AutoGluonModel(ModelInterface):
    """
    Multi-series time series forecasting using AutoGluon 1.4.

    - Identifies separate series by `station_id`.
    - Forecasts both `u` and `v` targets independently for 12 hours ahead.
    - Assumes a 10-minute sampling frequency -> 72 steps forecast horizon.
    - Treats -999 as missing values.
    - Uses engineered features (e.g., time encodings, lags, rolls) as known covariates.
    """

    def __init__(
        self, prediction_length_minutes: int = 8 * 60, interval_minutes: int = 10
    ):
        # prediction length in steps (10 minutes -> 6 steps/hour)
        self.interval_minutes = interval_minutes
        self.prediction_length_steps = int(
            prediction_length_minutes // interval_minutes
        )

        # Fitted predictors per target
        self._predictors: Dict[str, TimeSeriesPredictor] = {}
        self._timestamp_col: str = "record_date"
        self._item_id_col: str = "station_id"
        self._target_cols = ["u", "v"]
        self._freq = "10min"

    def _prepare_datasets(self, dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = dataset.copy()

        # Ensure timestamp dtype
        if not pd.api.types.is_datetime64_any_dtype(df[self._timestamp_col]):
            df[self._timestamp_col] = pd.to_datetime(
                df[self._timestamp_col], errors="coerce", utc=False
            )

        # Ensure item id column dtype is valid for AutoGluon (int or string)
        # Cast to string to be robust across sources
        if self._item_id_col in df.columns:
            df[self._item_id_col] = df[self._item_id_col].astype(str)

        # Replace -999 sentinel with NaN in numeric columns (avoid dtype casting warnings)
        # Exclude the item id column from numeric processing
        numeric_cols = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c != self._item_id_col
        ]
        for col in numeric_cols:
            if pd.api.types.is_integer_dtype(df[col].dtype):
                # Use pandas nullable float dtype to allow NaN without warnings
                df[col] = df[col].astype("Float64")
            df[col] = df[col].replace(-999, np.nan)

        # Sort for time-series modeling
        df = df.sort_values([self._item_id_col, self._timestamp_col])

        # Known covariates: restrict to features that are actually known into the future
        # (exclude identifiers, targets, and history-derived lags/rolls)
        future_known_candidates = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        known_covariates = [c for c in future_known_candidates if c in df.columns]

        # Build per-target long-format frames that AutoGluon expects
        per_target_frames: Dict[str, pd.DataFrame] = {}
        # Persist known covariates for use during training/prediction
        self._known_covariates_names = known_covariates
        for target in self._target_cols:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in dataset.")

            base_cols = [self._item_id_col, self._timestamp_col, target]
            target_df = df[base_cols + known_covariates].rename(
                columns={target: "target"}
            )
            per_target_frames[target] = target_df

        return per_target_frames

    def train(self, dataset: pd.DataFrame) -> None:
        per_target_frames = self._prepare_datasets(dataset)

        for target, train_df in per_target_frames.items():
            # Convert to AutoGluon TimeSeriesDataFrame
            ts_train = TimeSeriesDataFrame.from_data_frame(
                train_df,
                id_column=self._item_id_col,
                timestamp_column=self._timestamp_col,
            )

            predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length_steps,
                freq=self._freq,
                eval_metric="MAE",
                known_covariates_names=getattr(self, "_known_covariates_names", None),
                verbosity=4,
            )

            non_chronos_hyperparameters = {
                "Naive": {},
                "SeasonalNaive": {},
                "ETS": {},
                "ARIMA": {},
                "AutoARIMA": {},
                "Theta": {},
                "DirectTabular": {},
                "RecursiveTabular": {},
            }

            predictor.fit(
                train_data=ts_train,
                hyperparameters=non_chronos_hyperparameters,
                time_limit=600,
            )
            self._predictors[target] = predictor

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not self._predictors:
            raise RuntimeError("Model not trained. Call train() before predict().")

        per_target_frames = self._prepare_datasets(dataset)

        forecasts: Dict[str, pd.DataFrame] = {}
        for target, test_df in per_target_frames.items():
            predictor = self._predictors.get(target)
            if predictor is None:
                raise RuntimeError(
                    f"Predictor for target '{target}' not available. Train first."
                )

            ts_test = TimeSeriesDataFrame.from_data_frame(
                test_df,
                id_column=self._item_id_col,
                timestamp_column=self._timestamp_col,
            )

            predictions = predictor.predict(ts_test)

            # Convert to a flat pandas DataFrame with identifiers
            pred_df = predictions.to_pandas().reset_index()
            pred_df = pred_df.rename(
                columns={
                    "item_id": self._item_id_col,
                    "timestamp": self._timestamp_col,
                    "mean": f"{target}_pred",
                }
            )

            # Keep point forecast (mean)
            keep_cols = [self._item_id_col, self._timestamp_col, f"{target}_pred"]
            pred_point = pred_df[keep_cols]
            forecasts[target] = pred_point

        # Merge u and v predictions on station_id & timestamp
        merged = forecasts[self._target_cols[0]]
        for t in self._target_cols[1:]:
            merged = merged.merge(
                forecasts[t], on=[self._item_id_col, self._timestamp_col], how="outer"
            )

        return merged
