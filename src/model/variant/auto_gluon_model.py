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
        self, prediction_length_minutes: int = 12 * 60, interval_minutes: int = 10
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

        # Replace -999 sentinel with NaN in numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].replace(-999, np.nan)

        # Sort for time-series modeling
        df = df.sort_values([self._item_id_col, self._timestamp_col])

        # Known covariates: all non-target columns except identifiers and timestamp
        known_covariates = [
            c
            for c in df.columns
            if c not in {self._item_id_col, self._timestamp_col, *self._target_cols}
        ]

        # Build per-target long-format frames that AutoGluon expects
        per_target_frames: Dict[str, pd.DataFrame] = {}
        # Persist known covariates for use during training/prediction
        self._known_covariates_names = known_covariates
        for target in self._target_cols:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in dataset.")

            target_df = df[
                [self._item_id_col, self._timestamp_col, target] + known_covariates
            ].rename(columns={target: "target"})
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
            )

            predictor.fit(
                train_data=ts_train,
                presets="medium_quality",
                time_limit=120,
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
