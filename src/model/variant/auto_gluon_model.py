from src.model.variant.model_interface import ModelInterface
import pandas as pd
import numpy as np
from typing import Dict
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import os


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
        self,
        prediction_length_minutes: int = 8 * 60,
        interval_minutes: int = 10,
        save_dir: str | None = None,
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
        # Optional base directory to store fitted predictors
        self._save_dir: str | None = save_dir

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

            # If a save directory is provided, store each target under its own subdirectory
            predictor_output_dir = (
                os.path.join(self._save_dir, target) if self._save_dir else None
            )

            if predictor_output_dir:
                os.makedirs(predictor_output_dir, exist_ok=True)

            predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length_steps,
                freq=self._freq,
                eval_metric="MAE",
                known_covariates_names=getattr(self, "_known_covariates_names", None),
                verbosity=4,
                path=predictor_output_dir if predictor_output_dir else None,
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

            # Provide future known covariates explicitly if they were used during training
            known_covariates_names = getattr(self, "_known_covariates_names", [])
            if known_covariates_names:
                # Build the exact future index that the predictor expects
                future_index_df = predictor.make_future_data_frame(ts_test)
                future_reset = future_index_df.reset_index()
                cov_df = future_reset.rename(
                    columns={
                        "item_id": self._item_id_col,
                        "timestamp": self._timestamp_col,
                    }
                )[[self._item_id_col, self._timestamp_col]]

                # Compute deterministic time covariates identical to training
                ts = cov_df[self._timestamp_col]
                hour = ts.dt.hour.astype(int)
                cov_df["hour_sin"] = np.sin(2 * np.pi * (hour / 24.0))
                cov_df["hour_cos"] = np.cos(2 * np.pi * (hour / 24.0))
                day_of_year = ts.dt.dayofyear.astype(int)
                cov_df["doy_sin"] = np.sin(2 * np.pi * (day_of_year / 365.0))
                cov_df["doy_cos"] = np.cos(2 * np.pi * (day_of_year / 365.0))

                # Keep only the covariates used during training
                keep_covs = [c for c in known_covariates_names if c in cov_df.columns]
                cov_df = cov_df[[self._item_id_col, self._timestamp_col] + keep_covs]

                # Ensure item id dtype matches training
                cov_df[self._item_id_col] = cov_df[self._item_id_col].astype(str)

                cov_ts = TimeSeriesDataFrame.from_data_frame(
                    cov_df,
                    id_column=self._item_id_col,
                    timestamp_column=self._timestamp_col,
                )
                predictions = predictor.predict(ts_test, known_covariates=cov_ts)
            else:
                predictions = predictor.predict(ts_test)

            # Convert to a flat pandas DataFrame with identifiers
            pred_df = predictions.reset_index()
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
            # Ensure pure pandas DataFrame (not TimeSeriesDataFrame subclass)
            forecasts[target] = pd.DataFrame(pred_point)

        # Merge u and v predictions on station_id & timestamp
        merged = pd.DataFrame(forecasts[self._target_cols[0]])
        for t in self._target_cols[1:]:
            merged = pd.DataFrame(merged).merge(
                forecasts[t], on=[self._item_id_col, self._timestamp_col], how="outer"
            )

        return pd.DataFrame(merged)

    def save(self, path: str) -> None:
        """
        Save per-target predictors into subdirectories under the given path.
        Structure:
            path/
              u/
                predictor.pkl ...
              v/
                predictor.pkl ...
        """
        if not self._predictors:
            raise RuntimeError("Nothing to save. Train or load a model first.")

        os.makedirs(path, exist_ok=True)
        for target, predictor in self._predictors.items():
            target_dir = os.path.join(path, target)
            os.makedirs(target_dir, exist_ok=True)
            # AutoGluon handles its own directory structure
            predictor.save(path=target_dir)

    def load(self, path: str) -> None:
        """
        Load per-target predictors saved via `save(path)`.
        Expects subdirectories for each target under `path`.
        """
        loaded: Dict[str, TimeSeriesPredictor] = {}
        for target in self._target_cols:
            target_dir = os.path.join(path, target)
            if not os.path.isdir(target_dir):
                raise FileNotFoundError(
                    f"Expected directory for target '{target}' not found at: {target_dir}"
                )
            loaded[target] = TimeSeriesPredictor.load(target_dir)
        self._predictors = loaded

    def evaluate(self, test_dataset: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on a test DataFrame (same shape/columns style as `train`).

        Steps:
        1) Run `predict(test_dataset)` to obtain forecasts (u_pred, v_pred) per
           `station_id` and `record_date`.
        2) Align predictions with ground-truth `u` and `v` in `test_dataset` on
           keys [`station_id`, `record_date`]. Only rows present in both are
           evaluated.

        Returns a dictionary of metrics across aligned rows:
        - u_mae, u_rmse, u_mape
        - v_mae, v_rmse, v_mape
        - wind_speed_mae (|speed_true - speed_pred|)
        - vector_rmse (sqrt(mean((du)^2 + (dv)^2)))
        - overall_mae (mean of u_mae and v_mae)
        """

        # Prepare context (history) and evaluation horizon per series
        df = test_dataset.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self._timestamp_col]):
            df[self._timestamp_col] = pd.to_datetime(
                df[self._timestamp_col], errors="coerce", utc=False
            )
        if self._item_id_col in df.columns:
            df[self._item_id_col] = df[self._item_id_col].astype(str)
        df = df.sort_values([self._item_id_col, self._timestamp_col]).reset_index(
            drop=True
        )

        contexts: list[pd.DataFrame] = []
        truths: list[pd.DataFrame] = []
        horizon = self.prediction_length_steps
        for station_id, g in df.groupby(self._item_id_col, sort=False):
            if len(g) <= horizon:
                # Skip series with insufficient length for evaluation
                continue
            contexts.append(g.iloc[: len(g) - horizon])
            truths.append(g.iloc[len(g) - horizon :])

        if not contexts or not truths:
            raise ValueError(
                "Insufficient series length for evaluation. Provide longer histories per station."
            )

        context_df = pd.concat(contexts, axis=0).reset_index(drop=True)
        truth_horizon_df = pd.concat(truths, axis=0).reset_index(drop=True)

        # Forecast next horizon based on context
        pred_df = self.predict(context_df)

        # Prepare ground truth subset for the horizon
        truth_df = truth_horizon_df[[self._item_id_col, self._timestamp_col, "u", "v"]]

        # Merge predictions with ground truth on station/timestamp
        eval_df = pred_df.merge(
            truth_df,
            on=[self._item_id_col, self._timestamp_col],
            how="inner",
        )

        # Replace sentinel -999 with NaN if present on ground truth
        for col in ["u", "v"]:
            if pd.api.types.is_integer_dtype(eval_df[col].dtype):
                eval_df[col] = eval_df[col].astype("Float64")
            eval_df[col] = eval_df[col].replace(-999, np.nan)

        # Ensure pandas DataFrame, then drop rows with missing values in required columns
        eval_df = (
            pd.DataFrame(eval_df)
            .dropna(subset=["u", "v", "u_pred", "v_pred"])
            .reset_index(drop=True)
        )

        if eval_df.empty:
            raise ValueError(
                "No overlapping rows between predictions and truth. Ensure sufficient context length and matching stations."
            )

        y_u_true = eval_df["u"].to_numpy(dtype=float)
        y_v_true = eval_df["v"].to_numpy(dtype=float)
        y_u_pred = eval_df["u_pred"].to_numpy(dtype=float)
        y_v_pred = eval_df["v_pred"].to_numpy(dtype=float)

        def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(np.abs(y_true - y_pred)))

        def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        def mean_absolute_percentage_error(
            y_true: np.ndarray, y_pred: np.ndarray
        ) -> float:
            eps = 1e-9
            mask = np.abs(y_true) > eps
            if not np.any(mask):
                return float("nan")
            return float(
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
            )

        # Component-wise metrics
        u_mae = mean_absolute_error(y_u_true, y_u_pred)
        v_mae = mean_absolute_error(y_v_true, y_v_pred)
        u_rmse = root_mean_squared_error(y_u_true, y_u_pred)
        v_rmse = root_mean_squared_error(y_v_true, y_v_pred)
        u_mape = mean_absolute_percentage_error(y_u_true, y_u_pred)
        v_mape = mean_absolute_percentage_error(y_v_true, y_v_pred)

        # Speed error
        speed_true = np.sqrt(y_u_true**2 + y_v_true**2)
        speed_pred = np.sqrt(y_u_pred**2 + y_v_pred**2)
        wind_speed_mae = mean_absolute_error(speed_true, speed_pred)

        # Vector RMSE across components
        vector_rmse = float(
            np.sqrt(np.mean((y_u_true - y_u_pred) ** 2 + (y_v_true - y_v_pred) ** 2))
        )

        overall_mae = float(np.mean([u_mae, v_mae]))

        return {
            "u_mae": u_mae,
            "u_rmse": u_rmse,
            "u_mape": u_mape,
            "v_mae": v_mae,
            "v_rmse": v_rmse,
            "v_mape": v_mape,
            "wind_speed_mae": wind_speed_mae,
            "vector_rmse": vector_rmse,
            "overall_mae": overall_mae,
        }
