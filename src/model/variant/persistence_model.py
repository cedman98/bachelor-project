"""
Persistence (naive) baseline model for wind forecasting.

This model predicts the last known value for all future time steps,
serving as a simple baseline to compare against more sophisticated models.
"""

import os
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.model.variant.model_interface import ModelInterface


class PersistenceModel(ModelInterface):
    """
    Persistence baseline: predicts the last known value for all future steps.
    
    - Uses last observed [speed, dir_sin, dir_cos] values and repeats them for horizon_steps
    - Minimal preprocessing: -999 sentinel to NaN, forward/backward fill per station
    - Does not require training (no parameters to learn)
    - Uses same representation as BiLSTM: speed + direction as sin/cos components
    """

    def __init__(
        self,
        history_steps: int = 72,
        horizon_steps: int = 72,
    ) -> None:
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps

    def train(self, dataset: pd.DataFrame) -> None:
        """
        The persistence model has no parameters to train.
        This method is included to match the ModelInterface but does nothing.
        """
        logger.info("Persistence model requires no training")

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Predict by repeating the last known value for all horizon steps.
        
        For each station, uses the most recent history_steps to make a single prediction.
        If you need predictions for all sliding windows (for evaluation), use evaluate() instead.
        
        Returns:
            DataFrame with columns: station_id, record_date, average_wind_speed_pred, average_wind_direction_pred
        """
        df = self._prepare_dataframe(dataset)
        
        results = []
        freq = pd.Timedelta("10min")
        
        for station_id, group in df.groupby("station_id"):
            group = group.sort_values("record_date").reset_index(drop=True)
            
            # Use only the most recent history_steps for prediction
            if len(group) < self.history_steps:
                continue
                
            hist = group.tail(self.history_steps)
            
            # Get last known speed, dir_sin, dir_cos values
            last_speed = hist.iloc[-1]["speed"]
            last_dir_sin = hist.iloc[-1]["dir_sin"]
            last_dir_cos = hist.iloc[-1]["dir_cos"]
            
            # Skip if NaN
            if pd.isna(last_speed) or pd.isna(last_dir_sin) or pd.isna(last_dir_cos):
                continue
            
            # Repeat for all horizon steps (this creates the flat line)
            speed_pred = np.full(self.horizon_steps, last_speed)
            dir_sin_pred = np.full(self.horizon_steps, last_dir_sin)
            dir_cos_pred = np.full(self.horizon_steps, last_dir_cos)
            
            # Normalize and decode direction from sin/cos (coming-from convention)
            norm = np.sqrt(dir_sin_pred**2 + dir_cos_pred**2) + 1e-8
            dir_sin_n = dir_sin_pred / norm
            dir_cos_n = dir_cos_pred / norm
            direction_vals = (np.degrees(np.arctan2(dir_sin_n, dir_cos_n)) % 360.0)
            
            # Build timestamps for the forecast horizon
            last_ts = pd.to_datetime(hist.iloc[-1]["record_date"])
            future_index = [last_ts + freq * (j + 1) for j in range(self.horizon_steps)]
            
            out_df = pd.DataFrame(
                {
                    "station_id": station_id,
                    "record_date": future_index,
                    "average_wind_speed_pred": speed_pred,
                    "average_wind_direction_pred": direction_vals,
                }
            )
            results.append(out_df)
        
        if not results:
            raise ValueError("No predictions could be generated.")
        
        return pd.concat(results, axis=0).reset_index(drop=True)

    def evaluate(
        self, dataset: pd.DataFrame, max_batches: int | None = None
    ) -> Dict[str, float]:
        """
        Evaluate the persistence model on a test dataset.
        
        Returns:
            Dictionary with MAE and RMSE metrics for speed and direction
        """
        df = self._prepare_dataframe(dataset)
        sequences = self._build_sequences(df)
        
        if len(sequences) == 0:
            raise ValueError(
                "No evaluation sequences could be constructed. "
                "Ensure sufficient history per station."
            )

        logger.info(f"Evaluating on {len(sequences)} sequences")

        preds_speed: List[np.ndarray] = []
        trues_speed: List[np.ndarray] = []
        preds_dir_deg: List[np.ndarray] = []
        trues_dir_deg: List[np.ndarray] = []

        for idx, (x, y) in enumerate(sequences):
            if max_batches is not None and idx >= max_batches:
                break
            
            # x: [history_steps, 3], y: [horizon_steps, 3] where 3 = [speed, dir_sin, dir_cos]
            # Get last value from history
            last_speed = x[-1, 0]
            last_dir_sin = x[-1, 1]
            last_dir_cos = x[-1, 2]
            
            # Predict: repeat last value
            speed_pred = np.full(self.horizon_steps, last_speed)
            dir_sin_pred = np.full(self.horizon_steps, last_dir_sin)
            dir_cos_pred = np.full(self.horizon_steps, last_dir_cos)
            
            # Decode directions from sin/cos to degrees
            pred_norm = np.sqrt(dir_sin_pred**2 + dir_cos_pred**2) + 1e-8
            pred_dir = (np.degrees(np.arctan2(dir_sin_pred / pred_norm, dir_cos_pred / pred_norm)) % 360.0)
            
            true_sin = y[:, 1]
            true_cos = y[:, 2]
            true_norm = np.sqrt(true_sin**2 + true_cos**2) + 1e-8
            true_dir = (np.degrees(np.arctan2(true_sin / true_norm, true_cos / true_norm)) % 360.0)
            
            # Collect predictions and ground truth
            preds_speed.append(speed_pred)
            trues_speed.append(y[:, 0])
            preds_dir_deg.append(pred_dir)
            trues_dir_deg.append(true_dir)

        speed_pred_all = np.concatenate(preds_speed)
        speed_true_all = np.concatenate(trues_speed)
        dir_pred_all = np.concatenate(preds_dir_deg)
        dir_true_all = np.concatenate(trues_dir_deg)

        # Speed metrics
        mae_speed = float(np.mean(np.abs(speed_pred_all - speed_true_all)))
        rmse_speed = float(np.sqrt(np.mean((speed_pred_all - speed_true_all) ** 2)))

        # Direction error (minimal absolute angular difference)
        def angular_mae_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
            diff = np.abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)
            return float(np.mean(diff))

        mae_direction_deg = angular_mae_deg(dir_pred_all, dir_true_all)

        metrics = {
            "mae_speed": mae_speed,
            "rmse_speed": rmse_speed,
            "mae_direction_deg": mae_direction_deg,
        }

        logger.info(
            "Persistence Model Evaluation: "
            + ", ".join(
                [
                    f"mae_speed={mae_speed:.4f}",
                    f"rmse_speed={rmse_speed:.4f}",
                    f"mae_direction_deg={mae_direction_deg:.2f}°",
                ]
            )
        )

        return metrics

    def save(self, path: str) -> None:
        """Save model metadata to disk."""
        os.makedirs(path, exist_ok=True)
        metadata = {
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
        }
        joblib.dump(metadata, os.path.join(path, "metadata.joblib"))
        logger.info(f"Persistence model metadata saved to {path}")

    def load(self, path: str) -> None:
        """Load model metadata from disk."""
        metadata = joblib.load(os.path.join(path, "metadata.joblib"))
        self.history_steps = metadata["history_steps"]
        self.horizon_steps = metadata["horizon_steps"]
        logger.info(f"Persistence model metadata loaded from {path}")

    def _prepare_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for the persistence model.
        Minimal preprocessing: handle -999 sentinels, compute speed/dir_sin/dir_cos, resample.
        """
        required_cols = [
            "station_id",
            "record_date",
            "average_wind_speed",
            "average_wind_direction",
        ]
        missing = [c for c in required_cols if c not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df = dataset.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["record_date"]):
            df["record_date"] = pd.to_datetime(
                df["record_date"], errors="coerce", utc=False
            )

        # Replace sentinel -999 with NaN
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        if "station_id" in numeric_cols:
            numeric_cols.remove("station_id")
        for col in numeric_cols:
            df[col] = df[col].replace(-999, np.nan)

        # Compute speed and direction components (meteorological coming-from convention)
        direction_rad = np.deg2rad(df["average_wind_direction"].astype(float) % 360)
        speed = df["average_wind_speed"].astype(float)
        
        # Speed and direction as sin/cos (matching BiLSTM representation)
        df["speed"] = speed
        df["dir_sin"] = np.sin(direction_rad)
        df["dir_cos"] = np.cos(direction_rad)

        # Resample to regular 10-minute grid per station
        freq = "10min"

        def resample_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("record_date").set_index("record_date")
            g = g.resample(freq).first()
            
            # Forward and backward fill for speed, dir_sin, dir_cos
            g[["speed", "dir_sin", "dir_cos"]] = g[["speed", "dir_sin", "dir_cos"]].ffill().bfill()
            
            g = g.reset_index()
            g["station_id"] = g["station_id"].ffill().bfill()
            return g

        df = (
            df.groupby("station_id", group_keys=False)
            .apply(resample_group)
            .reset_index(drop=True)
        )

        return df

    def _build_sequences(
        self, df: pd.DataFrame
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """
        Build sliding-window sequences for evaluation.
        
        Returns:
            List of (X, y) tuples where:
            - X: [history_steps, 3] for [speed, dir_sin, dir_cos]
            - y: [horizon_steps, 3] for [speed, dir_sin, dir_cos]
        """
        sequences = []
        
        for station_id, group in df.groupby("station_id"):
            group = group.sort_values("record_date").reset_index(drop=True)
            
            for i in range(
                len(group) - self.history_steps - self.horizon_steps + 1
            ):
                hist = group.iloc[i : i + self.history_steps]
                future = group.iloc[
                    i + self.history_steps : i + self.history_steps + self.horizon_steps
                ]
                
                # Extract speed, dir_sin, dir_cos
                x = hist[["speed", "dir_sin", "dir_cos"]].values
                y = future[["speed", "dir_sin", "dir_cos"]].values
                
                # Skip if any NaN
                if np.isnan(x).any() or np.isnan(y).any():
                    continue
                
                sequences.append((x, y))
        
        return sequences

