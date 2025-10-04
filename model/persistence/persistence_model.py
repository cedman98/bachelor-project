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
    
    - Uses last observed [u, v] values and repeats them for horizon_steps
    - Minimal preprocessing: -999 sentinel to NaN, forward/backward fill per station
    - Does not require training (no parameters to learn)
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
        
        Returns:
            DataFrame with columns: station_id, record_date, u_pred, v_pred
        """
        df = self._prepare_dataframe(dataset)
        
        results = []
        freq = pd.Timedelta("10min")
        
        for station_id, group in df.groupby("station_id"):
            group = group.sort_values("record_date").reset_index(drop=True)
            
            # Build sliding windows
            for i in range(len(group) - self.history_steps - self.horizon_steps + 1):
                hist = group.iloc[i : i + self.history_steps]
                
                # Get last known u, v values
                last_u = hist.iloc[-1]["u"]
                last_v = hist.iloc[-1]["v"]
                
                # Skip if NaN
                if pd.isna(last_u) or pd.isna(last_v):
                    continue
                
                # Repeat for all horizon steps
                u_pred = np.full(self.horizon_steps, last_u)
                v_pred = np.full(self.horizon_steps, last_v)
                
                # Build timestamps for the forecast horizon
                last_ts = pd.to_datetime(hist.iloc[-1]["record_date"])
                future_index = [last_ts + freq * (j + 1) for j in range(self.horizon_steps)]
                
                out_df = pd.DataFrame(
                    {
                        "station_id": station_id,
                        "record_date": future_index,
                        "u_pred": u_pred,
                        "v_pred": v_pred,
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
            Dictionary with MAE and RMSE metrics for u, v, speed, and direction
        """
        df = self._prepare_dataframe(dataset)
        sequences = self._build_sequences(df)
        
        if len(sequences) == 0:
            raise ValueError(
                "No evaluation sequences could be constructed. "
                "Ensure sufficient history per station."
            )

        logger.info(f"Evaluating on {len(sequences)} sequences")

        preds_u: List[np.ndarray] = []
        preds_v: List[np.ndarray] = []
        trues_u: List[np.ndarray] = []
        trues_v: List[np.ndarray] = []

        for idx, (x, y) in enumerate(sequences):
            if max_batches is not None and idx >= max_batches:
                break
            
            # x: [history_steps, 2], y: [horizon_steps, 2]
            # Get last value from history
            last_u = x[-1, 0]
            last_v = x[-1, 1]
            
            # Predict: repeat last value
            u_pred = np.full(self.horizon_steps, last_u)
            v_pred = np.full(self.horizon_steps, last_v)
            
            # Collect predictions and ground truth
            preds_u.append(u_pred)
            preds_v.append(v_pred)
            trues_u.append(y[:, 0])
            trues_v.append(y[:, 1])

        u_pred = np.concatenate(preds_u)
        v_pred = np.concatenate(preds_v)
        u_true = np.concatenate(trues_u)
        v_true = np.concatenate(trues_v)

        # Basic metrics for u, v
        mae_u = float(np.mean(np.abs(u_pred - u_true)))
        rmse_u = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
        mae_v = float(np.mean(np.abs(v_pred - v_true)))
        rmse_v = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))

        # Derived speed metrics
        speed_pred = np.sqrt(u_pred**2 + v_pred**2)
        speed_true = np.sqrt(u_true**2 + v_true**2)
        mae_speed = float(np.mean(np.abs(speed_pred - speed_true)))
        rmse_speed = float(np.sqrt(np.mean((speed_pred - speed_true) ** 2)))

        # Direction error (meteorological coming-from): dir = atan2(-u, -v) in degrees [0, 360)
        def uv_to_dir_deg(u_arr: np.ndarray, v_arr: np.ndarray) -> np.ndarray:
            ang = np.degrees(np.arctan2(-u_arr, -v_arr))
            ang = np.mod(ang, 360.0)
            return ang

        def angular_mae_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
            # Minimal absolute angular difference on the circle
            diff = np.abs(((a_deg - b_deg + 180.0) % 360.0) - 180.0)
            return float(np.mean(diff))

        dir_pred = uv_to_dir_deg(u_pred, v_pred)
        dir_true = uv_to_dir_deg(u_true, v_true)
        mae_dir_deg = angular_mae_deg(dir_pred, dir_true)

        metrics = {
            "mae_u": mae_u,
            "rmse_u": rmse_u,
            "mae_v": mae_v,
            "rmse_v": rmse_v,
            "mae_speed": mae_speed,
            "rmse_speed": rmse_speed,
            "mae_direction_deg": mae_dir_deg,
        }

        logger.info(
            "Persistence Model Evaluation: "
            + ", ".join(
                [
                    f"mae_u={mae_u:.4f}",
                    f"rmse_u={rmse_u:.4f}",
                    f"mae_v={mae_v:.4f}",
                    f"rmse_v={rmse_v:.4f}",
                    f"mae_speed={mae_speed:.4f}",
                    f"rmse_speed={rmse_speed:.4f}",
                    f"mae_direction_deg={mae_dir_deg:.2f}°",
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
        Minimal preprocessing: handle -999 sentinels, compute u/v, resample.
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

        # Compute u, v from meteorological coming-from convention
        direction_rad = np.deg2rad(df["average_wind_direction"].astype(float) % 360)
        speed = df["average_wind_speed"].astype(float)
        df["u"] = -speed * np.sin(direction_rad)
        df["v"] = -speed * np.cos(direction_rad)

        # Resample to regular 10-minute grid per station
        freq = "10min"

        def resample_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("record_date").set_index("record_date")
            g = g.resample(freq).first()
            
            # Forward and backward fill for u, v
            g[["u", "v"]] = g[["u", "v"]].ffill().bfill()
            
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
            - X: [history_steps, 2] for [u, v]
            - y: [horizon_steps, 2] for [u, v]
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
                
                # Extract u, v
                x = hist[["u", "v"]].values
                y = future[["u", "v"]].values
                
                # Skip if any NaN
                if np.isnan(x).any() or np.isnan(y).any():
                    continue
                
                sequences.append((x, y))
        
        return sequences

