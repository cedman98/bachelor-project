import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.variant.model_interface import ModelInterface


class _SequenceDataset(Dataset):
    """
    Builds sliding-window sequences per station for supervised learning.

    Each sample consists of:
    - X: [history_steps, feature_dim]
    - y: [horizon_steps, 2] where the two targets are [u, v]
    - station_index: int index for an embedding lookup
    """

    def __init__(
        self,
        sequences: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, station_idx = self.sequences[idx]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(station_idx, dtype=torch.long),
        )


class _BiLSTMHead(nn.Module):
    """
    A bidirectional LSTM encoder that maps a history window to a fixed representation,
    followed by a feedforward projection to a multi-step horizon for two targets (u, v).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon_steps: int,
        station_count: int,
        station_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.station_embedding = (
            nn.Embedding(station_count, station_embedding_dim)
            if station_count > 0 and station_embedding_dim > 0
            else None
        )
        effective_input_dim = input_dim + (
            station_embedding_dim if self.station_embedding else 0
        )

        self.lstm = nn.LSTM(
            input_size=effective_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Use the final layer's forward/backward hidden states (concatenated)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, horizon_steps * 2),  # 2 targets: u & v
        )

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, history_steps, input_dim]
            station_idx: [batch]

        Returns:
            predictions: [batch, horizon_steps, 2]
        """
        if self.station_embedding is not None:
            emb = self.station_embedding(station_idx)  # [batch, emb_dim]
            # Repeat embedding across timesteps and concat to features
            emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb_seq], dim=-1)

        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers * num_directions, batch, hidden_size]
        # Take the last layer's forward and backward states
        # Indexing last two slices corresponds to the final layer's two directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=-1)  # [batch, hidden*2]
        out = self.proj(h)  # [batch, horizon*2]
        return out.view(x.size(0), -1, 2)


class BiLSTMModel(ModelInterface):
    """
    Bidirectional LSTM model for multi-station, multi-step forecasting of wind vector components.

    - Input data: 10-minute resolution time series with columns defined by the user.
    - Targets: next 12 hours (72 steps) of [u, v] where
      u = -speed * sin(direction_rad), v = -speed * cos(direction_rad)
      using meteorological convention (coming-from direction).
    - Handles multiple stations via a learned station embedding.
    - Drops `quality_level`.
    - Replaces numeric sentinel -999 with NaN and imputes within-station via forward/backward fill.
    - Standardizes features and targets; scalers are persisted with the model.
    """

    def __init__(
        self,
        cfg: OmegaConf | None = None,
        history_steps: int = 72,
        horizon_steps: int = 72,
        station_embedding_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        num_workers: int = 0,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.station_embedding_dim = station_embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be initialized during training/loading
        self._model: _BiLSTMHead | None = None
        self._feature_columns: List[str] | None = None
        self._target_columns: List[str] = ["u", "v"]
        self._feature_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._station_id_to_index: Dict[str, int] | None = None

    # ------------- Public API -------------
    def train(self, dataset: pd.DataFrame) -> None:
        df = self._prepare_dataframe(dataset)
        sequences, num_stations = self._build_sequences(df)

        if len(sequences) == 0:
            raise ValueError(
                "No training sequences could be constructed. Ensure sufficient history per station."
            )

        feature_dim = len(self._feature_columns)
        self._model = _BiLSTMHead(
            input_dim=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon_steps=self.horizon_steps,
            station_count=num_stations,
            station_embedding_dim=self.station_embedding_dim,
        ).to(self.device)

        train_ds = _SequenceDataset(sequences)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self._model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for xb, yb, sb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                sb = sb.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                preds = self._model(xb, sb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item()) * xb.size(0)

            epoch_loss /= len(train_ds)
            logger.info(
                f"BiLSTM epoch {epoch + 1}/{self.num_epochs} - loss={epoch_loss:.6f}"
            )

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a saved model."
            )

        df = self._prepare_dataframe(dataset, fit_scalers=False)
        results: List[pd.DataFrame] = []
        freq = pd.Timedelta(minutes=10)

        for station_id, g in df.groupby("station_id", sort=False):
            g = g.sort_values("record_date")
            if len(g) < self.history_steps:
                continue

            hist = g.tail(self.history_steps)
            x = hist[self._feature_columns].to_numpy(dtype=float)
            x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)

            # Map station to known index; fallback to 0 for unseen stations
            mapped_idx = 0
            if self._station_id_to_index is not None:
                mapped_idx = int(self._station_id_to_index.get(str(station_id), 0))
            sid_idx = torch.tensor([mapped_idx], dtype=torch.long, device=self.device)

            self._model.eval()
            with torch.no_grad():
                y_scaled = self._model(x, sid_idx).squeeze(0).cpu().numpy()

            # Inverse scale targets
            y_flat = y_scaled.reshape(-1, 2)
            y_inv = self._target_scaler.inverse_transform(y_flat)
            y_inv = y_inv.reshape(self.horizon_steps, 2)

            # Build timestamps for the forecast horizon
            last_ts = pd.to_datetime(hist.iloc[-1]["record_date"])
            future_index = [last_ts + freq * (i + 1) for i in range(self.horizon_steps)]

            out_df = pd.DataFrame(
                {
                    "station_id": station_id,
                    "record_date": future_index,
                    "u_pred": y_inv[:, 0],
                    "v_pred": y_inv[:, 1],
                }
            )
            results.append(out_df)

        if not results:
            raise ValueError("No stations had sufficient history for prediction.")

        return pd.concat(results, axis=0).reset_index(drop=True)

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))

        metadata = {
            "history_steps": self.history_steps,
            "horizon_steps": self.horizon_steps,
            "station_embedding_dim": self.station_embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "feature_columns": self._feature_columns,
            "target_columns": self._target_columns,
            "station_id_to_index": self._station_id_to_index,
            "feature_scaler": self._feature_scaler,
            "target_scaler": self._target_scaler,
        }
        joblib.dump(metadata, os.path.join(path, "metadata.joblib"))

    def load(self, path: str) -> None:
        meta_path = os.path.join(path, "metadata.joblib")
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model files in {path}")

        metadata = joblib.load(meta_path)
        self.history_steps = int(metadata["history_steps"])
        self.horizon_steps = int(metadata["horizon_steps"])
        self.station_embedding_dim = int(metadata["station_embedding_dim"])
        self.hidden_size = int(metadata["hidden_size"])
        self.num_layers = int(metadata["num_layers"])
        self.dropout = float(metadata["dropout"])
        self._feature_columns = list(metadata["feature_columns"])  # type: ignore[arg-type]
        self._target_columns = list(metadata["target_columns"])  # type: ignore[arg-type]
        self._station_id_to_index = dict(metadata["station_id_to_index"])  # type: ignore[arg-type]
        self._feature_scaler = metadata["feature_scaler"]
        self._target_scaler = metadata["target_scaler"]

        feature_dim = len(self._feature_columns)
        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )
        self._model = _BiLSTMHead(
            input_dim=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon_steps=self.horizon_steps,
            station_count=station_count,
            station_embedding_dim=self.station_embedding_dim,
        ).to(self.device)
        self._model.load_state_dict(torch.load(model_path, map_location=self.device))
        self._model.eval()

    # ------------- Internal helpers -------------
    def _prepare_dataframe(
        self, dataset: pd.DataFrame, fit_scalers: bool = True
    ) -> pd.DataFrame:
        """
        - Enforce dtypes and sort order.
        - Drop `quality_level` if present.
        - Create targets u, v from average_wind_speed/direction.
        - Handle -999 -> NaN and impute within station.
        - Fit/apply scalers on features and targets.
        """
        required_cols = [
            "station_id",
            "record_date",
            "average_wind_speed",
            "average_wind_direction",
            "air_pressure",
            "air_temperature_2m",
            "air_temperature_5cm",
            "relative_humidity",
            "dew_point_temperature",
            "precipitation_duration",
            "sum_precipitation_height",
            "precipitation_indicator",
        ]

        missing = [c for c in required_cols if c not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df = dataset.copy()
        if "quality_level" in df.columns:
            df = df.drop(columns=["quality_level"])

        if not pd.api.types.is_datetime64_any_dtype(df["record_date"]):
            df["record_date"] = pd.to_datetime(
                df["record_date"], errors="coerce", utc=False
            )

        # Create targets u, v (meteorological convention: coming-from)
        # u = -speed * sin(dir), v = -speed * cos(dir)
        direction_rad = np.deg2rad(df["average_wind_direction"].astype(float) % 360)
        speed = df["average_wind_speed"].astype(float)
        df["u"] = -speed * np.sin(direction_rad)
        df["v"] = -speed * np.cos(direction_rad)

        # Replace -999 with NaN in numeric columns
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        if "station_id" in numeric_cols:
            numeric_cols.remove("station_id")
        for col in numeric_cols:
            df[col] = df[col].replace(-999, np.nan)

        # Sort by station and time
        df = df.sort_values(["station_id", "record_date"]).reset_index(drop=True)

        # Impute missing values within each station by forward/backward fill, then global fill
        def _impute_station(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            g[numeric_cols] = g[numeric_cols].ffill().bfill()
            return g

        df = df.groupby("station_id", sort=False, group_keys=False).apply(
            _impute_station
        )
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].mean(numeric_only=True)
        )

        # Select features (exclude identifiers and raw direction/speed; keep derived u,v in targets only)
        feature_cols = [
            "air_pressure",
            "air_temperature_2m",
            "air_temperature_5cm",
            "relative_humidity",
            "dew_point_temperature",
            "precipitation_duration",
            "sum_precipitation_height",
            "precipitation_indicator",
            "u",
            "v",
        ]
        self._feature_columns = (
            feature_cols if self._feature_columns is None else self._feature_columns
        )

        # Station mapping (string-ify to be robust)
        if fit_scalers or self._station_id_to_index is None:
            station_ids = [str(s) for s in pd.unique(df["station_id"])]
            self._station_id_to_index = {sid: i for i, sid in enumerate(station_ids)}

        # Fit or apply scalers
        if fit_scalers or self._feature_scaler is None or self._target_scaler is None:
            self._feature_scaler = StandardScaler()
            self._target_scaler = StandardScaler()
            # Fit on available rows
            self._feature_scaler.fit(df[self._feature_columns].to_numpy(dtype=float))
            self._target_scaler.fit(df[["u", "v"]].to_numpy(dtype=float))

        # Apply scalers (store scaled copies; keep original df for timestamps and ids)
        df[self._feature_columns] = self._feature_scaler.transform(
            df[self._feature_columns].to_numpy(dtype=float)
        )
        df[["u", "v"]] = self._target_scaler.transform(
            df[["u", "v"]].to_numpy(dtype=float)
        )

        return df

    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], int]:
        sequences: List[Tuple[np.ndarray, np.ndarray, int]] = []
        hist = self.history_steps
        horiz = self.horizon_steps

        for station_id, g in df.groupby("station_id", sort=False):
            g = g.sort_values("record_date")
            features = g[self._feature_columns].to_numpy(dtype=float)
            targets = g[["u", "v"]].to_numpy(dtype=float)
            n = len(g)
            if n < hist + horiz:
                continue
            sid_idx = self._station_id_to_index[str(station_id)]  # type: ignore[index]

            for start in range(0, n - hist - horiz + 1):
                x = features[start : start + hist]
                y = targets[start + hist : start + hist + horiz]
                sequences.append((x, y, sid_idx))

        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )
        return sequences, station_count
