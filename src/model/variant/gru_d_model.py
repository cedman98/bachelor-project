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


class _GRUDDataset(Dataset):
    """
    Dataset for GRU-D that includes masks and time deltas for handling missing values.

    Each sample consists of:
    - x: [history_steps, 2] - u, v features (may contain NaN)
    - y: [horizon_steps, 2] - u, v targets
    - mask: [history_steps, 2] - binary mask (1=observed, 0=missing)
    - delta: [history_steps, 2] - time since last observation
    - x_last_obs: [history_steps, 2] - last observed values (forward-filled)
    - station_index: int - station embedding index
    """

    def __init__(
        self,
        sequences: List[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ],
    ) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x, y, mask, delta, x_last_obs, station_idx = self.sequences[idx]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(delta).float(),
            torch.from_numpy(x_last_obs).float(),
            torch.tensor(station_idx, dtype=torch.long),
        )


class _GRUDCell(nn.Module):
    """
    GRU-D cell that handles missing values using decay mechanisms.

    Reference: Che et al. "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Standard GRU gates
        self.w_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.w_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Decay parameters (learnable)
        self.gamma_x = nn.Parameter(torch.ones(input_dim))
        self.gamma_h = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        x_last_obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] - current input (may be NaN)
            mask: [batch, input_dim] - binary mask
            delta: [batch, input_dim] - time since last observation
            x_last_obs: [batch, input_dim] - last observed value
            h_prev: [batch, hidden_dim] - previous hidden state

        Returns:
            h: [batch, hidden_dim] - new hidden state
        """
        # Input decay: decay last observed value towards zero
        gamma_x_expanded = self.gamma_x.unsqueeze(0)  # [1, input_dim]
        x_decay = torch.exp(-torch.relu(delta * gamma_x_expanded))
        x_decayed = x_decay * x_last_obs

        # Replace NaN with decayed values
        x_filled = torch.where(mask.bool(), x, x_decayed)

        # Hidden state decay
        gamma_h_expanded = self.gamma_h.unsqueeze(0)  # [1, hidden_dim]
        h_decay = torch.exp(
            -torch.relu(delta.mean(dim=1, keepdim=True) * gamma_h_expanded)
        )
        h_decayed = h_decay * h_prev

        # Standard GRU computation with decayed states
        combined = torch.cat([x_filled, h_decayed], dim=1)

        r = torch.sigmoid(self.w_r(combined))
        z = torch.sigmoid(self.w_z(combined))

        combined_h = torch.cat([x_filled, r * h_decayed], dim=1)
        h_tilde = torch.tanh(self.w_h(combined_h))

        h = (1 - z) * h_decayed + z * h_tilde

        return h


class _GRUDHead(nn.Module):
    """
    GRU-D encoder with station embeddings for multi-step forecasting.
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Station embedding
        self.station_embedding = (
            nn.Embedding(station_count, station_embedding_dim)
            if station_count > 0 and station_embedding_dim > 0
            else None
        )

        effective_input_dim = input_dim + (
            station_embedding_dim if self.station_embedding else 0
        )

        # GRU-D cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_size = effective_input_dim if i == 0 else hidden_size
            self.cells.append(_GRUDCell(input_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size, horizon_steps * 2),  # 2 targets: u, v
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        x_last_obs: torch.Tensor,
        station_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len, input_dim]
            delta: [batch, seq_len, input_dim]
            x_last_obs: [batch, seq_len, input_dim]
            station_idx: [batch]

        Returns:
            predictions: [batch, horizon_steps, 2]
        """
        batch_size, seq_len, _ = x.shape

        # Add station embeddings
        if self.station_embedding is not None:
            emb = self.station_embedding(station_idx)  # [batch, emb_dim]
            emb_seq = emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, emb_seq], dim=-1)
            x_last_obs = torch.cat([x_last_obs, emb_seq], dim=-1)

        # Initialize hidden states
        h = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Process sequence through GRU-D layers
        for t in range(seq_len):
            x_t = x[:, t, :]
            mask_t = mask[:, t, :]
            delta_t = delta[:, t, :]
            x_last_t = x_last_obs[:, t, :]

            # First layer
            h[0] = self.cells[0](x_t, mask_t, delta_t, x_last_t, h[0])

            # Subsequent layers
            for layer_idx in range(1, self.num_layers):
                h_input = h[layer_idx - 1]
                if self.dropout is not None:
                    h_input = self.dropout(h_input)

                # For upper layers, create dummy mask/delta (all observed)
                mask_h = torch.ones_like(h_input)
                delta_h = torch.zeros_like(h_input)

                h[layer_idx] = self.cells[layer_idx](
                    h_input, mask_h, delta_h, h_input, h[layer_idx]
                )

        # Use final hidden state from last layer
        h_final = h[-1]

        # Project to output
        out = self.proj(h_final)  # [batch, horizon*2]
        return out.view(batch_size, -1, 2)


class GRUDModel(ModelInterface):
    """
    GRU-D model for multi-station, multi-step wind forecasting with missing value handling.

    - Input: 10-minute resolution time series with wind speed and direction
    - Targets: next 12 hours (72 steps) of [u, v] wind components
    - Features: only uses average_wind_speed and average_wind_direction
    - Missing values: handled explicitly via masks and decay mechanisms
    - History: 24 hours (144 steps) of past observations
    """

    def __init__(
        self,
        cfg: OmegaConf | None = None,
        history_steps: int = 144,
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
        val_split: float = 0.0,
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
        restore_best_weights: bool = True,
        shuffle_train: bool = True,
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

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.val_split = float(val_split)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.shuffle_train = bool(shuffle_train)

        # Will be initialized during training/loading
        self._model: _GRUDHead | None = None
        self._feature_columns: List[str] = ["u", "v"]
        self._target_columns: List[str] = ["u", "v"]
        self._feature_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._station_id_to_index: Dict[str, int] | None = None

    def train(self, dataset: pd.DataFrame) -> None:
        df = self._prepare_dataframe(dataset)

        if self.val_split and self.val_split > 0.0:
            train_sequences, val_sequences, num_stations = (
                self._build_sequences_train_val(df, self.val_split)
            )
        else:
            sequences, num_stations = self._build_sequences(df)
            train_sequences = sequences
            val_sequences = []

        if len(train_sequences) == 0:
            raise ValueError(
                "No training sequences could be constructed. Ensure sufficient history per station."
            )

        feature_dim = len(self._feature_columns)
        logger.info(
            f"Preparing GRU-D training: train_sequences={len(train_sequences)}, "
            f"val_sequences={len(val_sequences)}, stations={num_stations}, "
            f"feature_dim={feature_dim}, device={self.device}"
        )

        self._model = _GRUDHead(
            input_dim=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon_steps=self.horizon_steps,
            station_count=num_stations,
            station_embedding_dim=self.station_embedding_dim,
        ).to(self.device)

        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        logger.info(
            f"GRU-D model initialized: hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}, params={trainable_params:,}"
        )

        train_ds = _GRUDDataset(train_sequences)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        val_loader: DataLoader | None = None
        if len(val_sequences) > 0:
            val_ds = _GRUDDataset(val_sequences)
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device == "cuda"),
            )

        num_batches = len(train_loader)
        logger.info(
            f"DataLoader ready: train_samples={len(train_ds)}, "
            f"val_samples={len(val_sequences) if val_loader else 0}, "
            f"batch_size={self.batch_size}, batches_per_epoch={num_batches}"
        )

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA available: using GPU '{device_name}'")
            except Exception:
                logger.info("CUDA available: using GPU")
        elif self.device == "mps":
            logger.info("MPS available: using Apple Silicon GPU")

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state_dict: Dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0

        self._model.train()
        for epoch in range(self.num_epochs):
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} starting "
                f"(lr={optimizer.param_groups[0]['lr']:.3e})"
            )
            epoch_loss = 0.0
            log_every = max(1, num_batches // 10) if num_batches > 0 else 1

            for batch_idx, (xb, yb, mask_b, delta_b, x_last_b, sb) in enumerate(
                train_loader, start=1
            ):
                non_block = self.device == "cuda"
                xb = xb.to(self.device, non_blocking=non_block)
                yb = yb.to(self.device, non_blocking=non_block)
                mask_b = mask_b.to(self.device, non_blocking=non_block)
                delta_b = delta_b.to(self.device, non_blocking=non_block)
                x_last_b = x_last_b.to(self.device, non_blocking=non_block)
                sb = sb.to(self.device, non_blocking=non_block)

                optimizer.zero_grad(set_to_none=True)
                preds = self._model(xb, mask_b, delta_b, x_last_b, sb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item()) * xb.size(0)

                if batch_idx % log_every == 0 or batch_idx == num_batches:
                    logger.info(
                        f"Epoch {epoch + 1} [{batch_idx}/{num_batches}] "
                        f"batch_loss={loss.item():.6f}"
                    )

            epoch_loss /= len(train_ds)
            logger.info(
                f"GRU-D epoch {epoch + 1}/{self.num_epochs} - loss={epoch_loss:.6f}"
            )

            # Validation and early stopping
            if val_loader is not None:
                self._model.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb, mask_b, delta_b, x_last_b, sb in val_loader:
                        non_block = self.device == "cuda"
                        xb = xb.to(self.device, non_blocking=non_block)
                        yb = yb.to(self.device, non_blocking=non_block)
                        mask_b = mask_b.to(self.device, non_blocking=non_block)
                        delta_b = delta_b.to(self.device, non_blocking=non_block)
                        x_last_b = x_last_b.to(self.device, non_blocking=non_block)
                        sb = sb.to(self.device, non_blocking=non_block)

                        preds = self._model(xb, mask_b, delta_b, x_last_b, sb)
                        vloss = loss_fn(preds, yb)
                        val_total += float(vloss.detach().cpu().item()) * xb.size(0)
                        val_count += xb.size(0)

                val_loss = val_total / max(1, val_count)
                logger.info(
                    f"GRU-D epoch {epoch + 1}/{self.num_epochs} - val_loss={val_loss:.6f}"
                )

                improved = (best_val_loss - val_loss) > self.early_stopping_min_delta
                if improved:
                    best_val_loss = val_loss
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if (
                        self.early_stopping_patience > 0
                        and epochs_no_improve >= self.early_stopping_patience
                    ):
                        logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}: "
                            f"no improvement in {self.early_stopping_patience} epoch(s)."
                        )
                        break
                self._model.train()

        # Restore best weights
        if best_state_dict is not None and self.restore_best_weights:
            self._model.load_state_dict(best_state_dict)
            logger.info("Restored best model weights from validation")

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

            # Prepare GRU-D inputs
            x_np, mask_np, delta_np, x_last_np = self._prepare_grud_inputs(
                hist[self._feature_columns].to_numpy(dtype=float)
            )

            non_block = self.device == "cuda"
            x = (
                torch.from_numpy(x_np)
                .float()
                .unsqueeze(0)
                .to(self.device, non_blocking=non_block)
            )
            mask = (
                torch.from_numpy(mask_np)
                .float()
                .unsqueeze(0)
                .to(self.device, non_blocking=non_block)
            )
            delta = (
                torch.from_numpy(delta_np)
                .float()
                .unsqueeze(0)
                .to(self.device, non_blocking=non_block)
            )
            x_last = (
                torch.from_numpy(x_last_np)
                .float()
                .unsqueeze(0)
                .to(self.device, non_blocking=non_block)
            )

            # Map station to known index
            mapped_idx = 0
            if self._station_id_to_index is not None:
                mapped_idx = int(self._station_id_to_index.get(str(station_id), 0))
            sid_idx = torch.tensor([mapped_idx], dtype=torch.long).to(
                self.device, non_blocking=non_block
            )

            self._model.eval()
            with torch.no_grad():
                y_scaled = (
                    self._model(x, mask, delta, x_last, sid_idx)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            # Inverse scale targets
            y_flat = y_scaled.reshape(-1, 2)
            y_inv = self._target_scaler.inverse_transform(y_flat)
            y_inv = y_inv.reshape(self.horizon_steps, 2)

            # Build timestamps for forecast horizon
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

    def evaluate(self, dataset: pd.DataFrame) -> Dict[str, float]:
        if self._model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a saved model."
            )

        df = self._prepare_dataframe(dataset, fit_scalers=False)
        sequences, _ = self._build_sequences(df)
        if len(sequences) == 0:
            raise ValueError("No evaluation sequences could be constructed.")

        eval_ds = _GRUDDataset(sequences)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        logger.info(f"Evaluation: samples={len(eval_ds)}, batch_size={self.batch_size}")

        self._model.eval()
        non_block = self.device == "cuda"

        preds_u: List[np.ndarray] = []
        preds_v: List[np.ndarray] = []
        trues_u: List[np.ndarray] = []
        trues_v: List[np.ndarray] = []

        with torch.no_grad():
            for xb, yb, mask_b, delta_b, x_last_b, sb in eval_loader:
                xb = xb.to(self.device, non_blocking=non_block)
                mask_b = mask_b.to(self.device, non_blocking=non_block)
                delta_b = delta_b.to(self.device, non_blocking=non_block)
                x_last_b = x_last_b.to(self.device, non_blocking=non_block)
                sb = sb.to(self.device, non_blocking=non_block)

                y_pred_scaled = (
                    self._model(xb, mask_b, delta_b, x_last_b, sb).cpu().numpy()
                )
                y_true_scaled = yb.cpu().numpy()

                # Inverse scale
                B, H, _ = y_pred_scaled.shape
                y_pred_flat = y_pred_scaled.reshape(-1, 2)
                y_true_flat = y_true_scaled.reshape(-1, 2)
                y_pred = self._target_scaler.inverse_transform(y_pred_flat).reshape(
                    B, H, 2
                )
                y_true = self._target_scaler.inverse_transform(y_true_flat).reshape(
                    B, H, 2
                )

                preds_u.append(y_pred[:, :, 0].reshape(-1))
                preds_v.append(y_pred[:, :, 1].reshape(-1))
                trues_u.append(y_true[:, :, 0].reshape(-1))
                trues_v.append(y_true[:, :, 1].reshape(-1))

        u_pred = np.concatenate(preds_u)
        v_pred = np.concatenate(preds_v)
        u_true = np.concatenate(trues_u)
        v_true = np.concatenate(trues_v)

        # Metrics for u, v
        mae_u = float(np.mean(np.abs(u_pred - u_true)))
        rmse_u = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
        mae_v = float(np.mean(np.abs(v_pred - v_true)))
        rmse_v = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))

        # Derived speed metrics
        speed_pred = np.sqrt(u_pred**2 + v_pred**2)
        speed_true = np.sqrt(u_true**2 + v_true**2)
        mae_speed = float(np.mean(np.abs(speed_pred - speed_true)))
        rmse_speed = float(np.sqrt(np.mean((speed_pred - speed_true) ** 2)))

        # Direction error
        def uv_to_dir_deg(u_arr: np.ndarray, v_arr: np.ndarray) -> np.ndarray:
            ang = np.degrees(np.arctan2(-u_arr, -v_arr))
            return np.mod(ang, 360.0)

        def angular_mae_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
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
            "GRU-D evaluation metrics: "
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
            "val_split": self.val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "restore_best_weights": self.restore_best_weights,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "num_workers": self.num_workers,
            "shuffle_train": self.shuffle_train,
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
        self.val_split = float(metadata.get("val_split", 0.0))
        self.early_stopping_patience = int(metadata.get("early_stopping_patience", 0))
        self.early_stopping_min_delta = float(
            metadata.get("early_stopping_min_delta", 0.0)
        )
        self.restore_best_weights = bool(metadata.get("restore_best_weights", True))
        self.batch_size = int(metadata.get("batch_size", self.batch_size))
        self.learning_rate = float(metadata.get("learning_rate", self.learning_rate))
        self.num_epochs = int(metadata.get("num_epochs", self.num_epochs))
        self.num_workers = int(metadata.get("num_workers", self.num_workers))
        self.shuffle_train = bool(metadata.get("shuffle_train", self.shuffle_train))
        self._feature_columns = list(metadata["feature_columns"])
        self._target_columns = list(metadata["target_columns"])
        self._station_id_to_index = dict(metadata["station_id_to_index"])
        self._feature_scaler = metadata["feature_scaler"]
        self._target_scaler = metadata["target_scaler"]

        feature_dim = len(self._feature_columns)
        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )

        self._model = _GRUDHead(
            input_dim=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon_steps=self.horizon_steps,
            station_count=station_count,
            station_embedding_dim=self.station_embedding_dim,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    # ------------- Internal helpers -------------
    def _prepare_dataframe(
        self, dataset: pd.DataFrame, fit_scalers: bool = True
    ) -> pd.DataFrame:
        """
        Prepare dataframe with only wind features.
        Keep -999 for now to create masks later.
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

        df = dataset[required_cols].copy()

        if not pd.api.types.is_datetime64_any_dtype(df["record_date"]):
            df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")

        # Create u, v targets (meteorological convention)
        direction_rad = np.deg2rad(df["average_wind_direction"].astype(float) % 360)
        speed = df["average_wind_speed"].astype(float)
        df["u"] = -speed * np.sin(direction_rad)
        df["v"] = -speed * np.cos(direction_rad)

        # Sort by station and time
        df = df.sort_values(["station_id", "record_date"]).reset_index(drop=True)

        # Station mapping
        if fit_scalers or self._station_id_to_index is None:
            station_ids = [str(s) for s in pd.unique(df["station_id"])]
            self._station_id_to_index = {sid: i for i, sid in enumerate(station_ids)}

        # Fit or apply scalers (excluding -999 values)
        if fit_scalers or self._feature_scaler is None or self._target_scaler is None:
            self._feature_scaler = StandardScaler()
            self._target_scaler = StandardScaler()

            # Fit on non-missing values only
            valid_mask = (df["u"] != -999) & (df["v"] != -999)
            if valid_mask.sum() > 0:
                self._feature_scaler.fit(
                    df.loc[valid_mask, ["u", "v"]].to_numpy(dtype=float)
                )
                self._target_scaler.fit(
                    df.loc[valid_mask, ["u", "v"]].to_numpy(dtype=float)
                )
            else:
                raise ValueError("No valid data to fit scalers")

        # Apply scalers (will handle -999 in sequence building)
        df[["u", "v"]] = self._feature_scaler.transform(
            df[["u", "v"]].to_numpy(dtype=float)
        )

        return df

    def _prepare_grud_inputs(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare GRU-D specific inputs from feature array.

        Returns:
            x: original features (with NaN for missing)
            mask: binary mask (1=observed, 0=missing)
            delta: time since last observation
            x_last_obs: last observed values (forward-filled)
        """
        # Create mask (1 = observed, 0 = missing)
        # Assume -999 scaled will be a large negative value
        mask = ~np.isnan(x)

        # Replace NaN with 0 for computation
        x_filled = np.nan_to_num(x, nan=0.0)

        # Compute time deltas and last observed values
        seq_len, feat_dim = x.shape
        delta = np.zeros_like(x)
        x_last_obs = np.zeros_like(x)

        for feat in range(feat_dim):
            last_obs_val = 0.0
            time_since_obs = 0.0

            for t in range(seq_len):
                if mask[t, feat]:
                    last_obs_val = x_filled[t, feat]
                    time_since_obs = 0.0
                else:
                    time_since_obs += 1.0

                x_last_obs[t, feat] = last_obs_val
                delta[t, feat] = time_since_obs

        return x_filled, mask.astype(float), delta, x_last_obs

    def _build_sequences(self, df: pd.DataFrame) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
        int,
    ]:
        sequences = []
        hist = self.history_steps
        horiz = self.horizon_steps

        for station_id, g in df.groupby("station_id", sort=False):
            g = g.sort_values("record_date")
            features = g[self._feature_columns].to_numpy(dtype=float)
            targets = g[["u", "v"]].to_numpy(dtype=float)
            n = len(g)

            if n < hist + horiz:
                continue

            sid_idx = self._station_id_to_index[str(station_id)]

            for start in range(0, n - hist - horiz + 1):
                x = features[start : start + hist]
                y_raw = targets[start + hist : start + hist + horiz]

                # Scale targets
                y = self._target_scaler.transform(y_raw)

                # Prepare GRU-D inputs
                x_proc, mask, delta, x_last_obs = self._prepare_grud_inputs(x)

                sequences.append((x_proc, y, mask, delta, x_last_obs, sid_idx))

        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )
        return sequences, station_count

    def _build_sequences_train_val(
        self,
        df: pd.DataFrame,
        val_fraction: float,
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
        int,
    ]:
        train_sequences = []
        val_sequences = []
        hist = self.history_steps
        horiz = self.horizon_steps

        for station_id, g in df.groupby("station_id", sort=False):
            g = g.sort_values("record_date")
            features = g[self._feature_columns].to_numpy(dtype=float)
            targets = g[["u", "v"]].to_numpy(dtype=float)
            n = len(g)

            if n < hist + horiz:
                continue

            sid_idx = self._station_id_to_index[str(station_id)]

            cut = int(np.floor(n * (1.0 - float(val_fraction))))
            cut = int(np.clip(cut, 0, n))

            # Training sequences
            max_train_start = cut - hist - horiz
            if max_train_start >= 0:
                for start in range(0, max_train_start + 1):
                    x = features[start : start + hist]
                    y_raw = targets[start + hist : start + hist + horiz]
                    y = self._target_scaler.transform(y_raw)

                    x_proc, mask, delta, x_last_obs = self._prepare_grud_inputs(x)
                    train_sequences.append(
                        (x_proc, y, mask, delta, x_last_obs, sid_idx)
                    )

            # Validation sequences
            min_val_start = max(0, cut - hist)
            max_val_start = n - hist - horiz
            if max_val_start >= min_val_start:
                for start in range(min_val_start, max_val_start + 1):
                    if start + hist < cut:
                        continue
                    x = features[start : start + hist]
                    y_raw = targets[start + hist : start + hist + horiz]
                    y = self._target_scaler.transform(y_raw)

                    x_proc, mask, delta, x_last_obs = self._prepare_grud_inputs(x)
                    val_sequences.append((x_proc, y, mask, delta, x_last_obs, sid_idx))

        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )
        return train_sequences, val_sequences, station_count
