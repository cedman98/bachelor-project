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
        # Auto-detect best available device: CUDA > MPS > CPU
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
        self._model: _BiLSTMHead | None = None
        self._feature_columns: List[str] | None = None
        self._target_columns: List[str] = ["u", "v"]
        self._feature_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._station_id_to_index: Dict[str, int] | None = None

    # ------------- Public API -------------
    def train(self, dataset: pd.DataFrame) -> None:
        df = self._prepare_dataframe(dataset)
        train_sequences: List[Tuple[np.ndarray, np.ndarray, int]]
        val_sequences: List[Tuple[np.ndarray, np.ndarray, int]]
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
            f"Preparing training: train_sequences={len(train_sequences)}, val_sequences={len(val_sequences)}, stations={num_stations}, "
            f"feature_dim={feature_dim}, device={self.device}"
        )
        self._model = _BiLSTMHead(
            input_dim=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon_steps=self.horizon_steps,
            station_count=num_stations,
            station_embedding_dim=self.station_embedding_dim,
        ).to(self.device)

        # Log model parameter count
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model initialized: hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
            f"dropout={self.dropout}, params={trainable_params:,}"
        )

        train_ds = _SequenceDataset(train_sequences)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        val_loader: DataLoader | None = None
        if len(val_sequences) > 0:
            val_ds = _SequenceDataset(val_sequences)
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device == "cuda"),
            )

        num_batches = len(train_loader)
        logger.info(
            f"DataLoader ready: train_samples={len(train_ds)}, val_samples={(len(val_sequences) if val_loader is not None else 0)}, batch_size={self.batch_size}, batches_per_epoch={num_batches}"
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
                f"Epoch {epoch + 1}/{self.num_epochs} starting (lr={optimizer.param_groups[0]['lr']:.3e})"
            )
            epoch_loss = 0.0
            # Log ~10 times per epoch
            log_every = max(1, num_batches // 10) if num_batches > 0 else 1
            for batch_idx, (xb, yb, sb) in enumerate(train_loader, start=1):
                # Use non_blocking transfers only for CUDA. On MPS, non_blocking and
                # pin_memory can cause device mismatch errors.
                non_block = self.device == "cuda"
                xb = xb.to(self.device, non_blocking=non_block)
                yb = yb.to(self.device, non_blocking=non_block)
                sb = sb.to(self.device, non_blocking=non_block)

                optimizer.zero_grad(set_to_none=True)
                preds = self._model(xb, sb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item()) * xb.size(0)

                if batch_idx % log_every == 0 or batch_idx == num_batches:
                    logger.info(
                        f"Epoch {epoch + 1} [{batch_idx}/{num_batches}] batch_loss={loss.item():.6f}"
                    )

            epoch_loss /= len(train_ds)
            logger.info(
                f"BiLSTM epoch {epoch + 1}/{self.num_epochs} - loss={epoch_loss:.6f}"
            )

            # Validation and early stopping
            if val_loader is not None:
                self._model.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb, sb in val_loader:
                        non_block = self.device == "cuda"
                        xb = xb.to(self.device, non_blocking=non_block)
                        yb = yb.to(self.device, non_blocking=non_block)
                        sb = sb.to(self.device, non_blocking=non_block)
                        preds = self._model(xb, sb)
                        vloss = loss_fn(preds, yb)
                        val_total += float(vloss.detach().cpu().item()) * xb.size(0)
                        val_count += xb.size(0)
                val_loss = val_total / max(1, val_count)
                logger.info(
                    f"BiLSTM epoch {epoch + 1}/{self.num_epochs} - val_loss={val_loss:.6f}"
                )

                improved = (best_val_loss - val_loss) > self.early_stopping_min_delta
                if improved:
                    best_val_loss = val_loss
                    # Deep copy state dict to CPU tensors
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
                            f"Early stopping triggered at epoch {epoch + 1}: no improvement in {self.early_stopping_patience} epoch(s)."
                        )
                        break
                self._model.train()

        # Optionally restore best weights
        if best_state_dict is not None and self.restore_best_weights:
            self._model.load_state_dict(best_state_dict)

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
            x_np = hist[self._feature_columns].to_numpy(dtype=float)
            # Only enable pinned memory + non_blocking for CUDA to avoid MPS issues
            non_block = self.device == "cuda" and torch.cuda.is_available()
            x_t = torch.from_numpy(x_np).float()
            if non_block:
                x_t = x_t.pin_memory()
            x = x_t.unsqueeze(0).to(self.device, non_blocking=non_block)

            # Map station to known index; fallback to 0 for unseen stations
            mapped_idx = 0
            if self._station_id_to_index is not None:
                mapped_idx = int(self._station_id_to_index.get(str(station_id), 0))
            sid_cpu = torch.tensor([mapped_idx], dtype=torch.long)
            if non_block:
                sid_cpu = sid_cpu.pin_memory()
            sid_idx = sid_cpu.to(self.device, non_blocking=non_block)

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

    def evaluate(
        self, dataset: pd.DataFrame, max_batches: int | None = None
    ) -> Dict[str, float]:
        if self._model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a saved model."
            )

        if self._feature_scaler is None or self._target_scaler is None:
            raise ValueError(
                "Scalers not available. Train or load a model with scalers."
            )

        df = self._prepare_dataframe(dataset, fit_scalers=False)
        sequences, _ = self._build_sequences(df)
        if len(sequences) == 0:
            raise ValueError(
                "No evaluation sequences could be constructed. Ensure sufficient history per station."
            )

        eval_ds = _SequenceDataset(sequences)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        num_batches = len(eval_loader)
        logger.info(
            f"Evaluation: samples={len(eval_ds)}, batch_size={self.batch_size}, batches={num_batches}"
        )

        self._model.eval()
        non_block = self.device == "cuda" and torch.cuda.is_available()

        preds_u: List[np.ndarray] = []
        preds_v: List[np.ndarray] = []
        trues_u: List[np.ndarray] = []
        trues_v: List[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, (xb, yb, sb) in enumerate(eval_loader, start=1):
                xb = xb.to(self.device, non_blocking=non_block)
                sb = sb.to(self.device, non_blocking=non_block)

                y_pred_scaled = self._model(xb, sb).cpu().numpy()  # [B, H, 2]
                y_true_scaled = yb.cpu().numpy()  # [B, H, 2]

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

                if max_batches is not None and batch_idx >= max_batches:
                    break

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
            "Evaluation metrics: "
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

    def evaluate_per_horizon(
        self,
        dataset: pd.DataFrame,
        save_dir: str | None = None,
        max_batches: int | None = None,
    ) -> Dict[str, List[float]]:
        if self._model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a saved model."
            )

        if self._feature_scaler is None or self._target_scaler is None:
            raise ValueError(
                "Scalers not available. Train or load a model with scalers."
            )

        df = self._prepare_dataframe(dataset, fit_scalers=False)
        sequences, _ = self._build_sequences(df)
        if len(sequences) == 0:
            raise ValueError(
                "No evaluation sequences could be constructed. Ensure sufficient history per station."
            )

        eval_ds = _SequenceDataset(sequences)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        H = self.horizon_steps
        sum_abs_u = np.zeros(H, dtype=float)
        sum_sq_u = np.zeros(H, dtype=float)
        sum_abs_v = np.zeros(H, dtype=float)
        sum_sq_v = np.zeros(H, dtype=float)
        sum_abs_speed = np.zeros(H, dtype=float)
        sum_sq_speed = np.zeros(H, dtype=float)
        sum_abs_dir = np.zeros(H, dtype=float)
        count = np.zeros(H, dtype=float)

        self._model.eval()
        non_block = self.device == "cuda" and torch.cuda.is_available()

        def uv_to_dir_deg(u_arr: np.ndarray, v_arr: np.ndarray) -> np.ndarray:
            ang = np.degrees(np.arctan2(-u_arr, -v_arr))
            return np.mod(ang, 360.0)

        with torch.no_grad():
            for batch_idx, (xb, yb, sb) in enumerate(eval_loader, start=1):
                xb = xb.to(self.device, non_blocking=non_block)
                sb = sb.to(self.device, non_blocking=non_block)

                y_pred_scaled = self._model(xb, sb).cpu().numpy()  # [B, H, 2]
                y_true_scaled = yb.cpu().numpy()  # [B, H, 2]

                B, H_, _ = y_pred_scaled.shape
                assert H_ == H

                # Inverse scale
                y_pred = self._target_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 2)
                ).reshape(B, H, 2)
                y_true = self._target_scaler.inverse_transform(
                    y_true_scaled.reshape(-1, 2)
                ).reshape(B, H, 2)

                u_p = y_pred[:, :, 0]
                v_p = y_pred[:, :, 1]
                u_t = y_true[:, :, 0]
                v_t = y_true[:, :, 1]

                # u, v errors
                abs_u = np.abs(u_p - u_t).sum(axis=0)
                sq_u = ((u_p - u_t) ** 2).sum(axis=0)
                abs_v = np.abs(v_p - v_t).sum(axis=0)
                sq_v = ((v_p - v_t) ** 2).sum(axis=0)

                # speed errors
                sp_p = np.sqrt(u_p**2 + v_p**2)
                sp_t = np.sqrt(u_t**2 + v_t**2)
                abs_sp = np.abs(sp_p - sp_t).sum(axis=0)
                sq_sp = ((sp_p - sp_t) ** 2).sum(axis=0)

                # direction absolute error (degrees) with circular difference
                dir_p = uv_to_dir_deg(u_p, v_p)
                dir_t = uv_to_dir_deg(u_t, v_t)
                diff_dir = np.abs(((dir_p - dir_t + 180.0) % 360.0) - 180.0).sum(axis=0)

                sum_abs_u += abs_u
                sum_sq_u += sq_u
                sum_abs_v += abs_v
                sum_sq_v += sq_v
                sum_abs_speed += abs_sp
                sum_sq_speed += sq_sp
                sum_abs_dir += diff_dir
                count += float(B)

                if max_batches is not None and batch_idx >= max_batches:
                    break

        # Avoid division by zero
        count = np.maximum(count, 1.0)

        mae_u = (sum_abs_u / count).tolist()
        rmse_u = np.sqrt(sum_sq_u / count).tolist()
        mae_v = (sum_abs_v / count).tolist()
        rmse_v = np.sqrt(sum_sq_v / count).tolist()
        mae_speed = (sum_abs_speed / count).tolist()
        rmse_speed = np.sqrt(sum_sq_speed / count).tolist()
        mae_direction_deg = (sum_abs_dir / count).tolist()

        metrics: Dict[str, List[float]] = {
            "mae_u": mae_u,
            "rmse_u": rmse_u,
            "mae_v": mae_v,
            "rmse_v": rmse_v,
            "mae_speed": mae_speed,
            "rmse_speed": rmse_speed,
            "mae_direction_deg": mae_direction_deg,
        }

        logger.info("Computed per-horizon metrics for 1..%d steps", H)

        # Optional plotting
        if save_dir is not None:
            try:
                import importlib

                os.makedirs(save_dir, exist_ok=True)
                plt = importlib.import_module("matplotlib.pyplot")
                horizons = np.arange(1, H + 1)

                # Plot u, v, speed MAE/RMSE
                fig1 = plt.figure(figsize=(10, 6))
                plt.plot(horizons, mae_u, label="MAE u")
                plt.plot(horizons, rmse_u, label="RMSE u")
                plt.plot(horizons, mae_v, label="MAE v")
                plt.plot(horizons, rmse_v, label="RMSE v")
                plt.plot(horizons, mae_speed, label="MAE speed")
                plt.plot(horizons, rmse_speed, label="RMSE speed")
                plt.xlabel("Horizon (steps, 10-min)")
                plt.ylabel("Error")
                plt.title("Per-horizon errors: u, v, speed")
                plt.grid(True, alpha=0.3)
                plt.legend()
                fig1.tight_layout()
                fig1.savefig(
                    os.path.join(save_dir, "per_horizon_uv_speed.png"), dpi=150
                )
                plt.close(fig1)

                # Plot direction MAE
                fig2 = plt.figure(figsize=(10, 4))
                plt.plot(horizons, mae_direction_deg, label="MAE direction (deg)")
                plt.xlabel("Horizon (steps, 10-min)")
                plt.ylabel("Degrees")
                plt.title("Per-horizon direction error")
                plt.grid(True, alpha=0.3)
                plt.legend()
                fig2.tight_layout()
                fig2.savefig(
                    os.path.join(save_dir, "per_horizon_direction.png"), dpi=150
                )
                plt.close(fig2)

                logger.info(f"Saved per-horizon plots to {os.path.abspath(save_dir)}")
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")

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
        # Load model with proper device mapping - handle GPU->CPU/MPS conversion
        state_dict = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
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
        # Use column selection in groupby to avoid FutureWarning and ensure grouping col is excluded
        df[numeric_cols] = (
            df.groupby("station_id", sort=False)[numeric_cols]
            .apply(lambda g: g.ffill().bfill())
            .reset_index(level=0, drop=True)
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

    def _build_sequences_train_val(
        self,
        df: pd.DataFrame,
        val_fraction: float,
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, int]],
        List[Tuple[np.ndarray, np.ndarray, int]],
        int,
    ]:
        """
        Build time-aware train/validation sequences ensuring no target leakage across the split.

        For each station, a cutoff index is chosen based on val_fraction. Training targets are strictly
        before the cutoff; validation targets are at or after the cutoff. History windows for validation
        samples may overlap the training region, which reflects real-world forecasting usage.
        """
        train_sequences: List[Tuple[np.ndarray, np.ndarray, int]] = []
        val_sequences: List[Tuple[np.ndarray, np.ndarray, int]] = []
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

            cut = int(np.floor(n * (1.0 - float(val_fraction))))
            cut = int(np.clip(cut, 0, n))

            # Training sequences: targets entirely before cut
            max_train_start = cut - hist - horiz
            if max_train_start >= 0:
                for start in range(0, max_train_start + 1):
                    x = features[start : start + hist]
                    y = targets[start + hist : start + hist + horiz]
                    train_sequences.append((x, y, sid_idx))

            # Validation sequences: targets start at or after cut
            min_val_start = max(0, cut - hist)
            max_val_start = n - hist - horiz
            if max_val_start >= min_val_start:
                for start in range(min_val_start, max_val_start + 1):
                    # Ensure first target index is >= cut
                    if start + hist < cut:
                        continue
                    x = features[start : start + hist]
                    y = targets[start + hist : start + hist + horiz]
                    val_sequences.append((x, y, sid_idx))

        station_count = (
            len(self._station_id_to_index) if self._station_id_to_index else 0
        )
        return train_sequences, val_sequences, station_count
