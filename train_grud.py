import os
from omegaconf import OmegaConf
from loguru import logger
import pandas as pd

from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_service import WeatherStationService
from src.measurements.measurement_service import MeasurementService
from src.model.variant.gru_d_model import GRUDModel


def main():
    # Load config and initialize services
    cfg = OmegaConf.load("conf/config.yaml")

    measurements_df = pd.read_parquet("data/raw_measurements.parquet")
    logger.info(f"Measurements: {len(measurements_df)} rows")

    # Split data at 2024-04-01
    end_date = pd.Timestamp("2025-04-01")
    train_df = measurements_df[(measurements_df["record_date"] <= end_date)].copy()
    test_df = measurements_df[measurements_df["record_date"] > end_date].copy()

    logger.info(f"Train data: {len(train_df)} rows (up to {end_date})")
    logger.info(f"Test data: {len(test_df)} rows (after {end_date})")

    # Initialize GRU-D model
    model = GRUDModel(
        cfg=cfg,
        history_steps=144,  # 24 hours of history
        horizon_steps=72,  # 12 hours forecast
        station_embedding_dim=8,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=5,
        val_split=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=5e-5,
        restore_best_weights=True,
    )

    logger.info("Starting GRU-D model training...")
    model.train(train_df)

    # Save model
    save_dir = "models/grud_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved GRU-D model to {save_dir}")

    # Evaluate on test set
    logger.info("Evaluating GRU-D model on test set...")
    evaluation = model.evaluate(test_df)

    logger.info("=" * 80)
    logger.info("GRU-D MODEL EVALUATION RESULTS")
    logger.info("=" * 80)
    for metric, value in evaluation.items():
        if "direction" in metric:
            logger.info(f"{metric}: {value:.2f}°")
        else:
            logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
