import os
from omegaconf import OmegaConf
from loguru import logger
import pandas as pd

from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_service import WeatherStationService
from src.measurements.measurement_service import MeasurementService
from src.model.variant.patch_tst_model import PatchTSTModel


def main(): 
    # Load config and initialize services
    cfg = OmegaConf.load("conf/config.yaml")
    
    db = DatabaseService(cfg)
    ws_service = WeatherStationService(cfg, db)
    weather_stations = ws_service.load_from_database(only_relevant=True)
    
    ms_service = MeasurementService(cfg, db, weather_stations)
    
    measurements_df = pd.read_parquet("data/extended_hourly_measurements.parquet")
    measurements_df = measurements_df.sort_values(by=["station_id", "record_date"])
    logger.info(f"Measurements: {len(measurements_df)} rows")
    
    end_date = pd.Timestamp('2025-04-01')
    train_df = measurements_df[(measurements_df['record_date']<=end_date)].copy()
    test_df = measurements_df[measurements_df['record_date']>end_date].copy()
    
    # PatchTST hourly training hyperparameters - PROTOTYPE RUN
    # Using 24 hours of history and forecasting the next 12 hours.
    # patch_len=6 and stride=3 yield (24-6)//3 + 1 = 7 patches per sample.
    # Lightweight configuration for fast iteration and testing.
    model = PatchTSTModel(
        history_steps=168,        # last 7 days (hourly intervals)
        horizon_steps=24,         # next 24 hours forecast
        d_model=256,              # higher model dimension for full capacity
        nhead=8,                  # more attention heads for better feature extraction
        num_layers=4,             # deeper Transformer stack for complex patterns
        dim_feedforward=1024,     # larger feedforward network for richer representations
        patch_len=12,             # larger temporal patch for better local context
        stride=6,                 # moderate overlap between patches
        dropout=0.2,              # keep mild dropout for regularization
        batch_size=512,           # larger batch size for stable training (if GPU memory allows)
        learning_rate=5e-5,       # slightly lower LR for stable convergence on large data
        num_epochs=100,           # long training horizon, with early stopping
        val_split=0.2,            # 10% validation split since data is abundant
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True
    )
    
    model.train(train_df)

    save_dir = "models/patch_tst_prototype_large"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved PatchTST prototype model to {save_dir}")

    # Limit evaluation batches for quicker turnaround in prototype
    evaluation = model.evaluate(test_df, max_batches=10)
    logger.info(f"Prototype evaluation (limited batches): {evaluation}")


if __name__ == "__main__":
    main()
