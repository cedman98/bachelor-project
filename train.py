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
    
    measurements_df = ms_service.load_all_measurements_from_database()
    measurements_df = measurements_df.sort_values(by="record_date")
    logger.info(f"Measurements: {len(measurements_df)} rows")
    
    end_date = pd.Timestamp('2025-04-01')
    train_df = measurements_df[(measurements_df['record_date']<=end_date)].copy()
    test_df = measurements_df[measurements_df['record_date']>end_date].copy()
    
    # PatchTST hourly training hyperparameters (RTX 4080 Super)
    # Using 12 days (288 hours) of history and forecasting the next 12 hours.
    # patch_len=16 and stride=8 yield (288-16)//8 + 1 = 35 patches per sample.
    # d_model, nhead, num_layers, and dim_feedforward remain appropriate.
    # If you observe memory issues, reduce batch_size or d_model.
    model = PatchTSTModel(
        history_steps=144,   # last 24 hours (10-min intervals)
        horizon_steps=72,    # next 12 hours (10-min intervals)
        d_model=256,
        nhead=8,
        num_layers=3,        # slightly shallower for faster prototype
        dim_feedforward=512,
        patch_len=16,
        stride=8,
        dropout=0.1,
        batch_size=64,       # reduce if you run out of memory
        learning_rate=3e-4,
        num_epochs=10,       # short prototype run; early stopping will cut sooner if needed
        val_split=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True
    )

    model.train(train_df)

    save_dir = "models/patch_tst_prototype"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved PatchTST prototype model to {save_dir}")

    # Limit evaluation batches for quicker turnaround in prototype
    evaluation = model.evaluate(test_df, max_batches=10)
    logger.info(f"Prototype evaluation (limited batches): {evaluation}")


if __name__ == "__main__":
    main()
