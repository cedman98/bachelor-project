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
    logger.info(f"Measurements: {len(measurements_df)} rows")
    
    end_date = pd.Timestamp('2025-04-01')
    train_df = measurements_df[(measurements_df['record_date']<=end_date)].copy()
    test_df = measurements_df[measurements_df['record_date']>end_date].copy()
    
    # PatchTST prototype hyperparameters for RTX 4080 Super
    model = PatchTSTModel(
        history_steps=144,   # 24 hours of history (10-min resolution)
        horizon_steps=72,    # 12 hours forecast
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        patch_len=16,
        stride=8,
        dropout=0.1,
        batch_size=128,
        learning_rate=3e-4,
        num_epochs=5,
        val_split=0.2,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True
    )
    
    model.train(train_df)
    
    save_dir = "models/patch_tst"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved PatchTST model to {save_dir}")
    
    evaluation = model.evaluate(test_df)
    
    logger.info(f"Evaluation: {evaluation}")


if __name__ == "__main__":
    main()
