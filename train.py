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
    
    # PatchTST full train hyperparameters for RTX 4080 Super
    # With history_steps increased to 288 (24 hours at 10-min resolution), most parameters remain valid.
    # However, patch_len and stride should be considered in relation to the new sequence length.
    # patch_len=16 and stride=8 will result in (288-16)//8 + 1 = 35 patches per sample, which is reasonable.
    # d_model, nhead, num_layers, and dim_feedforward are also still appropriate for this input size.
    # If you observe memory issues, consider reducing batch_size or d_model.
    model = PatchTSTModel(
        history_steps=288,   # 24 hours of history (10-min resolution)
        horizon_steps=72,    # 12 hours forecast
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        patch_len=16,        # 16-step patches (still reasonable for 288 steps)
        stride=8,            # 8-step stride (overlapping patches)
        dropout=0.1,
        batch_size=128,      # If you run out of memory, reduce this
        learning_rate=3e-4,
        num_epochs=100,
        val_split=0.2,
        early_stopping_patience=10,
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
