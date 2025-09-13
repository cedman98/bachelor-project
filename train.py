import os
from omegaconf import OmegaConf
from loguru import logger
import pandas as pd

from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_service import WeatherStationService
from src.measurements.measurement_service import MeasurementService
from src.model.variant.bilstm_model import BiLSTMModel


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
    
    model = BiLSTMModel(
        history_steps=72,   # 12 hours of history
        horizon_steps=72,   # 12 hours forecast
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
        restore_best_weights=True  # increase for real training
    )
    
    model.train(train_df)
    
    save_dir = "models/new_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved BiLSTM model to {save_dir}")
    
    evaluation = model.evaluate(test_df)
    
    logger.info(f"Evaluation: {evaluation}")


if __name__ == "__main__":
    main()
