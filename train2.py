import os
import warnings
from omegaconf import OmegaConf
from loguru import logger
import pandas as pd

from src.model.variant.bilstm_model import BiLSTMModel
from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_service import WeatherStationService
from src.measurements.measurement_service import MeasurementService

# Suppress pandas FutureWarning about groupby behavior
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def main(): 
    # Load config and initialize services
    cfg = OmegaConf.load("conf/config.yaml")
    
    db = DatabaseService(cfg)
    ws_service = WeatherStationService(cfg, db)
    weather_stations = ws_service.load_from_database(only_relevant=True)
    
    ms_service = MeasurementService(cfg, db, weather_stations)
    
    measurements_df = pd.read_parquet("data/extended_hourly_measurements.parquet")
    measurements_df = measurements_df.sort_values(by="record_date")
    logger.info(f"Measurements: {len(measurements_df)} rows")
    
    end_date = pd.Timestamp('2025-04-01')
    train_df = measurements_df[(measurements_df['record_date']<=end_date)].copy()
    test_df = measurements_df[measurements_df['record_date']>end_date].copy()
    
    # BiLSTM full training hyperparameters (RTX 4080 Super)
    # Using last 24 hours (24 steps at 1-hour intervals) to forecast next 12 hours (12 steps).
    # BiLSTM processes the full sequence through bidirectional LSTM layers.
    # Features: wind speed, direction (sin/cos), and cyclical time encodings only (no weather features).
    # Expected training time: faster than 10-min model due to fewer timesteps per sequence
    model = BiLSTMModel(
        history_steps=24,            # 24 hours of history at 1-hour resolution
        horizon_steps=12,            # 12 hours forecast at 1-hour resolution
        hidden_size=128,             
        num_layers=2,                
        station_embedding_dim=16,    
        dropout=0.3,                 
        batch_size=256,              
        learning_rate=3e-4,          
        num_epochs=100,              
        val_split=0.15,              
        early_stopping_patience=15,  
        early_stopping_min_delta=1e-4, 
        restore_best_weights=True,   # Restore best weights for final model
        shuffle_train=True          
    )

    model.train(train_df)

    save_dir = "models/bilstm_full_last"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    logger.info(f"Saved BiLSTM full model to {save_dir}")

    # Full evaluation on test set
    logger.info("Running full evaluation on test set...")
    evaluation = model.evaluate(test_df)
    logger.info(f"Full evaluation metrics: {evaluation}")
    
    # Per-horizon evaluation with plots
    logger.info("Computing per-horizon metrics...")
    per_horizon_dir = os.path.join(save_dir, "per_horizon_plots")
    per_horizon_metrics = model.evaluate_per_horizon(test_df, save_dir=per_horizon_dir)
    logger.info(f"Per-horizon metrics saved to {per_horizon_dir}")


if __name__ == "__main__":
    main()
