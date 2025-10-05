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
    
    measurements_df = pd.read_parquet("data/measurements.parquet")
    measurements_df = measurements_df.sort_values(by="record_date")
    logger.info(f"Measurements: {len(measurements_df)} rows")
    
    end_date = pd.Timestamp('2025-04-01')
    train_df = measurements_df[(measurements_df['record_date']<=end_date)].copy()
    test_df = measurements_df[measurements_df['record_date']>end_date].copy()
    
    # BiLSTM full training hyperparameters (RTX 4080 Super)
    # Using last 12 hours (72 steps at 10-min intervals) to forecast next 12 hours (72 steps).
    # BiLSTM processes the full sequence through bidirectional LSTM layers.
    # Based on prototype results: model was overfitting (train loss 0.129, val loss 0.359)
    # Adjustments: increased dropout, larger model for capacity, more patience
    # Expected training time: ~10-15 hours for full run (based on ~5min/epoch × 100-150 epochs)
    model = BiLSTMModel(
        history_steps=144,           
        horizon_steps=72,            
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
        restore_best_weights=True,  
        shuffle_train=True          
    )

    model.train(train_df)

    save_dir = "models/bilstm_full"
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
