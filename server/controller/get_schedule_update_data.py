import os
import sys
from omegaconf import DictConfig


# Ensure project root is on PYTHONPATH so `src.*` imports work when running from `server/`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.calculation.calculation_service import CalculationService
from src.wind_turbines.wind_turbines_service import WindTurbinesService
from src.prediction.prediction_service import PredictionService
from src.model.variant.bilstm_model import BiLSTMModel
from src.model.model_service import ModelService
from src.measurements.measurement_service import MeasurementService
from src.aggregation.aggregation_service import AggregationService
from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_service import WeatherStationService


def get_schedule_update_data(cfg: DictConfig, database_service: DatabaseService):

    # 1. Get weather stations
    weather_station_service = WeatherStationService(cfg, database_service)
    weather_stations_df = weather_station_service.load_from_database()

    # 2. Fill database with new measurements
    measurement_service = MeasurementService(cfg, database_service, weather_stations_df)
    measurement_service.fill_database_with_measurements(only_now=True)

    # 3. Load the model
    model_service = ModelService(cfg, database_service, measurement_service)
    lstm = BiLSTMModel()
    lstm.load("model/lstm/")
    model_service.attach_model(lstm)

    # 4. Make predictions
    prediction_service = PredictionService(
        cfg, database_service, measurement_service, model_service
    )
    measurements_df = prediction_service.predict_measurements()

    # 5. Extrapolate to u and v for all wind turbines
    wind_turbines_service = WindTurbinesService(cfg, database_service)
    all_wind_turbines_df = wind_turbines_service.load_from_database()
    calculation_service = CalculationService(
        cfg,
        database_service,
        measurement_service,
        all_wind_turbines_df,
        weather_stations_df,
    )
    extrapolated_measurements_df = (
        calculation_service.extrapolate_u_and_v_to_all_wind_turbines(measurements_df)
    )

    # 6. Extrapolate to hub height
    extrapolated_measurements_df = calculation_service.extrapolate_to_hub_height(
        extrapolated_measurements_df
    )

    # 7. Calculate power production
    calculation_service.calculate_power_production(extrapolated_measurements_df)

    return {"message": "Data updated successfully"}
