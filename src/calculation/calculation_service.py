from datetime import datetime
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
from src.calculation.wind_calculation_data_provider import WindCalculationDataProvider
from src.database.database_service import DatabaseService
from src.measurements.measurement_service import MeasurementService


class CalculationService:

    cfg: OmegaConf
    database_service: DatabaseService
    wind_turbines: pd.DataFrame
    weather_stations: pd.DataFrame
    measurement_service: MeasurementService

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        measurement_service: MeasurementService,
        wind_turbines: pd.DataFrame,
        weather_stations: pd.DataFrame,
        wind_calculation_data_provider: WindCalculationDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.wind_turbines = wind_turbines
        self.weather_stations = weather_stations
        self.measurement_service = measurement_service
        self.wind_calculation_data_provider = (
            wind_calculation_data_provider
            if wind_calculation_data_provider
            else WindCalculationDataProvider(
                cfg, database_service, wind_turbines, weather_stations
            )
        )

    def create_dataset(self):

        test_datetime = datetime(2025, 9, 4, 10, 0)
        measurements_df = (
            self.measurement_service.load_measurements_from_database_for_datetime(
                test_datetime
            )
        )

        test_target = (52.0547, 13.2344)
        wind_speed, wind_direction = (
            self.wind_calculation_data_provider.idw_interpolation_df(
                test_target, measurements_df, self.weather_stations
            )
        )

        logger.info(f"Wind speed: {wind_speed}, Wind direction: {wind_direction}")

        return measurements_df
