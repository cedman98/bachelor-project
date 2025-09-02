from loguru import logger
from omegaconf import OmegaConf
import pandas as pd

from src.measurements.measurement_data_provider import MeasurementDataProvider
from src.database.database_service import DatabaseService


class MeasurementService:
    """
    The service provides the functionality for downloading and loading the measurements from DWD for the measurement stations.
    """

    cfg: OmegaConf
    database_service: DatabaseService
    weather_stations: pd.DataFrame

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        weather_stations: pd.DataFrame,
        measurement_data_provider: MeasurementDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.weather_stations = weather_stations
        self.measurement_data_provider = (
            measurement_data_provider
            if measurement_data_provider
            else MeasurementDataProvider(cfg, database_service)
        )

    def fill_database_with_measurements(
        self, only_now: bool = False, bulk_save: bool = False
    ):
        """
        Fill the database with the measurements from DWD for the measurement stations.
        If you want to fill the database for the first time, you should use bulk_save=True and only_now=False. Use only_now=True if and bulk_save=False during filling it hourly
        @param only_now: If True, only get the now data (current day).
        @param bulk_save: If True, use the bulk save function to save the measurements to the database.
        @return: The weather stations DataFrame. The df is also stored in the service.
        """
        for weather_station in self.weather_stations.itertuples():
            weather_station_id = weather_station.weather_station_id
            try:
                raw_measurement_df = self.measurement_data_provider.download_measurements_for_weather_station(
                    weather_station_id, only_now
                )

                if raw_measurement_df.empty:
                    logger.warning(
                        f"No dataframes downloaded for weather station {weather_station_id}. Skipping..."
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Error downloading measurements for weather station {weather_station_id}: {e}"
                )
                return None

            try:
                processed_measurement_df = (
                    self.measurement_data_provider.process_measurement_df(
                        raw_measurement_df
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error processing measurements for weather station {weather_station_id}: {e}"
                )
                return None

            try:
                if bulk_save:
                    self.measurement_data_provider.bulk_save_measurement_df_to_database(
                        processed_measurement_df
                    )
                else:
                    self.measurement_data_provider.save_measurement_df_to_database(
                        processed_measurement_df
                    )

            except Exception as e:
                logger.error(
                    f"Error saving measurements to database for weather station {weather_station_id}: {e}"
                )
                return None

        logger.info(
            f"Filled database with measurements for {len(self.weather_stations)} weather stations"
        )
