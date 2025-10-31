from datetime import datetime
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

    def fill_database_with_measurements(self, only_now: bool = False):
        """
        Fill the database with the measurements from DWD for the measurement stations.
        If you want to fill the database for the first time, you should use only_now=False to load also all historical data. Use only_now=True if and only if you want to load the current data (current day).
        @param only_now: If True, only get the now data (current day).
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
                raise e

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
                raise e

            try:
                self.measurement_data_provider.save_measurement_df_to_database(
                    processed_measurement_df
                )

            except Exception as e:
                logger.error(
                    f"Error saving measurements to database for weather station {weather_station_id}: {e}"
                )
                raise e

        logger.info(
            f"Filled database with measurements for {len(self.weather_stations)} weather stations"
        )

    def load_measurements_from_database_for_datetime(
        self, datetime: datetime
    ) -> pd.DataFrame:
        """
        @param datetime: The datetime.
        @return: The measurements DataFrame.
        """
        if self.weather_stations is None:
            raise ValueError(
                "No weather stations loaded. Please load the weather stations first."
            )

        try:
            return self.measurement_data_provider.load_measurements_from_database_for_datetime(
                self.weather_stations, datetime
            )
        except Exception as e:
            logger.error(
                f"Error loading measurements from database for datetime {datetime}: {e}"
            )
            return None

    def load_all_measurements_from_database(self) -> pd.DataFrame:
        """
        Load all measurements from the database.
        @return: The measurements DataFrame.
        """
        return self.measurement_data_provider.load_all_measurements_from_database()

    def load_all_measurements_grid_backfilled_from_database(self) -> pd.DataFrame:
        """
        Load all measurements from the database.
        @return: The measurements DataFrame.
        """
        return (
            self.measurement_data_provider.load_all_measurements_grid_backfilled_from_database()
        )

    def load_all_recent_measurements_from_database(self) -> pd.DataFrame:
        """
        The function loads for all stations the last 24 hours of 10-minute measurements,
        then aggregates them to hourly resolution by taking the mean (same as aggregate_hourly.ipynb).
        @return: The measurements DataFrame aggregated to hourly resolution.
        """
        logger.info(f"Loading all recent measurements from database")
        df = self.measurement_data_provider.load_all_recent_measurements_from_database()
        logger.info(f"Loaded {len(df)} recent measurements from database")
        return df

    def transform_measurements_to_prediction_format(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform the measurements to the prediction format.
        @param df: The measurements DataFrame.
        @return: The measurements DataFrame in the prediction format.
        """
        import numpy as np

        # Convert wind direction from degrees to radians
        direction_rad = np.deg2rad(df["average_wind_direction"])

        # Calculate u and v components using meteorological convention
        df["u"] = -df["average_wind_speed"] * np.sin(direction_rad)
        df["v"] = -df["average_wind_speed"] * np.cos(direction_rad)

        return df[["station_id", "record_date", "u", "v"]]
