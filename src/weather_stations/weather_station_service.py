from loguru import logger
from omegaconf import OmegaConf
import pandas as pd

from src.database.database_service import DatabaseService
from src.weather_stations.weather_station_data_provider import (
    WeatherStationDataProvider,
)


class WeatherStationService:
    """
    The service offers functions for working with the weather stations.
    """

    cfg: OmegaConf
    database_service: DatabaseService
    data_provider: WeatherStationDataProvider
    weather_stations: pd.DataFrame

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        data_provider: WeatherStationDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.data_provider = (
            data_provider
            if data_provider
            else WeatherStationDataProvider(cfg, database_service)
        )

    def fill_database_with_weather_stations(self):
        """
        Fill the database with the weather stations from the DWD.
        @return: The weather stations DataFrame. The df is also stored in the service.
        """
        # 1. Loading the weather stations file
        try:
            weather_stations_file = self.data_provider.download_weather_stations_file()
        except Exception as e:
            logger.error(f"Error downloading weather stations file: {e}")
            return None

        # 2. Parsing the weather stations file
        try:
            weather_stations = self.data_provider.parse_weather_stations_file(
                weather_stations_file
            )
        except Exception as e:
            logger.error(f"Error parsing weather stations file: {e}")
            return None

        # 3. Processing the weather stations file
        try:
            weather_stations = self.data_provider.process_weather_stations_df(
                weather_stations
            )
        except Exception as e:
            logger.error(f"Error processing weather stations file: {e}")
            return None

        # 4. Saving the weather stations to the database
        try:
            self.data_provider.save_weather_stations_to_database(weather_stations)
        except Exception as e:
            logger.error(f"Error saving weather stations to database: {e}")
            return None

        self.weather_stations = weather_stations

        return self.weather_stations

    def load_from_database(self, only_relevant: bool = True) -> pd.DataFrame:
        """
        Load the weather stations from the database.
        @param only_relevant: If True, only load the relevant weather stations, (active and in Brandenburg)
        @return: The weather stations DataFrame.
        """
        self.weather_stations = self.data_provider.load_from_database(only_relevant)
        return self.weather_stations
