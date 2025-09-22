from loguru import logger
from omegaconf import OmegaConf
import pandas as pd

from src.database.database_service import DatabaseService
from src.wind_turbines.wind_turbines_data_provider import WindTurbinesDataProvider


class WindTurbinesService:
    """
    The service provides the functionality for downloading and loading the wind turbines from the database.
    """

    cfg: OmegaConf
    database_service: DatabaseService
    wind_turbines: pd.DataFrame
    wind_turbines_data_provider: WindTurbinesDataProvider

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        wind_turbines_data_provider: WindTurbinesDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.wind_turbines_data_provider = (
            wind_turbines_data_provider
            if wind_turbines_data_provider
            else WindTurbinesDataProvider(cfg, database_service)
        )

    def fill_database_with_wind_turbines(self, download_files: bool = False):
        """
        Fill the database with the wind turbines.
        """
        # 1. Download the wind turbines and get raw df
        try:
            raw_wind_turbines = self.wind_turbines_data_provider.download_wind_turbines(
                download_files=download_files
            )
        except Exception as e:
            logger.error(f"Error downloading wind turbines: {e}")
            return None

        # 2. Proccess the wind turbines
        try:
            processed_wind_turbines = (
                self.wind_turbines_data_provider.process_wind_turbines_df(
                    raw_wind_turbines
                )
            )
        except Exception as e:
            logger.error(f"Error processing wind turbines: {e}")
            return None

        # 3. Save the wind turbines to the database
        try:
            self.wind_turbines_data_provider.save_wind_turbines_df_to_database(
                processed_wind_turbines
            )
        except Exception as e:
            logger.error(f"Error saving wind turbines to database: {e}")
            return None

        self.wind_turbines = processed_wind_turbines

        return self.wind_turbines

    def load_from_database(self) -> pd.DataFrame:
        """
        Load the wind turbines from the database.
        """
        self.wind_turbines = self.wind_turbines_data_provider.load_from_database()
        return self.wind_turbines

    def load_one_unit_from_database(self, unit_mastr_number: str) -> pd.DataFrame:
        """
        Load one unit from the database.
        """
        return self.wind_turbines_data_provider.load_one_unit_from_database(
            unit_mastr_number
        )
