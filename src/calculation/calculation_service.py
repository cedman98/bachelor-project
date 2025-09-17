from datetime import datetime
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
from src.calculation.power_curves_data_provider import PowerCurvesDataProvider
from src.calculation.wind_calculation_data_provider import WindCalculationDataProvider
from src.database.database_service import DatabaseService
from src.measurements.measurement_service import MeasurementService


class CalculationService:

    cfg: OmegaConf
    database_service: DatabaseService
    wind_turbines: pd.DataFrame
    weather_stations: pd.DataFrame
    measurement_service: MeasurementService
    wind_calculation_data_provider: WindCalculationDataProvider
    power_curves_data_provider: PowerCurvesDataProvider

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        measurement_service: MeasurementService,
        wind_turbines: pd.DataFrame,
        weather_stations: pd.DataFrame,
        wind_calculation_data_provider: WindCalculationDataProvider = None,
        power_curves_data_provider: PowerCurvesDataProvider = None,
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
        self.power_curves_data_provider = (
            power_curves_data_provider
            if power_curves_data_provider
            else PowerCurvesDataProvider(cfg, database_service)
        )

    def fill_database_with_power_curves(self):
        """
        Save the power curves from the data folder to the database.
        """
        self.power_curves_data_provider.save_power_curves_to_database()

    def load_power_curves_from_database(self) -> pd.DataFrame:
        """
        Load the power curves from the database.

        @return: The power curves DataFrame.
        """
        return self.power_curves_data_provider.load_from_database()

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

    def extrapolate_u_and_v_to_all_wind_turbines(
        self, measurements_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extrapolate u and v to all wind turbines for all intervals.
        @param measurements_df: The measurements DataFrame.
        @return: The extrapolated u and v DataFrame.
        """

        wind_turbines_df = self.wind_turbines[
            [
                "unit_mastr_number",
                "latitude",
                "longitude",
                "manufacturer",
                "type_designation",
                "hub_height",
            ]
        ]

        weather_stations_df = self.weather_stations[
            ["weather_station_id", "latitude", "longitude", "height"]
        ]

        return self.wind_calculation_data_provider.extrapolate_u_and_v_to_all_wind_turbines(
            wind_turbines_df, weather_stations_df, measurements_df
        )

    def extrapolate_to_hub_height(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrapolate to hub height.
        @param measurements_df: The measurements DataFrame.
        @return: The extrapolated measurements DataFrame.
        """
        return self.wind_calculation_data_provider.extrapolate_to_hub_height(
            measurements_df
        )

    def calculate_power_production(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the power production for all wind turbines.
        @param measurements_df: The measurements DataFrame.
        @return: The power production DataFrame.
        """
        if (
            "unit_mastr_number" not in measurements_df.columns
            or "hub_height_wind_speed" not in measurements_df.columns
        ):
            raise ValueError(
                "measurements_df must have the columns 'unit_mastr_number' and 'hub_height_wind_speed'"
            )

        matched_df = (
            self.power_curves_data_provider.get_all_turbine_id_for_mastr_number(
                measurements_df
            )
        )

        power_curves_df = self.power_curves_data_provider.load_from_database()

        power_production_df = (
            self.power_curves_data_provider.calculate_wind_power_production(
                measurements_df, matched_df, power_curves_df
            )
        )

        self.wind_calculation_data_provider.save_calculations_to_database(
            power_production_df
        )

        return power_production_df
