import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from src.database.database_service import DatabaseService


class WindCalculationDataProvider:

    cfg: OmegaConf
    database_service: DatabaseService
    wind_turbines: pd.DataFrame
    weather_stations: pd.DataFrame

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        wind_turbines: pd.DataFrame,
        weather_stations: pd.DataFrame,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.wind_turbines = wind_turbines
        self.weather_stations = weather_stations

    def idw_interpolation_df(
        self, target, measurements_df, weather_stations_df, beta=2
    ):
        """
        IDW interpolation for wind speed and direction using two DataFrames.

        Parameters:
        -----------
        target : tuple (lat, lon)
            The location where we want to estimate wind (lat, lon).
        measurements_df : pd.DataFrame
            Must contain columns: ["station_id", "average_wind_speed", "average_wind_direction"]
        weather_stations_df : pd.DataFrame
            Must contain columns: ["weather_station_id", "latitude", "longitude"]
        beta : int
            Power parameter for IDW (default=2).

        Returns:
        --------
        (speed, direction) : tuple
            Interpolated wind speed and direction at target location.
        """

        lat0, lon0 = target

        # Merge observations with station metadata
        df = measurements_df.merge(
            weather_stations_df[["weather_station_id", "latitude", "longitude"]],
            left_on="station_id",
            right_on="weather_station_id",
            how="inner",
        )

        u_list, v_list, weights = [], [], []

        for _, row in df.iterrows():
            speed = row["average_wind_speed"]
            direction = row["average_wind_direction"]
            lat, lon = row["latitude"], row["longitude"]

            # Convert direction to radians (meteorological convention)
            theta = np.deg2rad(direction)

            # Wind components
            u = speed * np.sin(theta)  # east-west
            v = speed * np.cos(theta)  # north-south

            # Euclidean distance (flat-earth approximation)
            d = np.sqrt((lat0 - lat) ** 2 + (lon0 - lon) ** 2)

            if d == 0:  # if target is exactly at a station
                return speed, direction

            w = d ** (-beta)

            u_list.append(u)
            v_list.append(v)
            weights.append(w)

        # Convert to numpy arrays
        u_list, v_list, weights = np.array(u_list), np.array(v_list), np.array(weights)

        # Weighted averages
        u_interp = np.sum(weights * u_list) / np.sum(weights)
        v_interp = np.sum(weights * v_list) / np.sum(weights)

        # Convert back to speed and direction
        speed_interp = np.sqrt(u_interp**2 + v_interp**2)
        direction_interp = (np.rad2deg(np.arctan2(u_interp, v_interp)) + 360) % 360

        return speed_interp, direction_interp
