import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.database.schema import WindPowerCalculations
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
        self,
        target,
        measurements_df,
        weather_stations_df,
        beta=2,
        return_components=False,
    ):
        """
        IDW interpolation using two DataFrames.

        Parameters:
        -----------
        target : tuple (lat, lon)
            The location where we want to estimate wind (lat, lon).
        measurements_df : pd.DataFrame
            Supported schemas:
            - ["station_id", "average_wind_speed", "average_wind_direction"]
            - ["station_id", "u", "v"]
        weather_stations_df : pd.DataFrame
            Must contain columns: ["weather_station_id", "latitude", "longitude"]
        beta : int
            Power parameter for IDW (default=2).

        Returns:
        --------
        If return_components is False (default):
            (speed, direction) : tuple of floats
                Interpolated wind speed and direction at target location.
        If return_components is True:
            (u, v) : tuple of floats
                Interpolated wind components at target location.
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
            has_components = ("u" in row.index) and ("v" in row.index)
            if has_components:
                u = float(row["u"])  # east-west
                v = float(row["v"])  # north-south
            else:
                speed = row["average_wind_speed"]
                direction = row["average_wind_direction"]
                # Convert direction to radians (meteorological "from" convention)
                theta = np.deg2rad(direction)
                # Wind components (u toward east, v toward north)
                # Meteorological convention: direction is where wind comes FROM → negative sign
                u = -speed * np.sin(theta)  # east-west
                v = -speed * np.cos(theta)  # north-south
            lat, lon = row["latitude"], row["longitude"]

            # Euclidean distance (flat-earth approximation)
            d = np.sqrt((lat0 - lat) ** 2 + (lon0 - lon) ** 2)

            if d == 0:  # if target is exactly at a station
                if return_components:
                    return float(u), float(v)
                # Convert back to speed and direction for compatibility
                speed_exact = float(np.sqrt(u**2 + v**2))
                direction_exact = (float(np.rad2deg(np.arctan2(u, v))) + 360) % 360
                return speed_exact, direction_exact

            w = d ** (-beta)

            u_list.append(u)
            v_list.append(v)
            weights.append(w)

        # Convert to numpy arrays
        u_list, v_list, weights = np.array(u_list), np.array(v_list), np.array(weights)

        # Weighted averages
        u_interp = np.sum(weights * u_list) / np.sum(weights)
        v_interp = np.sum(weights * v_list) / np.sum(weights)

        if return_components:
            return float(u_interp), float(v_interp)

        # Convert back to speed and direction
        speed_interp = np.sqrt(u_interp**2 + v_interp**2)
        direction_interp = (np.rad2deg(np.arctan2(u_interp, v_interp)) + 360) % 360

        return float(speed_interp), float(direction_interp)

    def extrapolate_u_and_v_to_all_wind_turbines(
        self,
        wind_turbines_df: pd.DataFrame,
        weather_stations_df: pd.DataFrame,
        measurements_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extrapolate u and v to all wind turbines for all intervals.
        @param measurements_df: The measurements DataFrame.
        @return: The extrapolated u and v DataFrame.
        """
        # Ensure required columns exist in inputs
        required_turb_cols = [
            "unit_mastr_number",
            "latitude",
            "longitude",
            "manufacturer",
            "type_designation",
            "hub_height",
        ]
        required_ws_cols = ["weather_station_id", "latitude", "longitude"]
        required_meas_cols = ["station_id", "record_date", "is_prediction"]

        for col in required_turb_cols:
            if col not in wind_turbines_df.columns:
                raise ValueError(f"Missing column in wind_turbines_df: {col}")
        for col in required_ws_cols:
            if col not in weather_stations_df.columns:
                raise ValueError(f"Missing column in weather_stations_df: {col}")
        for col in required_meas_cols:
            if col not in measurements_df.columns:
                raise ValueError(f"Missing column in measurements_df: {col}")
        if not (
            ("u" in measurements_df.columns and "v" in measurements_df.columns)
            or (
                "average_wind_speed" in measurements_df.columns
                and "average_wind_direction" in measurements_df.columns
            )
        ):
            raise ValueError(
                "measurements_df must include either ['u','v'] or ['average_wind_speed','average_wind_direction']"
            )

        # Work per record_date
        per_date_frames = []
        # Sort dates for deterministic output
        for record_date, df_date in measurements_df.groupby("record_date", sort=True):
            if df_date.empty:
                continue

            # Determine is_prediction for this date (expect constant). If not, use any True.
            pred_vals = df_date["is_prediction"].dropna().unique().tolist()
            if len(pred_vals) == 0:
                is_pred = False
            elif len(pred_vals) == 1:
                is_pred = bool(pred_vals[0])
            else:
                # If mixed flags exist for same date, prefer True if any True present
                is_pred = bool(any(pred_vals))

            # Merge once per date to attach station coordinates
            stations_df = df_date.merge(
                weather_stations_df[["weather_station_id", "latitude", "longitude"]],
                left_on="station_id",
                right_on="weather_station_id",
                how="inner",
            )
            if stations_df.empty:
                continue

            # Prepare station component arrays (vectorized)
            if "u" in stations_df.columns and "v" in stations_df.columns:
                u_s = stations_df["u"].to_numpy(dtype=float)
                v_s = stations_df["v"].to_numpy(dtype=float)
            else:
                speed = stations_df["average_wind_speed"].to_numpy(dtype=float)
                direction = stations_df["average_wind_direction"].to_numpy(dtype=float)
                theta = np.deg2rad(direction)
                # Meteorological convention (from-direction): negative sign
                u_s = -speed * np.sin(theta)
                v_s = -speed * np.cos(theta)

            s_lat = stations_df["latitude"].to_numpy(dtype=float)
            s_lon = stations_df["longitude"].to_numpy(dtype=float)

            # Turbine arrays
            t_lat = wind_turbines_df["latitude"].to_numpy(dtype=float)
            t_lon = wind_turbines_df["longitude"].to_numpy(dtype=float)

            # Compute distance matrix (S x T)
            # Using broadcasting: rows=stations, cols=turbines
            d_lat = t_lat[np.newaxis, :] - s_lat[:, np.newaxis]
            d_lon = t_lon[np.newaxis, :] - s_lon[:, np.newaxis]
            d = np.sqrt(d_lat * d_lat + d_lon * d_lon)

            # Handle exact co-locations: if any zero distance for a turbine, use that station's value
            zero_mask = d == 0.0
            has_zero = zero_mask.any(axis=0)  # per turbine

            # Compute weights safely
            d_safe = np.where(zero_mask, 1.0, d)
            beta = 2
            W = d_safe ** (-beta)
            # Zero out weights where distance was zero (we will assign later)
            W = np.where(zero_mask, 0.0, W)

            # Weighted sums for non-zero cases
            sum_W = W.sum(axis=0)
            # Avoid divide by zero
            sum_W = np.where(sum_W == 0.0, 1.0, sum_W)
            u_weighted = (u_s @ W) / sum_W
            v_weighted = (v_s @ W) / sum_W

            # For turbines that coincide with any station, assign exact (if multiple, average those stations)
            if has_zero.any():
                # For each turbine j with any zero, average u_s/v_s over zero stations
                idx_turbines = np.where(has_zero)[0]
                for j in idx_turbines:
                    idx_stations = np.where(zero_mask[:, j])[0]
                    if idx_stations.size == 1:
                        u_weighted[j] = float(u_s[idx_stations[0]])
                        v_weighted[j] = float(v_s[idx_stations[0]])
                    else:
                        u_weighted[j] = float(u_s[idx_stations].mean())
                        v_weighted[j] = float(v_s[idx_stations].mean())

            # Build per-date DataFrame efficiently by copying turbine meta and attaching results
            frame = wind_turbines_df[
                [
                    "unit_mastr_number",
                    "latitude",
                    "longitude",
                    "manufacturer",
                    "type_designation",
                    "hub_height",
                ]
            ].copy()
            frame.loc[:, "record_date"] = record_date
            frame.loc[:, "u"] = u_weighted.astype(float)
            frame.loc[:, "v"] = v_weighted.astype(float)
            frame.loc[:, "is_prediction"] = is_pred

            per_date_frames.append(frame)

        if not per_date_frames:
            # Return empty DataFrame with expected schema
            return pd.DataFrame(
                columns=[
                    "unit_mastr_number",
                    "latitude",
                    "longitude",
                    "manufacturer",
                    "type_designation",
                    "hub_height",
                    "record_date",
                    "u",
                    "v",
                    "is_prediction",
                ]
            )

        result_df = pd.concat(per_date_frames, ignore_index=True)
        # Order columns explicitly
        result_df = result_df[
            [
                "unit_mastr_number",
                "latitude",
                "longitude",
                "manufacturer",
                "type_designation",
                "hub_height",
                "record_date",
                "u",
                "v",
                "is_prediction",
            ]
        ]
        return result_df

    def extrapolate_to_hub_height(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrapolate to hub height.
        @param measurements_df: The measurements DataFrame.
        @return: The extrapolated measurements DataFrame.
        """
        # Calculate wind speed from u and v components
        measurements_df["wind_speed"] = np.sqrt(
            measurements_df["u"] ** 2 + measurements_df["v"] ** 2
        )

        # TODO: Extrapolate to hub height, this is mock
        measurements_df["hub_height_wind_speed"] = measurements_df["wind_speed"]

        return measurements_df

    def save_calculations_to_database(self, calculations_df: pd.DataFrame) -> None:
        """
        Save the calculations to the database.
        @param calculations_df: The calculations DataFrame.
        """
        if calculations_df is None or calculations_df.empty:
            logger.warning("No calculations to save")
            return

        # Rename columns to match schema
        rename_map = {
            "wind_speed": "extrapolated_wind_speed",
            "hub_height_wind_speed": "extrapolated_hub_height_wind_speed",
        }
        df = calculations_df.rename(columns=rename_map).copy()

        # Filter to allowed columns defined by the table
        table = WindPowerCalculations.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in df.to_dict(orient="records")
        ]

        if not records:
            logger.warning(
                "No compatible columns found to upsert into wind_power_calculations"
            )
            return

        chunk_size = 7500
        max_retries = 3
        total = len(records)

        for start in range(0, total, chunk_size):
            chunk = records[start : start + chunk_size]
            attempt = 0
            while True:
                try:
                    stmt = pg_insert(table).values(chunk)
                    update_columns = {
                        c.name: stmt.excluded[c.name]
                        for c in table.columns
                        if c.name not in {"id"}
                    }
                    # Prefer a deterministic upsert on natural key
                    # Use index_elements to avoid relying on a possibly-misnamed constraint
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["unit_mastr_number", "record_date"],
                        set_=update_columns,
                    )

                    with Session(self.database_service.engine) as session:
                        session.execute(stmt)
                        session.commit()

                    logger.info(
                        f"Upserted calculations chunk {start}-{min(start+chunk_size, total)} of {total}"
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(
                            f"Failed upserting calculations chunk {start}-{min(start+chunk_size, total)} after {max_retries} attempts: {e}"
                        )
                        raise
                    sleep_s = 2 ** (attempt - 1)
                    logger.warning(
                        f"Error during upsert (attempt {attempt}/{max_retries}) for chunk {start}-{min(start+chunk_size, total)}: {e}. Retrying in {sleep_s}s"
                    )
                    import time as _time

                    _time.sleep(sleep_s)
