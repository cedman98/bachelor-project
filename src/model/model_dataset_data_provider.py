from omegaconf import OmegaConf
from src.database.database_service import DatabaseService
import pandas as pd
from hamilton import driver

# Import feature module with Hamilton nodes
from src.model import model_dataset_preprocess as features


class ModelDatasetDataProvider:
    """
    The data provider provides the functionality for creating the traindataset
    """

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "station_id",
            "record_date",
            "average_wind_speed",
            "average_wind_direction",
            "air_pressure",
            "air_temperature_2m",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for feature generation: {missing}"
            )

        # Ensure datetime and sorted by station/time for correct lags/rolls
        if not pd.api.types.is_datetime64_any_dtype(df["record_date"]):
            df = df.copy()
            df["record_date"] = pd.to_datetime(
                df["record_date"], utc=False, errors="coerce"
            )

        df = df.sort_values(["station_id", "record_date"]).reset_index(drop=True)

        # Build Hamilton driver
        dr = driver.Driver({}, features)

        # Provide input columns directly
        inputs = {col: df[col] for col in df.columns}

        # Desired outputs: keep all original columns + computed features
        computed = [
            # targets
            "u",
            "v",
            # time encodings
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            # lags
            "u_lag_1",
            "u_lag_3",
            "u_lag_6",
            "u_lag_12",
            "u_lag_24",
            "v_lag_1",
            "v_lag_3",
            "v_lag_6",
            "v_lag_12",
            "v_lag_24",
            # rolling stats
            "u_roll_mean_3h",
            "u_roll_std_3h",
            "u_roll_mean_6h",
            "u_roll_std_6h",
            "v_roll_mean_3h",
            "v_roll_std_3h",
            "v_roll_mean_6h",
            "v_roll_std_6h",
            # tendencies
            "pressure_tendency_3h",
            "pressure_tendency_6h",
            "temperature_tendency_3h",
            "temperature_tendency_6h",
        ]

        outputs = list(df.columns) + computed

        result_df = dr.execute(final_vars=outputs, inputs=inputs)
        return result_df
