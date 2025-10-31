from loguru import logger
from omegaconf import OmegaConf
from src.model.variant.model_interface import ModelInterface
from src.model.model_dataset_data_provider import ModelDatasetDataProvider
from src.database.database_service import DatabaseService
from src.measurements.measurement_service import MeasurementService
import pandas as pd
import numpy as np


class ModelService:
    """
    The service provides the functionality for loading the model from the database.
    """

    cfg: OmegaConf
    database_service: DatabaseService
    measurement_service: MeasurementService
    model_dataset_data_provider: ModelDatasetDataProvider
    model: ModelInterface
    dataset: pd.DataFrame | None

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        measurement_service: MeasurementService,
        model_dataset_data_provider: ModelDatasetDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.measurement_service = measurement_service
        self.model_dataset_data_provider = (
            model_dataset_data_provider
            if model_dataset_data_provider
            else ModelDatasetDataProvider(cfg, database_service)
        )

    def load_dataset(self):
        """
        Load the dataset from the database.
        """
        try:
            raw_measurements_df = (
                self.measurement_service.load_all_measurements_from_database()
            )

        except Exception as e:
            logger.error(f"Error loading measurements from database: {e}")
            return None

        self.dataset = self.model_dataset_data_provider.create_all_features(
            raw_measurements_df
        )

        return self.dataset

    def save_dataset_as_pickle(self, path: str = "data/dataset.pkl") -> None:
        """
        Save the dataset as a pickle file.
        @param df: The dataset DataFrame.
        @param path: The path to save the dataset.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        self.model_dataset_data_provider.save_dataset_as_pickle(self.dataset, path)

    # --- Model lifecycle helpers ---
    def attach_model(self, model: ModelInterface) -> None:
        self.model = model

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not attached. Call attach_model(model) first.")
        df = self.model.predict(dataset)

        # Convert model output (speed, direction) to u/v components
        # The model outputs average_wind_speed_pred and average_wind_direction_pred
        if (
            "average_wind_speed_pred" in df.columns
            and "average_wind_direction_pred" in df.columns
        ):
            # Convert direction from degrees to radians
            direction_rad = np.deg2rad(df["average_wind_direction_pred"])
            # Calculate u and v components using meteorological convention
            df["u"] = -df["average_wind_speed_pred"] * np.sin(direction_rad)
            df["v"] = -df["average_wind_speed_pred"] * np.cos(direction_rad)

            # Expand hourly predictions back to 10-minute intervals
            # Each hourly prediction becomes 6 10-minute intervals with the same values
            expanded_rows = []
            for _, row in df.iterrows():
                station_id = row["station_id"]
                hourly_time = pd.to_datetime(row["record_date"])
                u_val = row["u"]
                v_val = row["v"]

                # Create 6 10-minute intervals for this hour
                for i in range(6):
                    ten_min_time = hourly_time + pd.Timedelta(minutes=i * 10)
                    expanded_rows.append(
                        {
                            "station_id": station_id,
                            "record_date": ten_min_time,
                            "u": u_val,
                            "v": v_val,
                        }
                    )

            df = pd.DataFrame(expanded_rows)
        else:
            raise ValueError(
                f"Model output missing required columns. Expected 'average_wind_speed_pred' and "
                f"'average_wind_direction_pred', got: {df.columns.tolist()}"
            )

        return df

    def train_model(self, save_path: str | None = None) -> None:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not attached. Call attach_model(model) first.")
        self.model.train(self.dataset)
        if save_path:
            self.model.save(save_path)

    def load_model(self, path: str) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not attached. Call attach_model(model) first.")
        self.model.load(path)
