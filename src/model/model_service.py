from loguru import logger
from omegaconf import OmegaConf
from src.model.variant.model_interface import ModelInterface
from src.model.model_dataset_data_provider import ModelDatasetDataProvider
from src.database.database_service import DatabaseService
from src.measurements.measurement_service import MeasurementService
import pandas as pd


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
