from omegaconf import OmegaConf
from src.prediction.prediction_data_provider import PredictionDataProvider
from src.database.database_service import DatabaseService
from src.measurements.measurement_service import MeasurementService
from src.model.model_service import ModelService
import pandas as pd


class PredictionService:
    """
    The service provides the functionality for predicting the measurements.
    """

    cfg: OmegaConf
    database_service: DatabaseService
    measurement_service: MeasurementService
    model_service: ModelService
    prediction_data_provider: PredictionDataProvider

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        measurement_service: MeasurementService,
        model_service: ModelService,
        prediction_data_provider: PredictionDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.measurement_service = measurement_service
        self.model_service = model_service
        self.prediction_data_provider = (
            prediction_data_provider
            if prediction_data_provider
            else PredictionDataProvider(cfg, database_service)
        )

    def predict_measurements(self) -> pd.DataFrame:
        # 1. Load the last 72 measurements for each station
        measurements_df = (
            self.measurement_service.load_all_recent_measurements_from_database()
        )
        # 2. Predict the measurements
        predictions_df = self.model_service.predict(measurements_df)
        # 3. Save the predictions to the database
        self.prediction_data_provider.save_predictions_to_database(predictions_df)
        # 4. Return the predictions
        return predictions_df

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
