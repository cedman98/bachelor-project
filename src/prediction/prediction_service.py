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

    def predict_measurements(self, upsert_to_database: bool = True) -> pd.DataFrame:
        # 1. Load the last 24 hours of 10-minute measurements, aggregated to hourly for model input
        measurements_df = (
            self.measurement_service.load_all_recent_measurements_from_database()
        )
        # 2. Predict the measurements
        predictions_df = self.model_service.predict(measurements_df)
        # 3. Save the predictions to the database
        if upsert_to_database:
            self.prediction_data_provider.save_predictions_to_database(predictions_df)
        # 4. Transform the measurements to the prediction format
        measurements_df = (
            self.measurement_service.transform_measurements_to_prediction_format(
                measurements_df
            )
        )
        # 5. Concate the measurements and predictions, but add a column to identify the predictions
        predictions_df["is_prediction"] = True
        measurements_df["is_prediction"] = False
        combined_df = pd.concat([measurements_df, predictions_df])
        combined_df = combined_df.sort_values(by=["record_date", "station_id"])

        # 5. Return the predictions
        return combined_df

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass
