from omegaconf import OmegaConf

from src.aggregation.aggregation_data_provider import AggregationDataProvider
from src.database.database_service import DatabaseService
import pandas as pd


class AggregationService:

    cfg: OmegaConf
    database_service: DatabaseService
    aggregation_data_provider: AggregationDataProvider

    def __init__(
        self,
        cfg: OmegaConf,
        database_service: DatabaseService,
        aggregation_data_provider: AggregationDataProvider = None,
    ):
        self.cfg = cfg
        self.database_service = database_service
        self.aggregation_data_provider = (
            aggregation_data_provider
            if aggregation_data_provider
            else AggregationDataProvider(cfg, database_service)
        )

    def get_aggregated_data_for_last_24_hours(self) -> pd.DataFrame:
        return self.aggregation_data_provider.get_aggregated_data_for_last_24_hours()

    def get_data_for_one_turbine_for_last_24_hours(
        self, mastr_number: str
    ) -> pd.DataFrame:
        return (
            self.aggregation_data_provider.get_data_for_one_turbine_for_last_24_hours(
                mastr_number
            )
        )
