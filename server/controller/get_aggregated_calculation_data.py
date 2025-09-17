import os
import sys
from omegaconf import DictConfig

# Ensure project root is on PYTHONPATH so `src.*` imports work when running from `server/`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.aggregation.aggregation_service import AggregationService
from src.database.database_service import DatabaseService
import pandas as pd


def get_aggregated_calculation_data(cfg: DictConfig, database_service: DatabaseService):

    aggregation_service = AggregationService(cfg, database_service)
    aggregated_data = aggregation_service.get_aggregated_data_for_last_24_hours()

    # Ensure record_date is converted to string (ISO format) for JSON serialization
    if "record_date" in aggregated_data.columns:
        aggregated_data["record_date"] = aggregated_data["record_date"].apply(
            lambda x: x.isoformat() if pd.notnull(x) else None
        )

    # Convert DataFrame to dict with record_date as key
    records = aggregated_data.to_dict(orient="records")
    result = {record["record_date"]: record for record in records}
    return result
