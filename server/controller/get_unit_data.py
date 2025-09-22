import os
import sys
from omegaconf import DictConfig


# Ensure project root is on PYTHONPATH so `src.*` imports work when running from `server/`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.wind_turbines.wind_turbines_service import WindTurbinesService
from src.database.database_service import DatabaseService


def get_unit_data(
    cfg: DictConfig, database_service: DatabaseService, unit_mastr_number: str
):

    wind_turbines_service = WindTurbinesService(cfg, database_service)

    df = wind_turbines_service.load_one_unit_from_database(unit_mastr_number)

    return df.to_dict(orient="records")
