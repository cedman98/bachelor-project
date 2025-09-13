from loguru import logger
from omegaconf import OmegaConf
from sqlalchemy.orm import Session
from src.database.schema import (
    WindStationMeasurements,
    WindStationMeasurementsPrediction,
)
from src.database.database_service import DatabaseService
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert


class PredictionDataProvider:
    """
    The data provider provides the functionality for creating the prediction dataset.
    """

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service
        pass

    def save_predictions_to_database(self, dataset: pd.DataFrame) -> pd.DataFrame:
        table = WindStationMeasurementsPrediction.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in dataset.to_dict(orient="records")
        ]

        if not records:
            logger.warning("No predictions to upsert")
            return

        chunk_size = 200
        total = len(records)
        for start in range(0, total, chunk_size):
            chunk = records[start : start + chunk_size]
            stmt = pg_insert(table).values(chunk)
            update_columns = {
                c.name: stmt.excluded[c.name]
                for c in table.columns
                if c.name not in {"id"}
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uix_station_date_prediction",
                set_=update_columns,
            )
            with Session(self.database_service.engine) as session:
                session.execute(stmt)
                session.commit()

            logger.info(f"Upserted {len(chunk)} predictions to database")
