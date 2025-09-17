from omegaconf import OmegaConf
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from src.database.database_service import DatabaseService
import pandas as pd


class AggregationDataProvider:
    """
    The data provider provides the functionality for creating the aggregation dataset.
    """

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service

    def get_aggregated_data_for_last_24_hours(self) -> pd.DataFrame:
        query = text(
            """
            WITH max_ts AS (
              SELECT max(record_date) AS max_record_date
              FROM wind_power_calculations
            ),
            filtered AS (
              SELECT
                w.*
              FROM wind_power_calculations w
              CROSS JOIN max_ts
              WHERE w.record_date > (max_record_date - INTERVAL '24 hours')
                AND w.record_date <= max_record_date
            )
            SELECT
              record_date,
              CASE WHEN SUM(CASE WHEN is_prediction THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_prediction,
              AVG(NULLIF(extrapolated_hub_height_wind_speed, 'NaN'::double precision)) AS avg_extrapolated_hub_height_wind_speed,
              SUM(NULLIF(pred_power_production, 'NaN'::double precision)) AS sum_pred_power_production,
              COUNT(*) AS num_rows
            FROM filtered
            GROUP BY record_date
            ORDER BY record_date ASC
            """
        )

        with Session(self.database_service.engine) as session:
            result = session.execute(query).mappings().all()
            df = pd.DataFrame(result)
            return df

    def get_data_for_one_turbine_for_last_24_hours(
        self, unit_mastr_number: str
    ) -> pd.DataFrame:
        query = text(
            """
            WITH max_ts AS (
              SELECT max(record_date) AS max_record_date
              FROM wind_power_calculations
            ),
            filtered AS (
              SELECT
                w.*
              FROM wind_power_calculations w
              CROSS JOIN max_ts
              WHERE w.record_date > (max_record_date - INTERVAL '24 hours')
                AND w.record_date <= max_record_date
            )
            SELECT
              record_date,
              is_prediction,
              extrapolated_hub_height_wind_speed,
              pred_power_production
            FROM filtered
            WHERE unit_mastr_number = :unit_mastr_number
            ORDER BY record_date ASC
            """
        ).bindparams(bindparam("unit_mastr_number", unit_mastr_number))

        with Session(self.database_service.engine) as session:
            result = session.execute(query).mappings().all()
            df = pd.DataFrame(result)
            return df
