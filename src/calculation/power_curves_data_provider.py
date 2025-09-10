from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import re
import time

from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database.schema import TurbinePowerCurves
from src.database.database_service import DatabaseService
from sqlalchemy.dialects.postgresql import insert as pg_insert


class PowerCurvesDataProvider:

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service
        pass

    def save_power_curves_to_database(self, path: str = "data/power_curves.csv"):
        df = pd.read_csv(
            path,
            sep=",",
            quotechar="'",
            engine="python",
            na_values=["#ND"],
        )
        df.columns = [self._convert_column_name(str(c)) for c in df.columns]

        # # Convert power curve columns to numeric (leave id/name/conditions as strings)
        # non_numeric_columns = {
        #     "manufacturer_id",
        #     "manufacturer_name",
        #     "turbine_id",
        #     "turbine_name",
        #     "conditions_nd_unknown",
        # }
        # numeric_columns = [c for c in df.columns if c not in non_numeric_columns]
        # df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        table = TurbinePowerCurves.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in df.to_dict(orient="records")
        ]

        if not records:
            logger.warning("No power curves to upsert")
            return

        chunk_size = 200
        max_retries = 3
        total = len(records)

        for start in range(0, total, chunk_size):
            chunk = records[start : start + chunk_size]
            attempt = 0
            while True:
                try:
                    stmt = pg_insert(table).values(chunk)
                    update_columns = {
                        c.name: stmt.excluded[c.name]
                        for c in table.columns
                        if c.name not in {"id"}
                    }
                    # Use named unique constraint (manufacturer_id, turbine_id)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uix_powercurve_manu_turbine",
                        set_=update_columns,
                    )

                    with Session(self.database_service.engine) as session:
                        session.execute(stmt)
                        session.commit()

                    logger.info(
                        f"Upserted power curves chunk {start}-{min(start+chunk_size, total)} of {total}"
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(
                            f"Failed upserting power curves chunk {start}-{min(start+chunk_size, total)} after {max_retries} attempts: {e}"
                        )
                        raise
                    sleep_s = 2 ** (attempt - 1)
                    logger.warning(
                        f"Error during upsert (attempt {attempt}/{max_retries}) for chunk {start}-{min(start+chunk_size, total)}: {e}. Retrying in {sleep_s}s"
                    )
                    time.sleep(sleep_s)

        logger.info(f"Upserted {total} power curves to database")

    def load_from_database(self) -> pd.DataFrame:
        """
        Load the power curves from the database.
        """
        with Session(self.database_service.engine) as session:
            table = TurbinePowerCurves.__table__
            query = select(table)
            rows = session.execute(query).mappings().all()
            df = pd.DataFrame(rows)
            logger.info(f"Loaded {len(df)} power curves from database")
            return df

    def _convert_column_name(self, column_name: str) -> str:
        """Convert CSV header to the desired format.

        - "kW at X m/s" -> speed label: "26" for 26.0, "10_5" for 10.5
        - Otherwise -> snake_case
        """
        cleaned = column_name.strip().strip("'")

        match = re.fullmatch(r"kW at (\d+(?:\.\d)?) m/s", cleaned)
        if match:
            speed = match.group(1)
            if speed.endswith(".0"):
                return str(int(float(speed)))
            return speed.replace(".", "_")

        return self._to_snake_case(cleaned)

    def _to_snake_case(self, name: str) -> str:
        name = name.strip()
        # Replace separators and remove non-alphanumeric (keep spaces/underscores for word splitting)
        name = name.replace("/", " ")
        name = name.replace("-", " ")
        name = name.replace("(", " ")
        name = name.replace(")", " ")
        name = name.replace("#", " ")
        name = re.sub(r"[^A-Za-z0-9_\s]", "", name)
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.lower()
