from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import re
import time

from sqlalchemy import select, text
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

    def get_all_turbine_id_for_mastr_number(self, mastr_number: str) -> pd.DataFrame:
        """
        Get all turbine ids for a given mastr number. It does this by checking all finding the closest value for the type designation.
        """
        with Session(self.database_service.engine) as session:
            query = text(
                """
                WITH register AS (
                  SELECT
                    r.unit_mastr_number,
                    r.manufacturer        AS register_manufacturer_id,
                    r.type_designation    AS register_type_designation,
                    r.net_nominal_power   AS register_power_kw,
                    r.rotor_diameter,
                    r.hub_height
                  FROM wind_turbines r
                ),
                curves AS (
                  SELECT
                    c.manufacturer_id,
                    c.manufacturer_name,
                    c.turbine_id,
                    c.turbine_name,
                    NULLIF(regexp_replace(c.turbine_name, '[^0-9]', '', 'g'), '')::int
                      AS curve_numeric_hint
                  FROM turbine_power_curves c
                ),
                manufacturer_map AS (
                  SELECT *
                  FROM (VALUES
                    (1614,42),(1657,63),(1654,76),(2888,9),(1001679,218),
                    (1592,51),(1596,4),(1587,87),(1646,10),(1652,75),
                    (1001684,87),(1595,155),(1660,14),(1627,8),(1586,3),
                    (2884,61),(1001682,8),(1628,8),(1584,1),(2873,5),
                    (1645,154),(2892,82),(1597,5),(1666,29),(1625,41),
                    (1593,44),(2890,39)
                  ) AS m(register_manufacturer_id, curve_manufacturer_id)
                ),
                prep AS (
                  SELECT
                    r.unit_mastr_number,
                    r.register_manufacturer_id,
                    mm.curve_manufacturer_id,
                    r.register_type_designation,
                    lower(regexp_replace(r.register_type_designation, '\s+', ' ', 'g')) AS reg_norm
                  FROM register r
                  JOIN manufacturer_map mm
                    ON mm.register_manufacturer_id = r.register_manufacturer_id
                ),
                candidates AS (
                  SELECT
                    p.unit_mastr_number,
                    p.register_manufacturer_id,
                    p.curve_manufacturer_id,
                    p.register_type_designation,
                    p.reg_norm,
                    c.turbine_id,
                    c.turbine_name,
                    lower(regexp_replace(c.turbine_name, '\s+', ' ', 'g')) AS curve_norm,
                    similarity(p.reg_norm,
                               lower(regexp_replace(c.turbine_name, '\s+', ' ', 'g'))) AS name_sim,
                    CASE
                      WHEN regexp_replace(p.register_type_designation, '\D', '', 'g')
                           = regexp_replace(c.turbine_name, '\D', '', 'g')
                      THEN 0.05 ELSE 0 END AS numeric_bonus
                  FROM prep p
                  JOIN curves c
                    ON c.manufacturer_id = p.curve_manufacturer_id
                ),
                scored AS (
                  SELECT
                    unit_mastr_number,
                    register_manufacturer_id,
                    curve_manufacturer_id,
                    register_type_designation,
                    turbine_id,
                    turbine_name,
                    reg_norm,
                    curve_norm,
                    name_sim + numeric_bonus AS score
                  FROM candidates
                  WHERE similarity(reg_norm, curve_norm) >= 0.2
                ),
                ranked AS (
                  SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY s.unit_mastr_number
                      ORDER BY s.score DESC
                    ) AS rn
                  FROM scored s
                )
                SELECT
                  r.unit_mastr_number,
                  r.register_manufacturer_id,
                  r.register_type_designation,
                  pc.turbine_id       AS matched_turbine_id,
                  pc.turbine_name     AS matched_turbine_name,
                  pc.curve_manufacturer_id AS matched_manufacturer_id,
                  pc.score,
                  CASE
                    WHEN pc.score >= 0.75 THEN 'high'
                    WHEN pc.score >= 0.55 THEN 'medium'
                    WHEN pc.score >= 0.40 THEN 'low'
                    ELSE 'very_low'
                  END AS confidence_band
                FROM prep r
                LEFT JOIN LATERAL (
                  SELECT turbine_id, turbine_name, curve_manufacturer_id, score
                  FROM ranked
                  WHERE ranked.unit_mastr_number = r.unit_mastr_number
                    AND rn = 1
                ) pc ON TRUE
                ORDER BY r.unit_mastr_number;
                """
            )

            results = session.execute(query).mappings().all()
            df = pd.DataFrame(results)
            logger.info(f"Loaded {len(df)} turbine ids")
            return df

    def calculate_wind_power_production(
        self,
        measurements_df: pd.DataFrame,
        matched_df: pd.DataFrame,
        power_curves_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate the wind power production for all wind turbines.
        @param measurements_df: The measurements DataFrame.
        @param matched_df: The matched DataFrame.
        @param power_curves_df: The power curves DataFrame.
        @return: The power production DataFrame.
        """
        # Validate required columns
        required_meas = {"unit_mastr_number", "hub_height_wind_speed"}
        if not required_meas.issubset(measurements_df.columns):
            missing = required_meas - set(measurements_df.columns)
            raise ValueError(
                f"measurements_df missing required columns: {sorted(missing)}"
            )

        required_match = {"unit_mastr_number", "matched_turbine_id"}
        if not required_match.issubset(matched_df.columns):
            missing = required_match - set(matched_df.columns)
            raise ValueError(f"matched_df missing required columns: {sorted(missing)}")

        if "turbine_id" not in power_curves_df.columns:
            raise ValueError("power_curves_df must contain column 'turbine_id'")

        # Identify speed columns like "0", "0_5", "1", "1_5", ...
        speed_cols = [
            c for c in power_curves_df.columns if re.fullmatch(r"\d+(?:_\d+)?", str(c))
        ]
        if not speed_cols:
            logger.warning("No speed columns found in power_curves_df. Returning NaNs.")
            result_df = measurements_df.copy()
            result_df["pred_power_production"] = pd.NA
            return result_df

        # Map column label -> numeric speed (e.g., "10_5" -> 10.5)
        def _col_to_speed(col: str) -> float:
            label = str(col)
            return float(label.replace("_", "."))

        speed_values = [
            _col_to_speed(col) for col in speed_cols  # type: ignore[arg-type]
        ]
        min_speed = min(speed_values)
        max_speed = max(speed_values)

        # Prepare long-form lookup: (turbine_id, speed_col) -> pred_power_production
        power_long = power_curves_df[["turbine_id", *speed_cols]].melt(
            id_vars=["turbine_id"],
            var_name="speed_col",
            value_name="pred_power_production",
        )
        # Ensure consistent types
        power_long["turbine_id"] = pd.to_numeric(
            power_long["turbine_id"], errors="coerce"
        ).astype("Int64")
        power_long["pred_power_production"] = pd.to_numeric(
            power_long["pred_power_production"], errors="coerce"
        )

        # Prepare matched mapping
        matched_map = matched_df[
            ["unit_mastr_number", "matched_turbine_id"]
        ].drop_duplicates()
        matched_map["matched_turbine_id"] = pd.to_numeric(
            matched_map["matched_turbine_id"], errors="coerce"
        ).astype("Int64")

        # Build a working frame to keep original row order
        work = measurements_df[["unit_mastr_number", "hub_height_wind_speed"]].copy()
        work["row_idx"] = measurements_df.index

        # Attach matched turbine id
        work = work.merge(matched_map, on="unit_mastr_number", how="left")

        # Compute closest speed column label per measurement (round to nearest 0.5 m/s)
        speeds = pd.to_numeric(work["hub_height_wind_speed"], errors="coerce")
        speeds = speeds.clip(lower=min_speed, upper=max_speed)

        # Round to nearest 0.5 without floating point glitches using x2 integer arithmetic
        # s2 represents speed*2 rounded to nearest integer
        s2 = (speeds * 2).round().astype("Int64")
        int_part = (s2 // 2).astype("Int64")
        is_half = s2 % 2 != 0
        speed_col = int_part.astype("string")
        speed_col = speed_col.mask(is_half, int_part.astype("string") + "_5")
        work["speed_col"] = speed_col

        # Merge to fetch predicted power
        merged = work.merge(
            power_long,
            left_on=["matched_turbine_id", "speed_col"],
            right_on=["turbine_id", "speed_col"],
            how="left",
        )

        pred_series = merged.set_index("row_idx")["pred_power_production"]

        # Return original measurements with additional column
        result_df = measurements_df.copy()
        result_df["pred_power_production"] = pred_series.reindex(measurements_df.index)
        return result_df

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
