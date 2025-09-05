from datetime import datetime
import io
import time
import zipfile
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.schema import WindStationMeasurements
from src.database.database_service import DatabaseService

from hamilton import driver as h_driver
from src.measurements import measurement_preprocess as mp


class MeasurementDataProvider:
    """
    The data provider offers functions for downloading the current measurements and the historical measurements from DWD. It also processes the data and saves it to the database.
    """

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service

    def download_measurements_for_weather_station(
        self, weather_station_id: int, only_now: bool = False
    ):
        """
        Download and combine wind and air temperature measurements for a specific weather station.
        The air temperature dataset uses base path segment 'air_temperature' and file prefix 'TU'.
        The wind dataset uses base path segment 'wind' and file prefix 'wind'.
        Both datasets are merged on ['STATIONS_ID', 'MESS_DATUM'].
        @param weather_station_id: The weather station id.
        @param only_now: If True, only get the now data (current day).
        @return: The merged dataframe containing columns from both datasets when available.
        """
        # Download per-dataset frames
        wind_df = self._download_dataset_for_station(
            weather_station_id=weather_station_id, dataset="wind", only_now=only_now
        )

        air_temp_df = self._download_dataset_for_station(
            weather_station_id=weather_station_id,
            dataset="air_temperature",
            only_now=only_now,
        )

        precipitation_df = self._download_dataset_for_station(
            weather_station_id=weather_station_id,
            dataset="precipitation",
            only_now=only_now,
        )

        # Always ensure we don't create duplicate QN/eor columns during merge
        # Drop from secondary datasets regardless of emptiness
        air_temp_df.drop(columns=["eor", "QN"], inplace=True, errors="ignore")
        precipitation_df.drop(columns=["eor", "QN"], inplace=True, errors="ignore")

        if wind_df.empty and air_temp_df.empty and precipitation_df.empty:
            logger.warning(
                f"No dataframes downloaded for weather station {weather_station_id} (wind, air_temperature, precipitation)"
            )
            # Continue to merge below to return an empty df with all expected columns

        # Merge on station and timestamp. Always include all datasets so that
        # missing datasets still contribute their columns (filled with NaN here).
        dataframes = [wind_df, air_temp_df, precipitation_df]

        # Initialize with empty frame that has join keys so merges never fail
        merged_df = pd.DataFrame(columns=["STATIONS_ID", "MESS_DATUM"])
        for df in dataframes:
            if df is None or df.empty:
                continue
            if not {"STATIONS_ID", "MESS_DATUM"}.issubset(df.columns):
                logger.warning(
                    "Skipping merge for dataset missing join keys ['STATIONS_ID','MESS_DATUM']"
                )
                continue
            merged_df = pd.merge(
                merged_df,
                df,
                on=["STATIONS_ID", "MESS_DATUM"],
                how="outer",
                copy=False,
            )

        # Ensure all expected original columns exist so downstream processing can rename
        expected_columns = [
            "STATIONS_ID",
            "MESS_DATUM",
            "QN",
            "eor",
            # wind
            "FF_10",
            "DD_10",
            # air pressure/temperature/humidity
            "PP_10",
            "TT_10",
            "TM5_10",
            "RF_10",
            "TD_10",
            # precipitation
            "RWS_DAU_10",
            "RWS_10",
            "RWS_IND_10",
        ]
        for col in expected_columns:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA

        # Final de-dup per station and timestamp
        merged_df = merged_df.drop_duplicates(
            subset=["STATIONS_ID", "MESS_DATUM"], keep="last"
        ).reset_index(drop=True)

        return merged_df

    def process_measurement_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the measurement dataframe using Hamilton dataflow defined in
        `src.measurements.measurement_preprocess`. The final DataFrame remains
        identical in structure and typing to the previous implementation.
        """

        dr = h_driver.Driver({}, mp)

        # Specify the final columns explicitly; this controls renaming & dropping.
        final_columns = [
            "station_id",
            "record_date",
            "average_wind_speed",
            "average_wind_direction",
            "air_pressure",
            "air_temperature_2m",
            "air_temperature_5cm",
            "relative_humidity",
            "dew_point_temperature",
            "precipitation_duration",
            "sum_precipitation_height",
            "precipitation_indicator",
            "quality_level",
        ]

        # We also compute the mask to drop invalid rows (no station_id or record_date), but don't keep it as a final column.
        outputs = final_columns + ["valid_row_mask"]
        result = dr.execute(outputs, inputs={"raw_df": df})

        # Convert to DataFrame and filter invalid rows, mirroring previous logic.
        out_df = pd.DataFrame(result)
        out_df = out_df[out_df["valid_row_mask"]].copy()
        out_df.drop(columns=["valid_row_mask"], inplace=True)

        return out_df.reset_index(drop=True)

    def save_measurement_df_to_database(self, df: pd.DataFrame) -> None:
        """
        Save the measurement DataFrame to the database.
        @param df: The measurement DataFrame.
        """
        table = WindStationMeasurements.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in df.to_dict(orient="records")
        ]

        if not records:
            logger.warning("No measurements to upsert")
            return

        chunk_size = getattr(
            getattr(self.cfg, "database", object()),
            "measurement_upsert_chunk_size",
            5000,
        )
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
                    stmt = stmt.on_conflict_do_update(
                        constraint="uix_station_date",
                        set_=update_columns,
                    )

                    with Session(self.database_service.engine) as session:
                        session.execute(stmt)
                        session.commit()

                    logger.info(
                        f"Upserted measurements chunk {start}-{min(start+chunk_size, total)} of {total}"
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(
                            f"Failed upserting measurements chunk {start}-{min(start+chunk_size, total)} after {max_retries} attempts: {e}"
                        )
                        raise
                    sleep_s = 2 ** (attempt - 1)
                    logger.warning(
                        f"Error during upsert (attempt {attempt}/{max_retries}) for chunk {start}-{min(start+chunk_size, total)}: {e}. Retrying in {sleep_s}s"
                    )
                    time.sleep(sleep_s)

    def load_measurements_from_database_for_datetime(
        self, weather_stations: pd.DataFrame, datetime: datetime
    ) -> pd.DataFrame:
        """
        Load the measurements from the database for a specific datetime.
        @param datetime: The datetime.
        @return: The measurements DataFrame.
        """
        table = WindStationMeasurements.__table__
        query = (
            select(table)
            .where(table.c.record_date == datetime)
            .where(
                table.c.station_id.in_(weather_stations["weather_station_id"].tolist())
            )
        )

        with Session(self.database_service.engine) as session:
            rows = session.execute(query).mappings().all()
            df = pd.DataFrame(rows)
            logger.info(
                f"Loaded {len(df)} measurements from database for datetime {datetime}"
            )

            # Drop rows where average_wind_direction or average_wind_speed is -999
            df = df[
                (df["average_wind_direction"] != -999)
                & (df["average_wind_speed"] != -999)
            ]

            return df

    def _get_download_urls(
        self, weather_station_id: int, only_now: bool = False, dataset: str = "wind"
    ) -> list[str]:
        """
        Get the download urls for the measurements.
        @param weather_station_id: The weather station id.
        @param only_now: If True, only get the now download url.
        @param dataset: Either 'wind' or 'air_temperature' or 'precipitation'. Determines base URL and filename prefix.
        @return: The download urls.
        """
        # Fill with pre zeros to have lentgh 5
        weather_station_id = str(weather_station_id).zfill(5)

        # Determine base url and filename prefix by dataset
        if dataset == "wind":
            base_url = f"{self.cfg.dwd.measurements_base_url}wind/"
            file_prefix = "10minutenwerte_wind"
        elif dataset == "air_temperature":
            base_url = f"{self.cfg.dwd.measurements_base_url}air_temperature/"
            file_prefix = "10minutenwerte_TU"
        elif dataset == "precipitation":
            base_url = f"{self.cfg.dwd.measurements_base_url}precipitation/"
            file_prefix = "10minutenwerte_nieder"
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. Expected 'wind' or 'air_temperature' or 'precipitation'"
            )

        if only_now:
            return [
                f"{base_url}now/{file_prefix}_{weather_station_id}_now.zip",
            ]
        else:
            return [
                f"{base_url}historical/{file_prefix}_{weather_station_id}_20200101_20241231_hist.zip",
                f"{base_url}recent/{file_prefix}_{weather_station_id}_akt.zip",
                f"{base_url}now/{file_prefix}_{weather_station_id}_now.zip",
            ]

    def _download_dataset_for_station(
        self, weather_station_id: int, dataset: str, only_now: bool
    ) -> pd.DataFrame:
        """
        Download and combine all chunks for a specific dataset (wind or air_temperature) for a station.
        Deduplicate by ['STATIONS_ID', 'MESS_DATUM'] keeping the newest occurrence.
        """
        all_dfs: list[pd.DataFrame] = []
        for download_url in self._get_download_urls(
            weather_station_id, only_now, dataset=dataset
        ):
            try:
                df = self._download_file(download_url)
                # Normalize columns
                df.columns = [
                    c.strip() if isinstance(c, str) else c for c in df.columns
                ]
                all_dfs.append(df)

            except Exception as e:
                logger.warning(f"Failed to download {dataset} from {download_url}: {e}")
                continue

        if not all_dfs:
            logger.warning(
                f"No {dataset} dataframes downloaded for weather station {weather_station_id}"
            )
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=["STATIONS_ID", "MESS_DATUM"], keep="last"
        ).reset_index(drop=True)

        return combined_df

    def _download_file(self, download_url: str) -> pd.DataFrame:
        """
        Download the file from the download url.
        @param download_url: The download url.
        @return: The dataframe.
        """
        response = requests.get(download_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Read the CSV file directly from the zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            file_list = zip_file.namelist()

            txt_files = [f for f in file_list if f.endswith(".txt")]

            if not txt_files:
                raise ValueError("No TXT file found in the zip archive")

            if len(txt_files) > 1:
                logger.warning(f"Multiple txt files found: {txt_files}")
                logger.warning(f"Using the first one: {txt_files[0]}")

            txt_filename = txt_files[0]

            with zip_file.open(txt_filename) as txt_file:
                df = pd.read_csv(txt_file, sep=";", encoding="latin-1")

                logger.info(
                    f"Downloaded file from {download_url} as dataframe with {len(df)} rows"
                )
                return df
