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
        Download the measurements from DWD for a specific weather station.
        @param weather_station_id: The weather station id.
        @param only_now: If True, only get the now data (current day).
        @return: The dataframe.
        """
        all_dfs = []
        for download_url in self._get_download_urls(weather_station_id, only_now):
            try:
                df = self._download_file(download_url)
                # Ensure column names have no leading/trailing spaces (e.g., '  QN' -> 'QN')
                df.columns = [
                    c.strip() if isinstance(c, str) else c for c in df.columns
                ]
                all_dfs.append(df)
                logger.info(
                    f"Successfully downloaded and processed data from {download_url}"
                )
            except Exception as e:
                logger.warning(f"Failed to download from {download_url}: {e}")
                continue

        if len(all_dfs) == 0:
            logger.warning(
                f"No dataframes downloaded for weather station {weather_station_id}"
            )
            return pd.DataFrame()

        logger.info(
            f"Downloaded {len(all_dfs)} dataframes for weather station {weather_station_id}"
        )

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Handle duplicates: keep the row with the most recent data (assumes later sources are more recent)
        # Sort by MESS_DATUM and drop duplicates, keeping the last occurrence (newest)
        combined_df = combined_df.drop_duplicates(
            subset=["MESS_DATUM"], keep="last"
        ).reset_index(drop=True)

        logger.info(
            f"Combined dataframe with {len(combined_df)} rows for weather station {weather_station_id}"
        )

        return combined_df

    def process_measurement_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the measurement dataframe.
        @param df: The measurement dataframe.
        @return: The processed dataframe.
        """
        # Fix column names
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        df["record_date"] = pd.to_datetime(
            df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce"
        )

        df.rename(
            columns={
                "FF_10": "average_wind_speed",
                "DD_10": "average_wind_direction",
                "QN": "quality_level",
                "STATIONS_ID": "station_id",
            },
            inplace=True,
        )

        df.drop(columns=["eor", "MESS_DATUM"], inplace=True)

        # Set data types
        df["station_id"] = df["station_id"].astype(int)
        df["quality_level"] = df["quality_level"].astype(int)
        df["average_wind_speed"] = df["average_wind_speed"].astype(float)
        df["average_wind_direction"] = df["average_wind_direction"].astype(int)

        logger.info(
            f"Processed {len(df)} measurements for weather station {df['station_id'].iloc[0]}"
        )

        return df

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
            logger.info("No measurements to upsert")
            return

        chunk_size = getattr(
            getattr(self.cfg, "database", object()),
            "measurement_upsert_chunk_size",
            10000,
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

        logger.info(
            f"Upserted {total} measurements to database in chunks of {chunk_size}"
        )

    def _get_download_urls(
        self, weather_station_id: int, only_now: bool = False
    ) -> list[str]:
        """
        Get the download urls for the measurements.
        @param weather_station_id: The weather station id.
        @param only_now: If True, only get the now download url.
        @return: The download urls.
        """
        # Fill with pre zeros to have lentgh 5
        weather_station_id = str(weather_station_id).zfill(5)

        if only_now:
            return [
                f"{self.cfg.dwd.base_url}now/10minutenwerte_wind_{weather_station_id}_now.zip",
            ]
        else:
            return [
                f"{self.cfg.dwd.base_url}historical/10minutenwerte_wind_{weather_station_id}_20200101_20241231_hist.zip",
                f"{self.cfg.dwd.base_url}recent/10minutenwerte_wind_{weather_station_id}_akt.zip",
                f"{self.cfg.dwd.base_url}now/10minutenwerte_wind_{weather_station_id}_now.zip",
            ]

    def _download_file(self, download_url: str) -> pd.DataFrame:
        """
        Download the file from the download url.
        @param download_url: The download url.
        @return: The dataframe.
        """
        logger.info(f"Start downloading file from {download_url}")
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
