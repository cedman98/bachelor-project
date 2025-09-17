from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import requests
from sqlalchemy.orm import Session
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.schema import WeatherStations
from src.database.database_service import DatabaseService
from hamilton import driver as h_driver
from src.weather_stations import weather_station_preprocess as wsp


class WeatherStationDataProvider:
    """
    The service offers functions for downloading the weather stations file, parsing it, processing it and saving it to the database.
    """

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service

    def download_weather_stations_file(self) -> str:
        """
        Download the weather stations file from the DWD.
        @return: The weather stations file as a string.
        """
        response = requests.get(self.cfg.dwd.weather_stations_url)
        response.raise_for_status()

        logger.info("Downloaded weather stations file")

        return response.content.decode("latin-1")

    def parse_weather_stations_file(self, weather_stations_file) -> pd.DataFrame:
        """
        Parse the weather stations file and return a pandas DataFrame.
        @param weather_stations_file: The weather stations file as a string.
        @return: A pandas DataFrame with the weather stations.
        """
        # Split lines and remove header and separator
        lines = weather_stations_file.strip().split("\n")

        # Find the data lines (skip header and separator)
        data_lines = []
        for i, line in enumerate(lines):
            if i < 2:  # Skip header and separator line
                continue
            if line.strip():  # Only non-empty lines
                data_lines.append(line)

        # Parse each line by splitting on whitespace
        parsed_data = []
        for line in data_lines:
            # Split on whitespace and filter out empty strings
            parts = [p for p in line.split() if p]

            if len(parts) >= 6:  # Minimum expected parts
                try:
                    # Basic structure: station_id von_datum bis_datum height lat lon name... state    [abgabe]
                    weather_station_id = parts[0]
                    start_date = parts[1]
                    end_date = parts[2]
                    height = parts[3]
                    latitude = parts[4]
                    longitude = parts[5]

                    # Reconstruct name and find state
                    # All German states for identification
                    states = [
                        "Baden-Württemberg",
                        "Bayern",
                        "Brandenburg",
                        "Hessen",
                        "Niedersachsen",
                        "Nordrhein-Westfalen",
                        "Rheinland-Pfalz",
                        "Schleswig-Holstein",
                        "Thüringen",
                        "Mecklenburg-Vorpommern",
                        "Sachsen",
                        "Sachsen-Anhalt",
                        "Berlin",
                        "Bremen",
                        "Hamburg",
                        "Saarland",
                    ]

                    # Join all text parts and then separate name/state/abgabe
                    full_text = " ".join(parts[6:])

                    name = None
                    state = None
                    accessible = None

                    # Look for state names in the text
                    state_found = False
                    for state in states:
                        if state in full_text:
                            # Split by state name
                            before_state = full_text.split(state)[0].strip()
                            after_state = (
                                full_text.split(state, 1)[1].strip()
                                if state in full_text
                                else ""
                            )

                            name = before_state
                            state = state
                            accessible = after_state if after_state else None
                            state_found = True
                            break

                    if not state_found:
                        raise ValueError(f"State not found in line: {line}")

                    parsed_data.append(
                        {
                            "weather_station_id": weather_station_id,
                            "start_date": start_date,
                            "end_date": end_date,
                            "height": height,
                            "latitude": latitude,
                            "longitude": longitude,
                            "name": name,
                            "state": state,
                            "accessible": accessible,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error parsing line: {line}")
                    logger.error(f"Error: {e}")
                    raise ValueError(f"Could not parse line: {line}")
            else:
                logger.error(f"Not enough parts in line: {line}")
                raise ValueError(f"Could not parse line: {line}")

        # Create DataFrame
        df = pd.DataFrame(parsed_data)

        logger.info(f"Parsed {len(df)} stations")

        return df

    def process_weather_stations_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the weather stations DataFrame and return a processed DataFrame using Hamilton.
        @param df: The weather stations DataFrame.
        @return: A processed DataFrame with the weather stations.
        """
        dr = h_driver.Driver({}, wsp)
        final_columns = self.cfg.processing.weather_stations.final_columns

        # Pass raw columns directly as inputs
        result = dr.execute(
            final_columns,
            inputs={
                "raw_weather_station_id": df["weather_station_id"],
                "raw_height": df["height"],
                "raw_latitude": df["latitude"],
                "raw_longitude": df["longitude"],
                "raw_name": df["name"],
                "raw_state": df["state"],
                "raw_start_date": df["start_date"],
                "raw_end_date": df["end_date"],
                "raw_accessible": df["accessible"],
            },
        )
        out_df = pd.DataFrame(result)
        logger.info(f"Processed {len(out_df)} stations")
        return out_df

    def save_weather_stations_to_database(self, df: pd.DataFrame):
        """
        Save the weather stations DataFrame to the database.
        @param df: The weather stations DataFrame.
        """
        table = WeatherStations.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in df.to_dict(orient="records")
        ]

        stmt = pg_insert(table).values(records)
        update_columns = {
            c.name: stmt.excluded[c.name]
            for c in table.columns
            if c.name != "weather_station_id"
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["weather_station_id"], set_=update_columns
        )

        with Session(self.database_service.engine) as session:
            session.execute(stmt)
            session.commit()

        logger.info(f"Upserted {len(records)} weather stations to database")

    def load_from_database(self, only_relevant: bool = True) -> pd.DataFrame:
        """
        Load the weather stations from the database.
        @param only_relevant: If True, only load the relevant weather stations, (active and in Brandenburg)
        @return: The weather stations DataFrame.
        """
        with Session(self.database_service.engine) as session:
            table = WeatherStations.__table__
            query = select(table)
            if only_relevant:
                query = (
                    query.where(table.c.is_active == True)
                    .where(
                        or_(
                            table.c.state == "Brandenburg",
                            table.c.weather_station_id.in_(
                                self.cfg.dwd.additional_measurement_stations
                            ),
                        )
                    )
                    .where(
                        ~table.c.weather_station_id.in_(
                            self.cfg.dwd.exclude_brandenburg_measurement_stations
                        )
                    )
                )
            query = query.order_by(table.c.weather_station_id)

            rows = session.execute(query).mappings().all()
            df = pd.DataFrame(rows)
            logger.info(f"Loaded {len(df)} weather stations from database")
            return df
