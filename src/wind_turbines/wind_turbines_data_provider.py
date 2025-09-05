import io
import os
from pathlib import Path
from typing import Any, Dict
import zipfile
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.database.schema import WindTurbines
from src.database.database_service import DatabaseService
from hamilton import driver as h_driver
from src.wind_turbines import wind_turbines_preprocess as wtp
import xml.etree.ElementTree as ET


class WindTurbinesDataProvider:
    """
    The data provider provides the functionality for downloading and loading the wind turbines from the database.
    """

    cfg: OmegaConf
    database_service: DatabaseService

    def __init__(self, cfg: OmegaConf, database_service: DatabaseService):
        self.cfg = cfg
        self.database_service = database_service

    def download_wind_turbines(self, download_files: bool = False) -> pd.DataFrame:

        if download_files:
            self._download_file(
                self.cfg.marktstammdatenregister.wind_turbines_data_download_url,
                self.cfg.marktstammdatenregister.wind_turbines_data_file_name,
            )
            self._download_file(
                self.cfg.marktstammdatenregister.wind_turbines_explanation_download_url,
                self.cfg.marktstammdatenregister.wind_turbines_explanation_file_name,
            )

        data_xml_path = Path(self.cfg.marktstammdatenregister.wind_turbines_local_path)
        explanation_xsd_path = Path(
            self.cfg.marktstammdatenregister.wind_turbines_explanation_local_path
        )

        # Parse the XSD file
        field_definitions = self._parse_xsd_schema(explanation_xsd_path)
        logger.info(f"Parsed XSD file with {len(field_definitions)} fields")

        # Parse the XML file
        with open(data_xml_path, "rb") as f:
            root = ET.parse(f).getroot()

            wind_turbines = []
            for element in root.findall("EinheitWind"):
                wind_turbine_unit_data = self._extract_wind_unit_data(
                    element, field_definitions
                )
                wind_turbines.append(wind_turbine_unit_data)

            logger.info(f"Parsed XML file with {len(wind_turbines)} wind turbines")

            return pd.DataFrame(wind_turbines)

    def process_wind_turbines_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the wind turbines dataframe using Hamilton.
        @param df: The wind turbines dataframe.
        @return: The processed dataframe.
        """
        dr = h_driver.Driver({}, wtp)

        final_columns = self.cfg.processing.wind_turbines.final_columns

        outputs = final_columns + ["keep_mask"]
        result = dr.execute(
            outputs,
            inputs={
                # raw filters/flags
                "raw_Bundesland": df["Bundesland"],
                # raw fields
                "raw_EinheitMastrNummer": df["EinheitMastrNummer"],
                "raw_DatumLetzteAktualisierung": df["DatumLetzteAktualisierung"],
                "raw_Laengengrad": df["Laengengrad"],
                "raw_Breitengrad": df["Breitengrad"],
                "raw_DatumEndgueltigeStilllegung": df["DatumEndgueltigeStilllegung"],
                "raw_Bruttoleistung": df["Bruttoleistung"],
                "raw_Nettonennleistung": df["Nettonennleistung"],
                "raw_Hersteller": df["Hersteller"],
                "raw_Technologie": df["Technologie"],
                "raw_Typenbezeichnung": df["Typenbezeichnung"],
                "raw_Nabenhoehe": df["Nabenhoehe"],
                "raw_Rotordurchmesser": df["Rotordurchmesser"],
            },
        )

        out_df = pd.DataFrame(result)
        out_df = out_df[out_df["keep_mask"]].copy()
        out_df.drop(columns=["keep_mask"], inplace=True)

        logger.info(f"Processed {len(out_df)} wind turbines")
        return out_df.reset_index(drop=True)

    def save_wind_turbines_df_to_database(self, df: pd.DataFrame) -> None:
        """
        Save the wind turbines DataFrame to the database.
        @param df: The wind turbines DataFrame.
        """
        table = WindTurbines.__table__
        allowed_columns = set(table.columns.keys())
        records = [
            {k: v for k, v in row.items() if k in allowed_columns}
            for row in df.to_dict(orient="records")
        ]

        stmt = pg_insert(table).values(records)
        update_columns = {
            c.name: stmt.excluded[c.name]
            for c in table.columns
            if c.name != "unit_mastr_number"
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["unit_mastr_number"], set_=update_columns
        )

        with Session(self.database_service.engine) as session:
            session.execute(stmt)
            session.commit()

        logger.info(f"Upserted {len(records)} wind turbines to database")

    def load_from_database(self) -> pd.DataFrame:
        """
        Load the wind turbines from the database.
        """
        with Session(self.database_service.engine) as session:
            table = WindTurbines.__table__
            query = select(table)
            rows = session.execute(query).mappings().all()
            df = pd.DataFrame(rows)
            logger.info(f"Loaded {len(df)} wind turbines from database")
            return df

    def _download_file(self, download_url: str, file_name: str) -> str:
        """
        Download the file from the download url.
        @param download_url: The download url.
        @return: The dataframe.
        """
        logger.info(f"Start downloading file from {download_url}")
        response = requests.get(download_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Helper: recursively search for a file by basename inside zip and nested zips
        def _read_file_from_zip(current_zip: zipfile.ZipFile, target_basename: str):
            # Direct match first
            for name in current_zip.namelist():
                if os.path.basename(name) == target_basename:
                    with current_zip.open(name) as f:
                        return f.read()
            # Search nested zips
            for name in current_zip.namelist():
                if name.lower().endswith(".zip"):
                    with current_zip.open(name) as nested:
                        nested_bytes = nested.read()
                    try:
                        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
                            result = _read_file_from_zip(nested_zip, target_basename)
                            if result is not None:
                                return result
                    except zipfile.BadZipFile:
                        continue
            return None

        # Read the requested file from the zip (supports nested zips)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            target_name = os.path.basename(file_name)
            file_bytes = _read_file_from_zip(zip_file, target_name)
            if file_bytes is None:
                raise ValueError(
                    f"File {file_name} not found in the zip archive (including nested zips)"
                )

            # Create data directory if it doesn't exist
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)

            # Save the file to data directory using the basename
            file_path = os.path.join(data_dir, target_name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            return file_path

    def _parse_xsd_schema(self, xsd_file_path: Path) -> Dict[str, Any]:
        """Parse XSD schema to extract field definitions and their types"""

        # Parse the XSD file
        xsd_tree = ET.parse(xsd_file_path)
        xsd_root = xsd_tree.getroot()

        # XML Schema namespace
        ns = {"xs": "http://www.w3.org/2001/XMLSchema"}

        # Find the EinheitWind element definition
        einheit_wind_element = xsd_root.find('.//xs:element[@name="EinheitWind"]', ns)
        if einheit_wind_element is None:
            raise ValueError("Could not find EinheitWind element in XSD schema")

        # Find the complex type sequence that contains all fields
        sequence = einheit_wind_element.find(".//xs:sequence", ns)
        if sequence is None:
            raise ValueError("Could not find sequence in EinheitWind element")

        # Extract field definitions
        fields = {}

        # Map XSD types to Python types
        type_mapping = {
            "xs:string": str,
            "xs:int": int,
            "xs:short": int,
            "xs:byte": int,
            "xs:float": float,
            "xs:date": str,  # We'll keep dates as strings for now
            "xs:dateTime": str,  # We'll keep datetime as strings for now
        }

        # Process each element in the sequence
        for element in sequence.findall("xs:element", ns):
            field_name = element.get("name")
            field_type_attr = element.get("type")
            is_optional = element.get("minOccurs") == "0"

            # Determine the Python type
            if field_type_attr in type_mapping:
                python_type = type_mapping[field_type_attr]
            else:
                # Handle complex types with restrictions (enumerations)
                simple_type = element.find("xs:simpleType", ns)
                if simple_type is not None:
                    restriction = simple_type.find("xs:restriction", ns)
                    if restriction is not None:
                        base_type = restriction.get("base")
                        if base_type in type_mapping:
                            python_type = type_mapping[base_type]
                        else:
                            python_type = str  # Default fallback
                    else:
                        python_type = str  # Default fallback
                else:
                    python_type = str  # Default fallback

            fields[field_name] = {
                "type": python_type,
                "optional": is_optional,
                "xsd_type": field_type_attr,
            }

        return fields

    def _extract_wind_unit_data(
        self, element, field_definitions: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Extract data from a single EinheitWind XML element using XSD-derived field definitions"""
        data = {}

        # Extract data for each field defined in the XSD
        for field_name, field_info in field_definitions.items():
            field_element = element.find(field_name)
            field_type = field_info["type"]

            if field_element is not None and field_element.text:
                try:
                    if field_type == int:
                        data[field_name] = int(field_element.text)
                    elif field_type == float:
                        data[field_name] = float(field_element.text)
                    else:
                        data[field_name] = field_element.text
                except (ValueError, TypeError):
                    data[field_name] = None
            else:
                data[field_name] = None

        return data
