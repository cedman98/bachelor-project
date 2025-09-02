import io
import os
from pathlib import Path
from typing import Any, Dict
import zipfile
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import requests
from sqlmodel import Session, select
from src.database.schema import WindTurbines
from src.database.database_service import DatabaseService
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
        Process the wind turbines dataframe.
        @param df: The wind turbines dataframe.
        @return: The processed dataframe.
        """

        # Drop rows where bundesland is not 1400 (Brandenburg)
        df = df[df["Bundesland"] == 1400].copy()
        df.drop(columns=["Bundesland"], inplace=True)

        df = df[
            [
                "EinheitMastrNummer",
                "DatumLetzteAktualisierung",
                "Laengengrad",
                "Breitengrad",
                "DatumEndgueltigeStilllegung",
                "Bruttoleistung",
                "Nettonennleistung",
                "Hersteller",
                "Technologie",
                "Typenbezeichnung",
                "Nabenhoehe",
                "Rotordurchmesser",
            ]
        ]

        df.rename(
            columns={
                "EinheitMastrNummer": "unit_mastr_number",
                "DatumLetzteAktualisierung": "last_update_date",
                "Laengengrad": "longitude",
                "Breitengrad": "latitude",
                "DatumEndgueltigeStilllegung": "final_decommission_date",
                "Bruttoleistung": "gross_power",
                "Nettonennleistung": "net_nominal_power",
                "Hersteller": "manufacturer",
                "Technologie": "technology",
                "Typenbezeichnung": "type_designation",
                "Nabenhoehe": "hub_height",
                "Rotordurchmesser": "rotor_diameter",
            },
            inplace=True,
        )

        # Drop rows where latitude or longitude is None
        df = df[
            df["latitude"].notna()
            & df["longitude"].notna()
            & df["hub_height"].notna()
            & df["rotor_diameter"].notna()
        ]

        # Convert data types
        df["last_update_date"] = pd.to_datetime(df["last_update_date"])
        s = pd.to_datetime(
            df["final_decommission_date"], format="%Y-%m-%d", errors="coerce"
        )
        df["final_decommission_date"] = s.dt.date.where(s.notna(), None)
        df["longitude"] = df["longitude"].astype(float)
        df["latitude"] = df["latitude"].astype(float)
        df["gross_power"] = df["gross_power"].astype(float)
        df["net_nominal_power"] = df["net_nominal_power"].astype(float)
        df["hub_height"] = df["hub_height"].astype(float)
        df["rotor_diameter"] = df["rotor_diameter"].astype(float)
        df["manufacturer"] = pd.to_numeric(df["manufacturer"], errors="coerce").astype(
            "Int64"
        )
        df["technology"] = pd.to_numeric(df["technology"], errors="coerce").astype(
            "Int64"
        )
        df["type_designation"] = df["type_designation"].astype(str)

        logger.info(f"Processed {len(df)} wind turbines")

        return df

    def save_wind_turbines_df_to_database(self, df: pd.DataFrame) -> None:
        """
        Save the wind turbines DataFrame to the database.
        @param df: The wind turbines DataFrame.
        """
        wind_turbines = [WindTurbines(**row) for row in df.to_dict(orient="records")]

        with Session(self.database_service.engine) as session:
            for wind_turbine in wind_turbines:
                # check if measurement already exists
                remote_model = session.exec(
                    select(WindTurbines)
                    .where(
                        WindTurbines.unit_mastr_number == wind_turbine.unit_mastr_number
                    )
                    .where(
                        WindTurbines.unit_mastr_number == wind_turbine.unit_mastr_number
                    )
                ).first()

                # if measurement already exists, update it
                if remote_model:
                    upsert_model = remote_model
                else:
                    upsert_model = wind_turbine

                for key, value in wind_turbine.model_dump(exclude_unset=True).items():
                    setattr(upsert_model, key, value)

                session.add(upsert_model)

            session.commit()

        logger.info(f"Saved {len(wind_turbines)} wind turbines to database")

    def load_from_database(self) -> pd.DataFrame:
        """
        Load the wind turbines from the database.
        """
        with Session(self.database_service.engine) as session:
            query = select(WindTurbines)
            results = session.exec(query).all()
            df = pd.DataFrame([row.model_dump() for row in results])
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
