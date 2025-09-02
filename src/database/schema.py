from datetime import date, datetime
from typing import Optional

from sqlalchemy import Column, String, UniqueConstraint
from sqlmodel import Field, SQLModel


class WeatherStations(SQLModel, table=True):
    __tablename__ = "weather_stations"

    weather_station_id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(255), nullable=False))
    latitude: float
    longitude: float
    height: float
    state: str
    start_date: date
    end_date: date
    is_active: bool


class WindStationMeasurements(SQLModel, table=True):
    __tablename__ = "wind_station_measurements"
    __table_args__ = (
        UniqueConstraint("station_id", "record_date", name="uix_station_date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    station_id: int = Field(index=True)
    quality_level: int
    average_wind_speed: float
    average_wind_direction: float
    record_date: datetime = Field(index=True)


class WindTurbines(SQLModel, table=True):
    __tablename__ = "wind_turbines"

    unit_mastr_number: str = Field(primary_key=True)
    last_update_date: datetime | None
    final_decommission_date: datetime | None
    gross_power: float | None
    net_nominal_power: float | None
    manufacturer: int | None
    technology: int | None
    type_designation: str | None
    hub_height: float
    rotor_diameter: float
    longitude: float = Field(index=True)
    latitude: float = Field(index=True)
