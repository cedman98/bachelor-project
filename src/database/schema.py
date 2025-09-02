from datetime import date
from typing import Optional

from sqlalchemy import Column, String
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
