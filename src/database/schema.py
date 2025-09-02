from datetime import date
from typing import Optional

from sqlalchemy import Column, String
from sqlmodel import Field, SQLModel


class WindStation(SQLModel, table=True):
    __tablename__ = "wind_stations"

    station_id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(255), nullable=False))
    latitude: float
    longitude: float
    height: float
    start_date: date
