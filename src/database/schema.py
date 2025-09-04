from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class WeatherStations(Base):
    __tablename__ = "weather_stations"

    weather_station_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)
    state: Mapped[str] = mapped_column(String)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean)


class WindStationMeasurements(Base):
    __tablename__ = "wind_station_measurements"
    __table_args__ = (
        UniqueConstraint("station_id", "record_date", name="uix_station_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weather_stations.weather_station_id"), index=True
    )
    quality_level: Mapped[int] = mapped_column(Integer)
    average_wind_speed: Mapped[float] = mapped_column(Float)
    average_wind_direction: Mapped[float] = mapped_column(Float)
    record_date: Mapped[datetime] = mapped_column(DateTime, index=True)


class WindTurbines(Base):
    __tablename__ = "wind_turbines"

    unit_mastr_number: Mapped[str] = mapped_column(String, primary_key=True)
    last_update_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    final_decommission_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    gross_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_nominal_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    manufacturer: Mapped[int | None] = mapped_column(Integer, nullable=True)
    technology: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type_designation: Mapped[str | None] = mapped_column(String, nullable=True)
    hub_height: Mapped[float] = mapped_column(Float)
    rotor_diameter: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float, index=True)
    latitude: Mapped[float] = mapped_column(Float, index=True)
