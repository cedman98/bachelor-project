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
        Index("ix_wsm_station_date_desc", "station_id", "record_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weather_stations.weather_station_id"), index=True
    )
    quality_level: Mapped[int] = mapped_column(Integer)
    average_wind_speed: Mapped[float] = mapped_column(Float)
    average_wind_direction: Mapped[float] = mapped_column(Float)
    air_pressure: Mapped[float] = mapped_column(Float)
    air_temperature_2m: Mapped[float] = mapped_column(Float)
    air_temperature_5cm: Mapped[float] = mapped_column(Float)
    relative_humidity: Mapped[float] = mapped_column(Float)
    dew_point_temperature: Mapped[float] = mapped_column(Float)
    precipitation_duration: Mapped[float] = mapped_column(Float)
    sum_precipitation_height: Mapped[float] = mapped_column(Float)
    precipitation_indicator: Mapped[int] = mapped_column(Integer)
    record_date: Mapped[datetime] = mapped_column(DateTime, index=True)


class WindStationMeasurementsPrediction(Base):
    __tablename__ = "wind_station_measurements_prediction"
    __table_args__ = (
        UniqueConstraint(
            "station_id", "record_date", name="uix_station_date_prediction"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weather_stations.weather_station_id"), index=True
    )
    u_pred: Mapped[float] = mapped_column(Float)
    v_pred: Mapped[float] = mapped_column(Float)
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


class TurbinePowerCurves(Base):
    __tablename__ = "turbine_power_curves"
    __table_args__ = (
        UniqueConstraint(
            "manufacturer_id", "turbine_id", name="uix_powercurve_manu_turbine"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    manufacturer_id: Mapped[int] = mapped_column(Integer, index=True)
    manufacturer_name: Mapped[str] = mapped_column(String(255))
    turbine_id: Mapped[int] = mapped_column(Integer, index=True)
    turbine_name: Mapped[str] = mapped_column(String(255))

    # Speed columns (kW at X m/s) mapped to numeric column names from the CSV
    ms_0: Mapped[float | None] = mapped_column("0", Float, nullable=True)
    ms_0_5: Mapped[float | None] = mapped_column("0_5", Float, nullable=True)
    ms_1: Mapped[float | None] = mapped_column("1", Float, nullable=True)
    ms_1_5: Mapped[float | None] = mapped_column("1_5", Float, nullable=True)
    ms_2: Mapped[float | None] = mapped_column("2", Float, nullable=True)
    ms_2_5: Mapped[float | None] = mapped_column("2_5", Float, nullable=True)
    ms_3: Mapped[float | None] = mapped_column("3", Float, nullable=True)
    ms_3_5: Mapped[float | None] = mapped_column("3_5", Float, nullable=True)
    ms_4: Mapped[float | None] = mapped_column("4", Float, nullable=True)
    ms_4_5: Mapped[float | None] = mapped_column("4_5", Float, nullable=True)
    ms_5: Mapped[float | None] = mapped_column("5", Float, nullable=True)
    ms_5_5: Mapped[float | None] = mapped_column("5_5", Float, nullable=True)
    ms_6: Mapped[float | None] = mapped_column("6", Float, nullable=True)
    ms_6_5: Mapped[float | None] = mapped_column("6_5", Float, nullable=True)
    ms_7: Mapped[float | None] = mapped_column("7", Float, nullable=True)
    ms_7_5: Mapped[float | None] = mapped_column("7_5", Float, nullable=True)
    ms_8: Mapped[float | None] = mapped_column("8", Float, nullable=True)
    ms_8_5: Mapped[float | None] = mapped_column("8_5", Float, nullable=True)
    ms_9: Mapped[float | None] = mapped_column("9", Float, nullable=True)
    ms_9_5: Mapped[float | None] = mapped_column("9_5", Float, nullable=True)
    ms_10: Mapped[float | None] = mapped_column("10", Float, nullable=True)
    ms_10_5: Mapped[float | None] = mapped_column("10_5", Float, nullable=True)
    ms_11: Mapped[float | None] = mapped_column("11", Float, nullable=True)
    ms_11_5: Mapped[float | None] = mapped_column("11_5", Float, nullable=True)
    ms_12: Mapped[float | None] = mapped_column("12", Float, nullable=True)
    ms_12_5: Mapped[float | None] = mapped_column("12_5", Float, nullable=True)
    ms_13: Mapped[float | None] = mapped_column("13", Float, nullable=True)
    ms_13_5: Mapped[float | None] = mapped_column("13_5", Float, nullable=True)
    ms_14: Mapped[float | None] = mapped_column("14", Float, nullable=True)
    ms_14_5: Mapped[float | None] = mapped_column("14_5", Float, nullable=True)
    ms_15: Mapped[float | None] = mapped_column("15", Float, nullable=True)
    ms_15_5: Mapped[float | None] = mapped_column("15_5", Float, nullable=True)
    ms_16: Mapped[float | None] = mapped_column("16", Float, nullable=True)
    ms_16_5: Mapped[float | None] = mapped_column("16_5", Float, nullable=True)
    ms_17: Mapped[float | None] = mapped_column("17", Float, nullable=True)
    ms_17_5: Mapped[float | None] = mapped_column("17_5", Float, nullable=True)
    ms_18: Mapped[float | None] = mapped_column("18", Float, nullable=True)
    ms_18_5: Mapped[float | None] = mapped_column("18_5", Float, nullable=True)
    ms_19: Mapped[float | None] = mapped_column("19", Float, nullable=True)
    ms_19_5: Mapped[float | None] = mapped_column("19_5", Float, nullable=True)
    ms_20: Mapped[float | None] = mapped_column("20", Float, nullable=True)
    ms_20_5: Mapped[float | None] = mapped_column("20_5", Float, nullable=True)
    ms_21: Mapped[float | None] = mapped_column("21", Float, nullable=True)
    ms_21_5: Mapped[float | None] = mapped_column("21_5", Float, nullable=True)
    ms_22: Mapped[float | None] = mapped_column("22", Float, nullable=True)
    ms_22_5: Mapped[float | None] = mapped_column("22_5", Float, nullable=True)
    ms_23: Mapped[float | None] = mapped_column("23", Float, nullable=True)
    ms_23_5: Mapped[float | None] = mapped_column("23_5", Float, nullable=True)
    ms_24: Mapped[float | None] = mapped_column("24", Float, nullable=True)
    ms_24_5: Mapped[float | None] = mapped_column("24_5", Float, nullable=True)
    ms_25: Mapped[float | None] = mapped_column("25", Float, nullable=True)
    ms_25_5: Mapped[float | None] = mapped_column("25_5", Float, nullable=True)
    ms_26: Mapped[float | None] = mapped_column("26", Float, nullable=True)
    ms_26_5: Mapped[float | None] = mapped_column("26_5", Float, nullable=True)
    ms_27: Mapped[float | None] = mapped_column("27", Float, nullable=True)
    ms_27_5: Mapped[float | None] = mapped_column("27_5", Float, nullable=True)
    ms_28: Mapped[float | None] = mapped_column("28", Float, nullable=True)
    ms_28_5: Mapped[float | None] = mapped_column("28_5", Float, nullable=True)
    ms_29: Mapped[float | None] = mapped_column("29", Float, nullable=True)
    ms_29_5: Mapped[float | None] = mapped_column("29_5", Float, nullable=True)
    ms_30: Mapped[float | None] = mapped_column("30", Float, nullable=True)
    ms_30_5: Mapped[float | None] = mapped_column("30_5", Float, nullable=True)
    ms_31: Mapped[float | None] = mapped_column("31", Float, nullable=True)
    ms_31_5: Mapped[float | None] = mapped_column("31_5", Float, nullable=True)
    ms_32: Mapped[float | None] = mapped_column("32", Float, nullable=True)
    ms_32_5: Mapped[float | None] = mapped_column("32_5", Float, nullable=True)
    ms_33: Mapped[float | None] = mapped_column("33", Float, nullable=True)
    ms_33_5: Mapped[float | None] = mapped_column("33_5", Float, nullable=True)
    ms_34: Mapped[float | None] = mapped_column("34", Float, nullable=True)
    ms_34_5: Mapped[float | None] = mapped_column("34_5", Float, nullable=True)
    ms_35: Mapped[float | None] = mapped_column("35", Float, nullable=True)

    conditions_nd_unknown: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
