from __future__ import annotations

from typing import Literal

import pandas as pd
from hamilton.function_modifiers import parameterize, source, value


@parameterize(
    **{
        "weather_station_id": {
            "raw_series": source("raw_weather_station_id"),
            "dtype": value("int"),
        },
        "height": {
            "raw_series": source("raw_height"),
            "dtype": value("int"),
        },
        "latitude": {
            "raw_series": source("raw_latitude"),
            "dtype": value("float"),
        },
        "longitude": {
            "raw_series": source("raw_longitude"),
            "dtype": value("float"),
        },
        "name": {
            "raw_series": source("raw_name"),
            "dtype": value("str"),
        },
        "state": {
            "raw_series": source("raw_state"),
            "dtype": value("str"),
        },
    }
)
def cast_column(
    raw_series: pd.Series,
    dtype: Literal["int", "float", "str"],
) -> pd.Series:
    """
    Generic casting for multiple columns based on raw input series.
    """
    if dtype == "int":
        return pd.to_numeric(raw_series, errors="coerce").astype(int)
    if dtype == "float":
        return pd.to_numeric(raw_series, errors="coerce").astype(float)
    return raw_series.astype(str)


def start_date(raw_start_date: pd.Series) -> pd.Series:
    return pd.to_datetime(raw_start_date, format="%Y%m%d", errors="coerce").dt.date


def end_date(raw_end_date: pd.Series) -> pd.Series:
    return pd.to_datetime(raw_end_date, format="%Y%m%d", errors="coerce").dt.date


def accessible(raw_accessible: pd.Series) -> pd.Series:
    return (raw_accessible.str.strip() == "Frei").astype(bool)


def is_active(end_date: pd.Series) -> pd.Series:
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=5)).date()
    return end_date >= cutoff
