from __future__ import annotations

from typing import Literal

import pandas as pd
from hamilton.function_modifiers import parameterize, source, value
import numpy as np


def normalized_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by stripping whitespace.
    """
    raw_df.columns = [c.strip() if isinstance(c, str) else c for c in raw_df.columns]
    return raw_df


def record_date(normalized_df: pd.DataFrame) -> pd.Series:
    """
    Parse `MESS_DATUM` into a pandas datetime.
    """
    return pd.to_datetime(
        normalized_df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce"
    )


def station_id_raw(normalized_df: pd.DataFrame) -> pd.Series:
    """
    Coerce `station_id` to numeric, leaving missing values as NaN for filtering later.
    """
    return pd.to_numeric(normalized_df["STATIONS_ID"], errors="coerce")


def valid_row_mask(station_id_raw: pd.Series, record_date: pd.Series) -> pd.Series:
    """
    Rows are valid if both station_id and record_date are present.
    """
    return station_id_raw.notna() & record_date.notna()


@parameterize(
    **{
        # wind
        "average_wind_speed": {
            "normalized_df": source("normalized_df"),
            "source_column": value("FF_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "average_wind_direction": {
            "normalized_df": source("normalized_df"),
            "source_column": value("DD_10"),
            "dtype": value("int"),
            "default_value": value(np.nan),
        },
        # air pressure/temperature/humidity
        "air_pressure": {
            "normalized_df": source("normalized_df"),
            "source_column": value("PP_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "air_temperature_2m": {
            "normalized_df": source("normalized_df"),
            "source_column": value("TT_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "air_temperature_5cm": {
            "normalized_df": source("normalized_df"),
            "source_column": value("TM5_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "relative_humidity": {
            "normalized_df": source("normalized_df"),
            "source_column": value("RF_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "dew_point_temperature": {
            "normalized_df": source("normalized_df"),
            "source_column": value("TD_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        # precipitation
        "precipitation_duration": {
            "normalized_df": source("normalized_df"),
            "source_column": value("RWS_DAU_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "sum_precipitation_height": {
            "normalized_df": source("normalized_df"),
            "source_column": value("RWS_10"),
            "dtype": value("float"),
            "default_value": value(np.nan),
        },
        "precipitation_indicator": {
            "normalized_df": source("normalized_df"),
            "source_column": value("RWS_IND_10"),
            "dtype": value("int"),
            "default_value": value(np.nan),
        },
        # quality
        "quality_level": {
            "normalized_df": source("normalized_df"),
            "source_column": value("QN"),
            "dtype": value("int"),
            "default_value": value(np.nan),
        },
    }
)
def numeric_cleaned_column(
    normalized_df: pd.DataFrame,
    source_column: str,
    dtype: Literal["int", "float"],
    default_value: int | float,
) -> pd.Series:
    """
    Generic numeric conversion for multiple columns. Fills NaNs with a default and casts to the desired dtype.
    """
    series = pd.to_numeric(normalized_df[source_column], errors="coerce").fillna(
        default_value
    )
    if dtype == "int":
        return series.astype(int)
    return series.astype(float)


def station_id(station_id_raw: pd.Series) -> pd.Series:
    """
    Return station_id as numeric; casting to integer happens after filtering valid rows.
    """
    return station_id_raw


def final_df(
    record_date: pd.Series,
    station_id: pd.Series,
    valid_row_mask: pd.Series,
    average_wind_speed: pd.Series,
    average_wind_direction: pd.Series,
    air_pressure: pd.Series,
    air_temperature_2m: pd.Series,
    air_temperature_5cm: pd.Series,
    relative_humidity: pd.Series,
    dew_point_temperature: pd.Series,
    precipitation_duration: pd.Series,
    sum_precipitation_height: pd.Series,
    precipitation_indicator: pd.Series,
    quality_level: pd.Series,
) -> pd.DataFrame:
    """
    Assemble the final DataFrame, filtering invalid rows and casting `station_id` to int.
    """
    mask = valid_row_mask
    data = {
        "station_id": station_id[mask].astype(int),
        "record_date": record_date[mask],
        "average_wind_speed": average_wind_speed[mask],
        "average_wind_direction": average_wind_direction[mask],
        "air_pressure": air_pressure[mask],
        "air_temperature_2m": air_temperature_2m[mask],
        "air_temperature_5cm": air_temperature_5cm[mask],
        "relative_humidity": relative_humidity[mask],
        "dew_point_temperature": dew_point_temperature[mask],
        "precipitation_duration": precipitation_duration[mask],
        "sum_precipitation_height": sum_precipitation_height[mask],
        "precipitation_indicator": precipitation_indicator[mask],
        "quality_level": quality_level[mask],
    }
    return pd.DataFrame(data).reset_index(drop=True)
