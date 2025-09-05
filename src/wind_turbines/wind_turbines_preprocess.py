from __future__ import annotations

from typing import Literal

import pandas as pd
from hamilton.function_modifiers import parameterize, source, value


@parameterize(
    **{
        # Simple casts from raw series
        "unit_mastr_number": {
            "raw_series": source("raw_EinheitMastrNummer"),
            "dtype": value("str"),
        },
        "longitude": {
            "raw_series": source("raw_Laengengrad"),
            "dtype": value("float"),
        },
        "latitude": {
            "raw_series": source("raw_Breitengrad"),
            "dtype": value("float"),
        },
        "gross_power": {
            "raw_series": source("raw_Bruttoleistung"),
            "dtype": value("float"),
        },
        "net_nominal_power": {
            "raw_series": source("raw_Nettonennleistung"),
            "dtype": value("float"),
        },
        "type_designation": {
            "raw_series": source("raw_Typenbezeichnung"),
            "dtype": value("str"),
        },
        "hub_height": {
            "raw_series": source("raw_Nabenhoehe"),
            "dtype": value("float"),
        },
        "rotor_diameter": {
            "raw_series": source("raw_Rotordurchmesser"),
            "dtype": value("float"),
        },
    }
)
def cast_column(
    raw_series: pd.Series,
    dtype: Literal["float", "str"],
) -> pd.Series:
    if dtype == "float":
        return pd.to_numeric(raw_series, errors="coerce").astype(float)
    return raw_series.astype(str)


@parameterize(
    **{
        "manufacturer": {
            "raw_series": source("raw_Hersteller"),
        },
        "technology": {
            "raw_series": source("raw_Technologie"),
        },
    }
)
def cast_nullable_int(raw_series: pd.Series) -> pd.Series:
    return pd.to_numeric(raw_series, errors="coerce").astype("Int64")


def last_update_date(raw_DatumLetzteAktualisierung: pd.Series) -> pd.Series:
    return pd.to_datetime(raw_DatumLetzteAktualisierung)


def final_decommission_date(raw_DatumEndgueltigeStilllegung: pd.Series) -> pd.Series:
    s = pd.to_datetime(
        raw_DatumEndgueltigeStilllegung, format="%Y-%m-%d", errors="coerce"
    )
    return s.dt.date.where(s.notna(), None)


def brandenburg(raw_Bundesland: pd.Series) -> pd.Series:
    return raw_Bundesland == 1400


def keep_mask(
    brandenburg: pd.Series,
    latitude: pd.Series,
    longitude: pd.Series,
    hub_height: pd.Series,
    rotor_diameter: pd.Series,
) -> pd.Series:
    return (
        brandenburg
        & latitude.notna()
        & longitude.notna()
        & hub_height.notna()
        & rotor_diameter.notna()
    )
