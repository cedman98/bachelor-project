import numpy as np
import pandas as pd

# Sampling configuration
SAMPLES_PER_HOUR = 6  # 10-minute data

# Helper utilities (not Hamilton nodes)


def _group_shift(series: pd.Series, group: pd.Series, hours: int) -> pd.Series:
    """
    Computes a grouped shift by station with the current index order preserved.
    The shift is specified in hours and scaled to the dataset sampling frequency
    defined by SAMPLES_PER_HOUR (10-minute → 6 samples per hour).
    Assumes upstream code sorted by [station_id, record_date] for correctness.
    """
    periods = hours * SAMPLES_PER_HOUR
    return series.groupby(group, sort=False).shift(periods)


def _ensure_range_index(series: pd.Series) -> pd.Series:
    """Ensures RangeIndex to keep Hamilton index checks happy."""
    return series.reset_index(drop=True)


def _group_rolling_mean(series: pd.Series, group: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(group, sort=False)
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _group_rolling_std(series: pd.Series, group: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(group, sort=False)
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


# Hamilton nodes


def wind_direction_radians(average_wind_direction: pd.Series) -> pd.Series:
    """
    Convert wind direction in degrees (meteorological: coming-from) to radians in [0, 2π).
    """
    return _ensure_range_index(np.deg2rad(average_wind_direction.mod(360)))


def u(average_wind_speed: pd.Series, wind_direction_radians: pd.Series) -> pd.Series:
    """
    Zonal wind component (east-west), meteorological convention.
    u = -speed * sin(dir)
    """
    return _ensure_range_index(-average_wind_speed * np.sin(wind_direction_radians))


def v(average_wind_speed: pd.Series, wind_direction_radians: pd.Series) -> pd.Series:
    """
    Meridional wind component (north-south), meteorological convention.
    v = -speed * cos(dir)
    """
    return _ensure_range_index(-average_wind_speed * np.cos(wind_direction_radians))


def hour_of_day(record_date: pd.Series) -> pd.Series:
    return _ensure_range_index(record_date.dt.hour.astype(int))


def hour_sin(hour_of_day: pd.Series) -> pd.Series:
    return _ensure_range_index(np.sin(2 * np.pi * (hour_of_day / 24.0)))


def hour_cos(hour_of_day: pd.Series) -> pd.Series:
    return _ensure_range_index(np.cos(2 * np.pi * (hour_of_day / 24.0)))


def day_of_year(record_date: pd.Series) -> pd.Series:
    return _ensure_range_index(record_date.dt.dayofyear.astype(int))


def doy_sin(day_of_year: pd.Series) -> pd.Series:
    return _ensure_range_index(np.sin(2 * np.pi * (day_of_year / 365.0)))


def doy_cos(day_of_year: pd.Series) -> pd.Series:
    return _ensure_range_index(np.cos(2 * np.pi * (day_of_year / 365.0)))


# Lags for u


def u_lag_1(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(u, station_id, 1))


def u_lag_3(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(u, station_id, 3))


def u_lag_6(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(u, station_id, 6))


def u_lag_12(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(u, station_id, 12))


def u_lag_24(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(u, station_id, 24))


# Lags for v


def v_lag_1(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(v, station_id, 1))


def v_lag_3(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(v, station_id, 3))


def v_lag_6(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(v, station_id, 6))


def v_lag_12(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(v, station_id, 12))


def v_lag_24(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_shift(v, station_id, 24))


# Rolling stats for u


def u_roll_mean_3h(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_mean(u, station_id, 3))


def u_roll_std_3h(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_std(u, station_id, 3))


def u_roll_mean_6h(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_mean(u, station_id, 6))


def u_roll_std_6h(u: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_std(u, station_id, 6))


# Rolling stats for v


def v_roll_mean_3h(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_mean(v, station_id, 3))


def v_roll_std_3h(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_std(v, station_id, 3))


def v_roll_mean_6h(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_mean(v, station_id, 6))


def v_roll_std_6h(v: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(_group_rolling_std(v, station_id, 6))


# Tendencies


def pressure_tendency_3h(air_pressure: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(air_pressure - _group_shift(air_pressure, station_id, 3))


def pressure_tendency_6h(air_pressure: pd.Series, station_id: pd.Series) -> pd.Series:
    return _ensure_range_index(air_pressure - _group_shift(air_pressure, station_id, 6))


def temperature_tendency_3h(
    air_temperature_2m: pd.Series, station_id: pd.Series
) -> pd.Series:
    return _ensure_range_index(
        air_temperature_2m - _group_shift(air_temperature_2m, station_id, 3)
    )


def temperature_tendency_6h(
    air_temperature_2m: pd.Series, station_id: pd.Series
) -> pd.Series:
    return _ensure_range_index(
        air_temperature_2m - _group_shift(air_temperature_2m, station_id, 6)
    )
