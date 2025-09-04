CREATE TABLE IF NOT EXISTS wind_station_measurements (
    id SERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL,
    quality_level INTEGER NOT NULL,
    average_wind_speed DOUBLE PRECISION NOT NULL,
    average_wind_direction DOUBLE PRECISION NOT NULL,
    air_pressure DOUBLE PRECISION,
    air_temperature_2m DOUBLE PRECISION,
    air_temperature_5cm DOUBLE PRECISION,
    relative_humidity DOUBLE PRECISION,
    dew_point_temperature DOUBLE PRECISION,
    record_date TIMESTAMP NOT NULL,
    UNIQUE (station_id, record_date)
);

CREATE INDEX IF NOT EXISTS idx_wind_station_measurements_station_id
    ON wind_station_measurements (station_id);

CREATE INDEX IF NOT EXISTS idx_wind_station_measurements_record_date
    ON wind_station_measurements (record_date);


