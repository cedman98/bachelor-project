# Wind Energy Forecasting and Calculation System (Bachelor Thesis)

This project provides a comprehensive framework for calculating and predicting the electrical power output of wind turbines in Germany, with a focus on the Brandenburg region. It integrates weather data from the German Weather Service (DWD) with turbine technical specifications from the Marktstammdatenregister (MaStR).

## 🌟 Key Features

- **Automated Data Pipelines**: Fetches and processes 10-minute weather measurements from DWD and turbine metadata from MaStR using [Hamilton](https://github.com/DAGWorks-Inc/hamilton).
- **Spatial Interpolation**: Uses **Inverse Distance Weighting (IDW)** to estimate wind conditions at specific wind turbine coordinates based on nearby weather stations.
- **Vertical Extrapolation**: Extrapolates wind speed from measurement height (typically 10m) to turbine **hub height** (often 100m+) using the log wind profile/power law.
- **Power Production Calculation**: Maps hub-height wind speeds to electrical power output using turbine-specific **power curves**.
- **Advanced Forecasting**: Implements several time-series models for weather prediction:
  - **PatchTST** (Patch Time Series Transformer)
  - **LSTM / BiLSTM**
  - **LightGBM**
  - **Persistence Model** (Baseline)
- **REST API**: A Flask-based server providing endpoints for retrieving single and aggregated turbine calculation data.

## 🏗️ Architecture

The project is structured into modular services:

- `src/measurements`: Handling DWD weather station data.
- `src/wind_turbines`: Handling MaStR turbine specifications.
- `src/calculation`: Core logic for IDW interpolation, hub-height extrapolation, and power curve mapping.
- `src/model`: Machine learning models and training logic for time-series forecasting.
- `src/prediction`: Orchestrates the prediction flow using historical data and trained models.
- `server/`: Flask API and controllers for data access.

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Data Engineering**: Pandas, NumPy, Scipy, SF-Hamilton
- **Machine Learning**: PyTorch (Transformers/LSTM), Scikit-learn, LightGBM
- **Database**: PostgreSQL (SQLAlchemy / Psycopg3)
- **API**: Flask, Uvicorn
- **Config Management**: Hydra / OmegaConf

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- PostgreSQL database
- Environment variable `DATABASE_URL` (see `conf/config.yaml`)

### Installation

1. Clone the repository.
2. Install dependencies (recommended to use `uv` or `pip`):
   ```bash
   pip install .
   ```

### Running the API

```bash
python server/app.py
```

The server will be available at `http://localhost:5000`.

### Training a Model

```bash
python train.py
```

## 📊 Data Sources

- **DWD CDC**: [Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/)
- **MaStR**: [Marktstammdatenregister](https://www.marktstammdatenregister.de/)

## 📝 License

This project is part of a Bachelor Thesis. See the author for usage permissions.
