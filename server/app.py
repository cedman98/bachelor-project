from flask import Flask, jsonify
from controller.get_aggregated_calculation_data import get_aggregated_calculation_data
import os
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.database.database_service import DatabaseService

# Initialize Hydra/OmegaConf configuration similar to notebooks
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "conf")

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg: DictConfig = compose(config_name="config")

app = Flask(__name__)
app.config["CFG"] = cfg

database_service = DatabaseService(app.config["CFG"])


@app.route("/calculations/aggregated")
def calculations_aggregated():
    return jsonify(get_aggregated_calculation_data(app.config["CFG"], database_service))
