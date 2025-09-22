from flask import Flask, jsonify, request, make_response
from controller.get_aggregated_calculation_data import get_aggregated_calculation_data
import os
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from controller.get_single_calcuation_data import get_single_calculation_data
from controller.get_schedule_update_data import get_schedule_update_data
from controller.get_unit_data import get_unit_data
from src.database.database_service import DatabaseService

# Initialize Hydra/OmegaConf configuration similar to notebooks
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "conf")

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg: DictConfig = compose(config_name="config")

app = Flask(__name__)
app.config["CFG"] = cfg

database_service = DatabaseService(app.config["CFG"])


@app.before_request
def handle_cors_preflight():
    if request.method == "OPTIONS":
        response = make_response("", 204)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers", "*"
        )
        return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.route("/calculations/single/<unit_mastr_number>")
def calculations_single(unit_mastr_number):
    return jsonify(
        get_single_calculation_data(
            app.config["CFG"], database_service, unit_mastr_number
        )
    )


@app.route("/unit/<unit_mastr_number>")
def unit(unit_mastr_number):
    return jsonify(
        get_unit_data(app.config["CFG"], database_service, unit_mastr_number)
    )


@app.route("/calculations/aggregated")
def calculations_aggregated():
    return jsonify(get_aggregated_calculation_data(app.config["CFG"], database_service))


@app.route("/schedule/update_data")
def schedule_update_data():
    return jsonify(get_schedule_update_data(app.config["CFG"], database_service))
