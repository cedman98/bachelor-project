import os
import joblib
from src.model.variant.model_interface import ModelInterface
import pandas as pd
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
from lightgbm import LGBMModel
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LightGBMModel(ModelInterface):

    cfg: OmegaConf
    model: MultiOutputRegressor
    targets: List[str]

    def __init__(self, cfg: OmegaConf, targets: List[str]):
        self.cfg = cfg
        self.targets = targets
        self.models = {}

    def train(self, dataset: pd.DataFrame) -> None:
        train_df = self._process_dataset(dataset)

        x_train_df = train_df.drop(columns=self.targets)
        y_train_df = train_df[self.targets]

        base = lgb.LGBMRegressor(
            objective="regression",
            metric="mae",
            random_state=42,
            n_jobs=-1,
            verbosity=4,
            n_estimators=8000,
        )

        self.model = MultiOutputRegressor(base)
        self.model.fit(x_train_df, y_train_df, categorical_feature=["station_id"])

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        x_test_df = self._process_dataset(dataset).drop(columns=self.targets)
        return self.model.predict(x_test_df)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "multioutput_lgbm.joblib"))

    def load(self, path: str) -> None:
        self.model = joblib.load(os.path.join(path, "multioutput_lgbm.joblib"))

    def evaluate(self, test_dataset: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        test_df = self._process_dataset(test_dataset)
        # Prepare features and targets
        x_test = test_df.drop(columns=self.targets)
        y_test = test_df[self.targets]

        # Make predictions
        y_pred = self.model.predict(x_test)

        # Evaluate per target and overall
        mse_per_target = ((y_pred - y_test) ** 2).mean(axis=0)
        rmse_per_target = np.sqrt(mse_per_target)
        rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate additional metrics
        mae_per_target = np.abs(y_pred - y_test).mean(axis=0)
        mae_overall = mean_absolute_error(y_test, y_pred)

        # Build results dictionary
        results = {
            "rmse_overall": rmse_overall,
            "mae_overall": mae_overall,
        }

        # Add per-target metrics
        for i, target in enumerate(self.targets):
            results[f"{target}_rmse"] = rmse_per_target[i]
            results[f"{target}_mae"] = mae_per_target[i]

        return results

    def _process_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = dataset.copy()
        dataset.drop(columns=["record_date_timestamp"], inplace=True)

        return dataset
