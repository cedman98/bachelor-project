from abc import ABC, abstractmethod
import pandas as pd


class ModelInterface(ABC):
    @abstractmethod
    def train(self, dataset: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
