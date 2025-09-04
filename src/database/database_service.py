from loguru import logger
from omegaconf import OmegaConf
from sqlalchemy import create_engine

from src.database.schema import Base  # ensure models are imported
import src.database.schema  # noqa: F401 to register models


class DatabaseService:
    def __init__(self, cfg: OmegaConf):
        self.engine = create_engine(
            cfg.database.url,
            echo=cfg.database.echo,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=10,
            max_overflow=20,
        )

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("Tables created")
