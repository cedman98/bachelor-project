from loguru import logger
from omegaconf import OmegaConf
from sqlmodel import SQLModel, create_engine


class DatabaseClass:
    def __init__(self, cfg: OmegaConf):
        self.engine = create_engine(cfg.database.url, echo=cfg.database.echo)

    def create_tables(self):
        SQLModel.metadata.create_all(self.engine)
        logger.info("Tables created")
