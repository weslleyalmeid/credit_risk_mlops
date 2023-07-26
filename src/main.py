import sys
import logging
from logging import basicConfig                     # configuracao do logging
from logging import DEBUG, INFO                     # levels
from logging import FileHandler, StreamHandler      # Mostrar log no terminal e pode salver em N arquivos

from data_processing.preprocess import run_data_prep
from models.train_model import run_register_model


basicConfig(
    level=INFO,
    format='%(levelname)s:%(asctime)s:%(message)s',
    handlers=[
        StreamHandler()
    ]
)

log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("Program started")
    run_data_prep()
    run_register_model()
    log.info("Program finished")