import logging
from source.utils.config_logging import config_logging

config_logging("INFO")

logger = logging.getLogger(__name__)
logger.info("Pipeline chosen: train_decoder")