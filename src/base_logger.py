import logging
import os

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))