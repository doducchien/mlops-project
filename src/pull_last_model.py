import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import time
from prefect import flow, task
from dotenv import load_dotenv
# Logging configuration
LOG_FILE = "logs/pull_model.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
])
logger = logging.getLogger(__name__)

MODEL_PATH = "models/fine_tuned_gpt2"
LOAD_MODEL_INTERVAL = int(os.getenv("LOAD_MODEL_INTERVAL", 3600))  # Default to 3600 seconds if not specified

@task
def pull_latest_model():
    """
    Function to check and pull the latest model from DVC remote.
    """
    try:
        logger.info("Checking for updates to the model...")
        result = subprocess.run(["dvc", "status", f"{MODEL_PATH}.dvc"], capture_output=True, text=True, check=True)
        logger.debug(f"DVC status output: {result.stdout}")
        if "data is up to date" not in result.stdout:
            logger.info("New model version detected. Pulling latest model from DVC...")
            subprocess.run(["dvc", "pull", f"{MODEL_PATH}.dvc"], check=True)
            logger.info("Model pulled successfully.")
        else:
            logger.info("Model is already up to date. No action needed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking or pulling model: {e.stderr}")
        raise RuntimeError("Failed to check or pull the latest model from DVC.")
    


@flow
def mainLine():
    while True:
        pull_latest_model()
        time.sleep(LOAD_MODEL_INTERVAL)


if __name__ == "__main__":
   mainLine()
