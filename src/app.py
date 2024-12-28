import os
import subprocess
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel

# Define input schema
class InputText(BaseModel):
    input_text: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Cấu hình đường dẫn credentials cho Google Drive
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials/credentials.json"

# Path to model
MODEL_PATH = "models/fine_tuned_gpt2"
# Pull the latest model from DVC
def pull_latest_model():
    try:
        logger.info("Pulling the latest model from DVC...")
        subprocess.run(["dvc", "pull", f"{MODEL_PATH}.dvc"], check=True)
        logger.info("Model pulled successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling model: {e}")
        raise RuntimeError("Failed to pull the latest model from DVC.")

# Load model and tokenizer
def load_model():
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

# FastAPI setup with lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pull and load the model on startup
    logger.info("Starting up API...")
    pull_latest_model()
    model, tokenizer = load_model()
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield
    # Cleanup if needed
    logger.info("Shutting down API...")
app = FastAPI(lifespan=lifespan)

@app.post("/predict/")
async def predict(payload: InputText):
    """
    Generate text using the loaded model and tokenizer from app state
    """
    model = app.state.model
    tokenizer = app.state.tokenizer

    # Lấy giá trị input_text từ payload
    input_text = payload.input_text

    # Kiểm tra kiểu dữ liệu của input_text
    if not isinstance(input_text, str):
        raise ValueError("Input text must be a string.")

    # Tokenize và generate kết quả
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}


# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello World"}
