import os
import subprocess
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from contextlib import asynccontextmanager

# Cấu hình đường dẫn credentials cho Google Drive
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials/credentials.json"

# Path to model
MODEL_PATH = "models/fine_tuned_gpt2"

# Pull the latest model from DVC
def pull_latest_model():
    try:
        print("Pulling the latest model from DVC...")
        subprocess.run(["dvc", "pull", f"{MODEL_PATH}.dvc"], check=True)
        print("Model pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        raise RuntimeError("Failed to pull the latest model from DVC.")

# Load model and tokenizer
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# FastAPI setup with lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pull and load the model on startup
    print("Starting up API...")
    pull_latest_model()
    model, tokenizer = load_model()
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield
    # Cleanup if needed
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(input_text: str):
    # Use the loaded model and tokenizer from app state
    model = app.state.model
    tokenizer = app.state.tokenizer
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
