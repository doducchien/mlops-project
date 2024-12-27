import os
import subprocess
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# FastAPI setup
app = FastAPI()

# Global variables to store model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    print("Starting up API...")
    pull_latest_model()  # Pull the latest model
    model, tokenizer = load_model()  # Load the model and tokenizer

@app.post("/predict")
def predict(input_text: str):
    global model, tokenizer
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
