import os
import json
import subprocess
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from fastapi import FastAPI
from time import sleep
from prefect import flow, task
from train_model import  fine_tune_model
RAW_DATA_PATH = "data/raw_data.json"
PROCESSED_DATA_PATH = "data/processed_data.json"
MODEL_SAVE_PATH = "models/fine_tuned_gpt2"

# Data Processing
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

@task
def fetch_dataset():
    print("Fetching dataset...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump([sample["text"] for sample in dataset["train"]], f)
    print(f"Dataset saved to {RAW_DATA_PATH}")
    return RAW_DATA_PATH

@task
def preprocess_data(raw_data_path):
    print("Preprocessing data...")
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)
    processed_data = [" ".join(text.split()).strip() for text in raw_data if len(text) > 0]
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    with open(PROCESSED_DATA_PATH, "w") as f:
        json.dump(processed_data, f)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    return PROCESSED_DATA_PATH

# # Fine-tuning
# @task
# def fine_tune_model(processed_data_path):
#     print("Fine-tuning GPT-2 model...")
#     with open(processed_data_path, "r") as f:
#         processed_data = json.load(f)

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenized_data = tokenizer(processed_data, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
#     dataset = CustomDataset(tokenized_data)

#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         save_steps=500,
#         logging_dir="./logs",
#     )

#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#     )
#     trainer.train()

#     os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
#     model.save_pretrained(MODEL_SAVE_PATH)
#     tokenizer.save_pretrained(MODEL_SAVE_PATH)
#     print(f"Model saved to {MODEL_SAVE_PATH}")
#     return MODEL_SAVE_PATH

# DVC Tracking
@task
def track_with_dvc(data_path, model_path):
    print("Tracking and pushing changes with DVC...")
    subprocess.run(["dvc", "add", data_path], check=True)
    subprocess.run(["dvc", "add", model_path], check=True)
    subprocess.run(["git", "add", f"{data_path}.dvc", f"{model_path}.dvc", ".gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", "Track data and model with DVC"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)

# API Deployment
@task
def deploy_model(model_path):
    print("Deploying API...")
    app = FastAPI()
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    @app.post("/predict")
    def predict(input_text: str):
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": response}

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Monitoring
@task
def monitor_model():
    while True:
        print("Checking model health...")
        sleep(60)

# MLOps Pipeline
@flow(name="MLOps Pipeline")
def mlops_pipeline():
    # Step 1: Fetch dataset
    # raw_data_path = fetch_dataset()
    
    # Step 2: Preprocess data
    # processed_data_path = preprocess_data(raw_data_path)
    
    # Step 3: Fine-tune the model
    # model_path = fine_tune_model(processed_data_path)
    
    # Step 4: Track data and model with DVC
    # track_with_dvc(processed_data_path, model_path)
    
    # Step 5: Deploy the model as an API
    # deploy_model(model_path)
    
    # Step 6: Monitor the deployed model
    # monitor_model()
    print("Todo")

if __name__ == "__main__":
    mlops_pipeline()
