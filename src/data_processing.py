import json
import os
from datasets import load_dataset

RAW_DATA_PATH = "data/raw_data.json"
PROCESSED_DATA_PATH = "data/processed_data.json"

def fetch_dataset():
    print("Fetching dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump([sample["text"] for sample in dataset["train"]], f)
    print(f"Dataset saved to {RAW_DATA_PATH}")
    return RAW_DATA_PATH

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
