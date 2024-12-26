from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import json

MODEL_SAVE_PATH = os.path.join("models", "fine_tuned_gpt2")

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def fine_tune_model(processed_data_path):
    print("Fine-tuning GPT-2...")
    with open(processed_data_path, "r") as f:
        processed_data = json.load(f)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_data = tokenizer(processed_data, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    dataset = CustomDataset(tokenized_data)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        logging_dir="./logs",
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return MODEL_SAVE_PATH
