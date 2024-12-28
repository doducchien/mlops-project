from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import json
from prefect import task

# Định nghĩa đường dẫn lưu model
MODEL_SAVE_PATH =  "models/fine_tuned_gpt2"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

print(f"Model will be saved to: {MODEL_SAVE_PATH}")  # Debug thông tin đường dẫn

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
@task
def fine_tune_model(processed_data_path):
    print("Fine-tuning GPT-2 model...")
    with open(processed_data_path, "r") as f:
        processed_data = json.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(processed_data[:140], truncation=True, padding=True, max_length=10, return_tensors="pt")
    encodings["labels"] = encodings["input_ids"].clone()  # Thêm nhãn để mô hình tính loss
    dataset = CustomDataset(encodings)

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="logs",
        logging_steps=100,
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model fine-tuned and saved to {MODEL_SAVE_PATH}")