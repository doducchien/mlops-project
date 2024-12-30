from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import json
import mlflow
import mlflow.pytorch
from transformers.integrations import MLflowCallback

# Định nghĩa đường dẫn lưu model
MODEL_SAVE_PATH = "models/fine_tuned_gpt2"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

print(f"Model will be saved to: {MODEL_SAVE_PATH}")

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def fine_tune_model(processed_data_path):
    print("Fine-tuning GPT-2 model...")

    # Tích hợp MLflow
    mlflow.set_experiment("mlops_project")  # Tên experiment trong MLflow
    with mlflow.start_run(run_name="Fine-tune GPT-2"):
        with open(processed_data_path, "r") as f:
            processed_data = json.load(f)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(processed_data[:10000], truncation=True, padding=True, max_length=10, return_tensors="pt")
        encodings["labels"] = encodings["input_ids"].clone()  # Thêm nhãn để mô hình tính loss
        dataset = CustomDataset(encodings)

        training_args = TrainingArguments(
            output_dir="results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_dir="logs",
            logging_steps=10,  # Log số liệu mỗi 10 bước
            report_to=["mlflow"],  # Tích hợp trực tiếp với MLflow
        )

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[MLflowCallback()]  # Tích hợp callback để log song song
        )

        # Log các tham số vào MLflow
        mlflow.log_param("epochs", 3)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("max_length", 10)

        # Huấn luyện mô hình
        trainer.train()

        # Lưu mô hình vào MLflow
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        mlflow.pytorch.log_model(model, "gpt2_model")

        print(f"Model fine-tuned and saved to {MODEL_SAVE_PATH}")
