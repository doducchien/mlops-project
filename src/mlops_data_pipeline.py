# Import các thư viện cần thiết
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import re
import json
import os
from torch.utils.data import Dataset, DataLoader
from prefect import task, flow

# Đường dẫn lưu trữ dữ liệu
RAW_DATA_PATH = "data/raw_data.json"
PROCESSED_DATA_PATH = "data/processed_data.json"
MODEL_SAVE_PATH = "models/fine_tuned_gpt2"

# Custom Dataset class
class CustomTextDataset(Dataset):
    def __init__(self, tokenized_batches):
        self.data = tokenized_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

# Bước 1: Tải dữ liệu từ Hugging Face Datasets
@task
def fetch_dataset():
    print("Starting dataset fetch...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    print("Dataset loaded successfully.")
    train_data = dataset["train"]
    print(f"Fetched {len(train_data)} samples from the dataset.")
    # Lưu dữ liệu thô
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    with open(RAW_DATA_PATH, "w") as f:
        json.dump([sample["text"] for sample in train_data], f)
    print(f"Raw data saved to {RAW_DATA_PATH}.")
    print("Finished dataset fetch.")
    return RAW_DATA_PATH

# Bước 2: Tiền xử lý dữ liệu
@task
def preprocess_data(raw_data_path):
    print(f"Starting data preprocessing for {raw_data_path}...")
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw samples.")

    def clean_text(text):
        print(f"Cleaning text: {text[:50]}...")  # Log đoạn đầu của văn bản
        text = re.sub(r'\[.*?\]', '', text)  # Loại bỏ nội dung trong ngoặc vuông
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Loại bỏ ký tự đặc biệt
        text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng dư thừa
        return text.strip()

    print("Cleaning data...")
    processed_data = [clean_text(sample) for sample in raw_data]
    print(f"Processed {len(processed_data)} samples.")

    # Lưu dữ liệu đã xử lý
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    with open(PROCESSED_DATA_PATH, "w") as f:
        json.dump(processed_data, f)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}.")
    print("Finished data preprocessing.")
    return PROCESSED_DATA_PATH

# Bước 3: Fine-Tune GPT-2
@task
def fine_tune_gpt2(processed_data_path):
    print(f"Starting fine-tuning using processed data from {processed_data_path}...")
    with open(processed_data_path, "r") as f:
        processed_data = json.load(f)

    print(f"Loaded {len(processed_data)} processed samples.")

    # Tải tokenizer và model GPT-2
    print("Loading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Thiết lập token để padding
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("Tokenizer and model loaded successfully.")

    # Kiểm tra dữ liệu đầu vào cho tokenizer
    if not all(isinstance(sample, str) for sample in processed_data):
        raise ValueError("All samples must be strings for tokenization.")

    # Chia dữ liệu thành batch và thêm labels
    print("Tokenizing data with labels...")
    tokenized_batches = tokenizer(
        processed_data, 
        truncation=True, 
        padding="max_length", 
        max_length=128, 
        return_tensors="pt"
    )
    tokenized_batches["labels"] = tokenized_batches["input_ids"].clone()
    print("All data tokenized successfully.")

    # Chuẩn bị dataset
    dataset = CustomTextDataset(tokenized_batches)

    # Thiết lập tham số huấn luyện
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
    )
    print("Training arguments set up successfully.")

    # Huấn luyện model
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    print("Training completed.")

    # Lưu model đã fine-tune
    print("Saving fine-tuned model...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Fine-tuned model saved to {MODEL_SAVE_PATH}.")
    print("Finished fine-tuning.")

# Pipeline tổng thể
@flow
def data_ingestion_pipeline():
    print("Initializing pipeline...")
    raw_data = fetch_dataset()
    processed_data = preprocess_data(raw_data)
    fine_tune_gpt2(processed_data)
    print("Pipeline execution completed.")

# Bước 5: Quản lý phiên bản dữ liệu và mô hình với DVC
# Hướng dẫn (thực hiện trong terminal sau khi chạy pipeline):
# 1. Khởi tạo DVC: `dvc init`
# 2. Thêm file dữ liệu: `dvc add data/processed_data.json`
# 3. Thêm file model: `dvc add models/fine_tuned_gpt2`
# 4. Theo dõi bằng Git:
#    ```
#    git add data/processed_data.json.dvc models/fine_tuned_gpt2.dvc .gitignore
#    git commit -m "Add processed data and fine-tuned model"
#    ```
# 5. Đẩy dữ liệu và model lên remote storage:
#    ```
#    dvc remote add -d myremote s3://mybucket/path
#    dvc push
#    ```

if __name__ == "__main__":
    # Chạy pipeline Prefect
    print("Starting the pipeline...")
    data_ingestion_pipeline()
