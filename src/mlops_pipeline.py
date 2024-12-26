from prefect import flow
from src.data_processing import fetch_dataset, preprocess_data
from src.train_model import fine_tune_model
from src.api import deploy_model
from src.util import track_with_dvc

@flow
def mlops_pipeline():
    # Bước 1: Tải dữ liệu
    raw_data_path = fetch_dataset()
    
    # Bước 2: Tiền xử lý dữ liệu
    processed_data_path = preprocess_data(raw_data_path)
    
    # Bước 3: Fine-tune mô hình
    model_path = fine_tune_model(processed_data_path)
    
    # Bước 4: Theo dõi dữ liệu/mô hình bằng DVC
    track_with_dvc(processed_data_path, model_path)
    
    # Bước 5: Triển khai API
    deploy_model()

if __name__ == "__main__":
    mlops_pipeline()
