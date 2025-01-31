# Sử dụng image Python nhẹ
FROM python:3.12-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép file yêu cầu vào container
COPY requirements.txt requirements.txt

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir fastapi transformers uvicorn torch tensorflow dvc[gdrive] prefect

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Copy the credentials.json file
# COPY ggdrive/credentials.json ggdrive/credentials.json
# COPY ggdrive/credentials.json ggdrive1/credentials.json

# Create ggdrive folder and credentials.json
# ARG GDRIVE_CREDENTIALS_JSON
# RUN mkdir -p ggdrive && \
#     echo "$GDRIVE_CREDENTIALS_JSON" > ggdrive/credentials.json
# Lệnh khởi chạy ứng dụng
# CMD [ "sh", "-c", "uvicorn src.app:app --host=0.0.0.0 --port=${PORT:-8000}" ]

# CMD ["sh", "-c", "python src/pull_last_model.py & uvicorn src.app:app --host=0.0.0.0 --port=${PORT:-8000}"]
CMD ["sh", "-c", "prefect server start --host 0.0.0.0 --port 4200 & python src/pull_last_model.py & uvicorn src.app:app --host=0.0.0.0 --port=${PORT:-8000}"]