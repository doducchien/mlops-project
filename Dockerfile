# Sử dụng image Python nhẹ
FROM python:3.12-slim

# Đặt thư mục làm việc
WORKDIR /app/src

# Sao chép file yêu cầu vào container
COPY requirements.txt requirements.txt

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir fastapi transformers uvicorn torch tensorflow dvc[gdrive]

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng 8000
# EXPOSE 8080

# Create ggdrive folder and credentials.json
ARG GDRIVE_CREDENTIALS_JSON
RUN mkdir -p ggdrive && \
    echo $GDRIVE_CREDENTIALS_JSON > ggdrive/credentials.json
# Lệnh khởi chạy ứng dụng
CMD [ "sh", "-c", "uvicorn src.app:app --host=0.0.0.0 --port=${PORT:-8000}" ]