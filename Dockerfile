# Sử dụng image Python nhẹ
FROM python:3.12-slim

# Đặt thư mục làm việc
WORKDIR /app/src

# Sao chép file yêu cầu vào container
COPY requirements.txt requirements.txt

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir fastapi transformers uvicorn

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng 8000
# EXPOSE 8080

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
