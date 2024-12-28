# Sử dụng Python base image
FROM python:3.12-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file phụ thuộc và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Expose cổng 8000 (Heroku yêu cầu)
EXPOSE 8000

# Lệnh khởi chạy API
CMD ["python", "src/app.py"]
