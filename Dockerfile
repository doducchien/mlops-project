FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
