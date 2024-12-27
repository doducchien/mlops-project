from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tải model khi ứng dụng khởi động
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("models/fine_tuned_gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("models/fine_tuned_gpt2")
    app.state.model = model
    app.state.tokenizer = tokenizer
    print("Model loaded successfully.")
    
    yield  # Ứng dụng sẵn sàng hoạt động

    # Tắt ứng dụng (nếu cần cleanup)
    print("Shutting down application...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(input_text: str):
    # Sử dụng model đã tải trong lifespan
    model = app.state.model
    tokenizer = app.state.tokenizer
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
