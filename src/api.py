from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

MODEL_PATH = "models/fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

@app.post("/predict")
def predict(input_text: str):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": response}
