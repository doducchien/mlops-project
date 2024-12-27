from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

MODEL_PATH = "models/fine_tuned_gpt2"

try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class InputText(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"message": "API is running successfully!"}

@app.post("/predict")
def predict(input_text: InputText):
    try:
        inputs = tokenizer.encode(input_text.text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")
