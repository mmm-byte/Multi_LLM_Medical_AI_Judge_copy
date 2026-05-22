import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BioGptTokenizer, BioGptForCausalLM
import uvicorn

app = FastAPI()
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to("cuda").half()

class ChatRequest(BaseModel):
    model: str = "microsoft/biogpt"
    messages: list
    max_tokens: int = 200

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = req.messages[-1]["content"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)