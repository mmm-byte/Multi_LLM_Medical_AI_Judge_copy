import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BioGptTokenizer, BioGptForCausalLM
import uvicorn

HF_CACHE = os.environ.get("HF_HOME", "/lustre/smuexa01/client/users/mkotha/CS7325/hf_models")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading BioGPT on {DEVICE} ...")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", cache_dir=HF_CACHE)
model     = BioGptForCausalLM.from_pretrained("microsoft/biogpt", cache_dir=HF_CACHE)
model     = model.to(DEVICE)
if DEVICE == "cuda": model = model.half()
model.eval()
print("BioGPT loaded.")

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "microsoft/biogpt"
    messages: list
    max_tokens: int = 200
    temperature: float = 0.0

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = req.messages[-1]["content"]
    if prompt.strip().endswith("?") or any(prompt.lower().startswith(w) for w in ("what","how","why","describe","explain","rate","score","evaluate","assess","judge")):
        prompt = prompt.rstrip("?").strip() + " is characterized by"
    inputs   = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    in_len   = inputs["input_ids"].shape[1]
    max_new  = min(req.max_tokens, 400)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens  = max_new,
            pad_token_id    = tokenizer.eos_token_id,
            do_sample       = req.temperature > 0,
            temperature     = req.temperature if req.temperature > 0 else 1.0,
        )
    text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    return {
        "id": "biogpt", "object": "chat.completion", "model": "microsoft/biogpt",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": in_len, "completion_tokens": len(out[0])-in_len, "total_tokens": len(out[0])}
    }

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/v1/models")
def models(): return {"data": [{"id": "microsoft/biogpt"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)