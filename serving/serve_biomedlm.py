"""BioMedLM FastAPI server — GPU 3, port 8004.

Exposes BOTH endpoints so the adapter can use /v1/completions (prompt key)
and health checks use /health.

FIX (2026-05-17): Added /v1/completions endpoint (prompt key, base-model style)
FIX (2026-05-17): /v1/chat/completions merges all messages (system+user)
FIX (2026-05-18): stop token list honoured in both endpoints
"""
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

HF_CACHE = os.environ.get('HF_HOME', '/lustre/smuexa01/client/users/mkotha/CS7325/hf_models')
MODEL_ID = 'stanford-crfm/BioMedLM'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_CTX  = 1536

print(f'Loading BioMedLM on {DEVICE} ...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=HF_CACHE,
    torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
)
model = model.to(DEVICE)
model.eval()
print('BioMedLM loaded.')

app = FastAPI()


class Message(BaseModel):
    role: str = 'user'
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.0
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    stop: Optional[List[str]] = None


def _generate(prompt_text: str, max_new: int, temperature: float,
              stop: Optional[List[str]]) -> str:
    inputs = tokenizer(
        prompt_text, return_tensors='pt',
        truncation=True, max_length=MAX_CTX,
    ).to(DEVICE)
    in_len  = inputs['input_ids'].shape[1]
    max_new = min(max_new, 512)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = max_new,
            pad_token_id   = tokenizer.eos_token_id,
            do_sample      = temperature > 0,
            temperature    = temperature if temperature > 0 else 1.0,
        )
    text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    # Honour stop tokens
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
    return text, in_len, len(out[0])


@app.post('/v1/chat/completions')
def chat(req: ChatRequest):
    # Merge all messages so system rubric instructions are not dropped
    parts = []
    for m in req.messages:
        role = m.role.upper()
        parts.append(f'[{role}]\n{m.content}')
    prompt_text = '\n\n'.join(parts)
    text, in_len, total = _generate(
        prompt_text, req.max_tokens, req.temperature, req.stop
    )
    return {
        'id': 'biomedlm', 'object': 'chat.completion', 'model': MODEL_ID,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': in_len,
                  'completion_tokens': total - in_len,
                  'total_tokens': total},
    }


@app.post('/v1/completions')
def completions(req: CompletionRequest):
    text, in_len, total = _generate(
        req.prompt, req.max_tokens, req.temperature, req.stop
    )
    return {
        'id': 'biomedlm', 'object': 'text_completion', 'model': MODEL_ID,
        'choices': [{
            'index': 0,
            'text': text,
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': in_len,
                  'completion_tokens': total - in_len,
                  'total_tokens': total},
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/v1/models')
def models():
    return {'data': [{'id': MODEL_ID}]}


if __name__ == '__main__':
    port = int(os.environ.get('BIOMEDLM_PORT', 8004))
    uvicorn.run(app, host='0.0.0.0', port=port)
