"""Generic CPU-friendly OpenAI-compatible HF model server.

Exposes the OpenAI HTTP shapes used by `core/wrapper.py`:
  * POST /v1/chat/completions
  * POST /v1/completions
  * GET  /health
  * GET  /v1/models

Designed for environments without a GPU. Loads any small HuggingFace
causal LM (e.g. Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct, TinyLlama)
and serves it as a single-process FastAPI app.

Used by the "real LLMs without GPU" experiment configuration when running
the clinical-QA judge panel on machines that cannot host the original
4B-7B medical models.

Env vars:
    MODEL_ID    HuggingFace repo id (required)
    PORT        HTTP port to listen on (required)
    HOST        bind host (default 0.0.0.0)
    HF_HOME     HF cache dir (default ~/.cache/huggingface)
    MAX_CTX     max prompt tokens fed into the model (default 1024)
    DTYPE       'float32' (default) or 'bfloat16'

Run:
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct PORT=8002 python serve_hf_model.py
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ['MODEL_ID']
PORT     = int(os.environ['PORT'])
HOST     = os.environ.get('HOST', '0.0.0.0')
HF_CACHE = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
MAX_CTX  = int(os.environ.get('MAX_CTX', '1024'))
DTYPE    = os.environ.get('DTYPE', 'float32')

dtype = torch.bfloat16 if DTYPE == 'bfloat16' else torch.float32
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))

print(f'[{MODEL_ID}] Loading on CPU (dtype={DTYPE}, ctx={MAX_CTX}) ...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=HF_CACHE, torch_dtype=dtype,
)
model.eval()
print(f'[{MODEL_ID}] Ready on port {PORT}.', flush=True)


app = FastAPI()


class Message(BaseModel):
    role: str = 'user'
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.0
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stop: Optional[List[str]] = None


def _format_messages_as_prompt(messages: List[Message]) -> str:
    """Apply the tokenizer's chat template, with fallbacks for templates that
    do not allow a system role (e.g. Mistral) and for tokenizers without any
    chat template at all (e.g. AdaptLLM)."""
    msgs = [{'role': m.role, 'content': m.content} for m in messages]

    has_template = bool(getattr(tokenizer, 'chat_template', None))
    if has_template:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            # Templates like Mistral's don't accept a 'system' role.
            # Merge system into the first user message and retry.
            merged: List[Dict[str, str]] = []
            system_buf: List[str] = []
            for m in msgs:
                if m['role'] == 'system':
                    system_buf.append(m['content'])
                    continue
                content = m['content']
                if m['role'] == 'user' and system_buf:
                    content = '\n\n'.join(system_buf) + '\n\n' + content
                    system_buf = []
                merged.append({'role': m['role'], 'content': content})
            if system_buf and not merged:
                merged.append({'role': 'user', 'content': '\n\n'.join(system_buf)})
            try:
                return tokenizer.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass

    # No chat template (or every attempt failed) -> plain role-prefixed join
    parts = []
    for m in msgs:
        parts.append(f'[{m["role"].upper()}]\n{m["content"]}')
    parts.append('[ASSISTANT]\n')
    return '\n\n'.join(parts)


def _generate(prompt_text: str, max_new: int, temperature: float,
              stop: Optional[List[str]]):
    inputs = tokenizer(
        prompt_text, return_tensors='pt',
        truncation=True, max_length=MAX_CTX,
    )
    in_len  = inputs['input_ids'].shape[1]
    max_new = max(8, min(max_new, 384))
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )
    text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    if stop:
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
    return text, in_len, int(out.shape[1])


@app.post('/v1/chat/completions')
def chat(req: ChatRequest):
    prompt_text = _format_messages_as_prompt(req.messages)
    text, in_len, total = _generate(
        prompt_text, req.max_tokens, req.temperature, req.stop,
    )
    return {
        'id': f'cmpl-{MODEL_ID}', 'object': 'chat.completion',
        'model': MODEL_ID,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop',
        }],
        'usage': {
            'prompt_tokens': in_len,
            'completion_tokens': total - in_len,
            'total_tokens': total,
        },
    }


@app.post('/v1/completions')
def completions(req: CompletionRequest):
    """/v1/completions — true raw-prompt mode.

    The client is expected to format the prompt itself (Alpaca / Llama-2-chat /
    base-model continuation). We do NOT apply any chat template here so that
    base models and tokenizer-template-less models can be driven precisely
    by their adapter prompts.

    Behaviour switch (env): if RAW_COMPLETION_MODE=0 we instead wrap the
    prompt as a single user chat turn (useful for instruction-tuned models
    whose adapter still wants the chat shape).
    """
    if os.environ.get('RAW_COMPLETION_MODE', '1') == '0':
        wrapped = _format_messages_as_prompt(
            [Message(role='user', content=req.prompt)]
        )
    else:
        wrapped = req.prompt
    text, in_len, total = _generate(
        wrapped, req.max_tokens, req.temperature, req.stop,
    )
    return {
        'id': f'cmpl-{MODEL_ID}', 'object': 'text_completion',
        'model': MODEL_ID,
        'choices': [{
            'index': 0,
            'text': text,
            'finish_reason': 'stop',
        }],
        'usage': {
            'prompt_tokens': in_len,
            'completion_tokens': total - in_len,
            'total_tokens': total,
        },
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_ID}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model'}]}


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, log_level='warning')
