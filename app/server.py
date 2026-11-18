import json
import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================
# PATHS AND DEVICE
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(
    os.environ.get("VIJJU_PROJECT_DIR", str(APP_DIR.parent))
).resolve()

MODEL_PATH = PROJECT_DIR / "model" / "vijju_2.5_eco_v1_smart_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "vijju_tokenizer.json"
FRONTEND_PATH = PROJECT_DIR / "frontend" / "index.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TOKENIZER
# ============================================================

class VijjuTokenizer:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        self.special_tokens = data["special_tokens"]
        self.byte_offset = int(data["byte_offset"])
        self.vocab_size = int(data["vocab_size"])

        if "<BOS>" not in self.special_tokens or "<EOS>" not in self.special_tokens:
            raise ValueError("Tokenizer must define <BOS> and <EOS> special tokens.")

    def encode(self, text: str):
        token_ids = [int(self.special_tokens["<BOS>"])]
        token_ids.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        token_ids.append(int(self.special_tokens["<EOS>"]))
        return token_ids

    def decode(self, token_ids):
        output = bytearray()
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id >= self.byte_offset:
                byte_value = token_id - self.byte_offset
                if 0 <= byte_value <= 255:
                    output.append(byte_value)
        return output.decode("utf-8", errors="replace")


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, context: int):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"embedding_dim ({dim}) must be divisible by num_heads ({heads})."
            )

        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context, context, dtype=torch.bool)),
            persistent=True,
        )

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~self.mask[:seq_len, :seq_len], float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        return self.out(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = int(config["embedding_dim"])
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(
            dim,
            int(config["num_heads"]),
            int(config["context_length"]),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(config["ffn_dim"])),
            nn.GELU(),
            nn.Linear(int(config["ffn_dim"]), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VijjuEco(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size = int(config["vocab_size"])
        dim = int(config["embedding_dim"])
        context = int(config["context_length"])

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(context, dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(int(config["num_layers"]))]
        )

        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Keep the same tied-weight design used by this server architecture.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        _, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        return self.lm_head(x)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def normalize_checkpoint_keys(state_dict):
    """
    Translate only the two naming differences identified in the
    supplied checkpoint-loading traceback:
      blocks.N.attention.* -> blocks.N.attn.*
      ln_f.*               -> ln.*

    The loader still checks every key and tensor shape before loading.
    """
    normalized = {}

    for original_key, tensor in state_dict.items():
        key = original_key

        if key.startswith("module."):
            key = key[len("module."):]

        key = key.replace(".attention.", ".attn.")

        if key.startswith("ln_f."):
            key = "ln." + key[len("ln_f."):]

        if key in normalized:
            raise RuntimeError(
                f"Checkpoint key collision after name conversion: {key}"
            )

        normalized[key] = tensor

    return normalized


def load_vijju_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Check that the checkpoint is in the project's model/ folder."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain the required 'config' entry.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain the required 'model_state_dict' entry."
        )

    config = checkpoint["config"]
    required_config = [
        "vocab_size",
        "embedding_dim",
        "context_length",
        "num_heads",
        "num_layers",
        "ffn_dim",
    ]
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise KeyError(f"Missing model config fields: {missing_config}")

    model = VijjuEco(config)
    source_state = normalize_checkpoint_keys(checkpoint["model_state_dict"])
    target_state = model.state_dict()

    missing_keys = sorted(set(target_state) - set(source_state))
    unexpected_keys = sorted(set(source_state) - set(target_state))

    shape_errors = []
    for key in sorted(set(target_state).intersection(source_state)):
        if tuple(target_state[key].shape) != tuple(source_state[key].shape):
            shape_errors.append(
                f"{key}: checkpoint {tuple(source_state[key].shape)} "
                f"!= model {tuple(target_state[key].shape)}"
            )

    if missing_keys or unexpected_keys or shape_errors:
        details = ["Checkpoint does not exactly match this model architecture."]
        if missing_keys:
            details.append(f"Missing keys: {missing_keys[:20]}")
        if unexpected_keys:
            details.append(f"Unexpected keys: {unexpected_keys[:20]}")
        if shape_errors:
            details.append(f"Shape mismatches: {shape_errors[:20]}")
        raise RuntimeError("\n".join(details))

    model.load_state_dict(source_state, strict=True)
    model.to(DEVICE)
    model.eval()

    return model, config


# Load model and tokenizer once when the API process starts.
model, CONFIG = load_vijju_model()
tokenizer = VijjuTokenizer(TOKENIZER_PATH)

if tokenizer.vocab_size != int(CONFIG["vocab_size"]):
    raise ValueError(
        "Tokenizer/model vocabulary mismatch: "
        f"tokenizer={tokenizer.vocab_size}, model={CONFIG['vocab_size']}"
    )


# ============================================================
# TEXT GENERATION
# ============================================================

@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int = 180, temperature: float = 0.8):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    context_length = int(CONFIG["context_length"])
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-context_length:]

    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    generated_ids = []
    eos_id = int(tokenizer.special_tokens["<EOS>"])

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_length:]
        logits = model(x_cond)[:, -1, :]
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1)
        next_id = int(next_token.item())

        if next_id == eos_id:
            break

        generated_ids.append(next_id)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(generated_ids).strip()


# ============================================================
# OPTIONAL WEB SEARCH
# ============================================================

def needs_web(text: str) -> bool:
    keywords = [
        "latest", "today", "news", "current", "right now",
        "recent", "price", "weather", "what happened",
    ]
    lowered = text.lower()
    return any(word in lowered for word in keywords)


def web_search(query: str):
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote

        url = "https://html.duckduckgo.com/html/?q=" + quote(query)
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 Vijju-Eco/1.0"},
            timeout=10,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for item in soup.select(".result")[:5]:
            anchor = item.select_one(".result__a")
            snippet = item.select_one(".result__snippet")

            if anchor is None:
                continue

            results.append({
                "title": anchor.get_text(" ", strip=True),
                "url": anchor.get("href", ""),
                "snippet": (
                    snippet.get_text(" ", strip=True) if snippet else ""
                ),
            })

        return results

    except Exception as exc:
        print("WEB SEARCH ERROR:", str(exc))
        return []


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Vijju 2.5 Eco",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    response: str
    web_used: bool = False
    sources: List[str] = []


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model": "Vijju 2.5 Eco",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(DEVICE),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    web_used = False
    sources = []

    try:
        if needs_web(question):
            results = web_search(question)
        else:
            results = []

        if results:
            web_used = True
            web_context = "\n".join(
                f"{item['title']}: {item['snippet']}" for item in results
            )
            prompt = (
                f"User: {question}\n\n"
                f"Web information:\n{web_context}\n\n"
                "Answer using the information above. If it is insufficient, "
                "say so clearly.\nAssistant:"
            )
            sources = [item["url"] for item in results if item["url"]]
            answer = generate(prompt, max_new_tokens=180, temperature=0.7)
        else:
            prompt = f"User: {question}\nAssistant:"
            answer = generate(prompt, max_new_tokens=180, temperature=0.75)

        answer = answer.replace("<EOS>", "").strip()
        if not answer:
            answer = "I couldn't generate a response. Please try again."

        return ChatResponse(
            response=answer,
            web_used=web_used,
            sources=sources,
        )

    except Exception as exc:
        print("CHAT ERROR:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Vijju could not process this message. Check the backend logs.",
        )


# ============================================================
# SERVE FRONTEND FROM THE SAME SERVER
# ============================================================

@app.get("/")
def frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend file not found: {FRONTEND_PATH}",
        )
    return FileResponse(FRONTEND_PATH)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
