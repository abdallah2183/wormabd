# server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, time, json, base64
from io import BytesIO
from PIL import Image

# LLM
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    print("llama_cpp not available:", e)

# Diffusers
try:
    import torch
    from diffusers import StableDiffusionXLPipeline
except Exception as e:
    torch = None
    StableDiffusionXLPipeline = None
    print("diffusers/torch not available:", e)

app = FastAPI()

# Allow frontend to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths - change if necessary
MODEL_GGUF_PATH = os.path.join(os.path.dirname(__file__), "model_files", "dolphin-2.9-llama3-8b-q8_0.gguf")
SDXL_PATH = os.path.join(os.path.dirname(__file__), "model_files", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors")

# Load LLM if available
llm = None
if Llama and os.path.exists(MODEL_GGUF_PATH):
    try:
        llm = Llama(model_path=MODEL_GGUF_PATH, n_ctx=4096, n_threads=6)
        print("Loaded LLM from", MODEL_GGUF_PATH)
    except Exception as e:
        print("Failed to load LLM:", e)
else:
    print("LLM model not loaded. Check model path and llama_cpp installation.")

# Load SD pipeline if available
pipe = None
if StableDiffusionXLPipeline and os.path.exists(SDXL_PATH):
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(SDXL_PATH)
        pipe = pipe.to(device)
        print("Loaded SDXL model on", device)
    except Exception as e:
        print("Failed to load SDXL:", e)
else:
    print("SDXL pipeline not loaded. Check model path and diffusers installation.")

class ChatRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

@app.get("/")
async def root():
    return {"status": "ok", "llm_loaded": bool(llm), "sd_loaded": bool(pipe)}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not llm:
        return {"error": "LLM not available on server."}
    try:
        out = llm(req.prompt, max_tokens=512, temperature=0.7)
        text = out["choices"][0]["text"] if "choices" in out else str(out)
        return {"response": text}
    except Exception as e:
        return {"error": f"LLM error: {e}"}

@app.post("/image")
async def image(req: ImageRequest):
    if not pipe:
        return {"error": "Image pipeline not available."}
    try:
        # generate image
        image = pipe(req.prompt, width=req.width, height=req.height).images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image_base64": img_b64}
    except Exception as e:
        return {"error": f"Image generation error: {e}"}
