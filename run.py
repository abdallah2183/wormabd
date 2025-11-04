from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

# تحميل النموذج من مجلد model_files
llm = Llama(
    model_path="model_files/dolphin-2.9-llama3-8b-q8_0.gguf",
    n_ctx=4096,      # حجم السياق (كبره حسب الرام)
    n_threads=8      # عدد الأنوية (غيرها حسب جهازك)
)

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    result = llm(
        req.prompt,
        max_tokens=500,
        temperature=0.7,
        stop=["</s>"]
    )
    
    response = result["choices"][0]["text"]
    return {"response": response}

# لتشغيله مباشرة: python run.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
