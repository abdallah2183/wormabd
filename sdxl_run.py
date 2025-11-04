from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

model_path = "./model_files/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a futuristic cyberpunk city, ultra realistic, 4k"
image = pipe(prompt).images[0]

image.save("output.png")
print("✅ Image saved: output.png")
