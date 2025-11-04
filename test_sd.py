from diffusers import StableDiffusionXLPipeline
import torch

model_path = "/mnt/c/Users/one by one/Downloads/New folder/wormgpt_-the-obsidian-flame/model_files/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

prompt = "ultra realistic portrait of a cyberpunk warrior, dramatic lighting, 8k, cinematic"
image = pipe(prompt).images[0]
image.save("output.png")

print("✅ Image saved as output.png")
