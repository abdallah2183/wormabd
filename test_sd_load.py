import os
import sys
import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Configuration
MODEL_PATH = os.path.join(os.environ.get("MODEL_PATH", "model_files"), 
                         os.environ.get("IMAGE_MODEL", "Juggernaut-XL-v9-RunDiffusionPhoto_v2.safetensors"))

def test_sd_loading():
    print("="*50)
    print("Testing Stable Diffusion Model Loading")
    print("="*50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    print(f"[INFO] Model path: {MODEL_PATH}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please make sure the model file exists and the path is correct.")
        return False
    
    print("\n[INFO] Model file found. Attempting to load...")
    
    try:
        # Try to load the model
        print("\n[INFO] Creating Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=torch.float32,  # Use float32 for CPU, float16 for GPU
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        
        print("[INFO] Moving model to device...")
        pipe = pipe.to(device)
        
        # Test a small inference
        print("\n[INFO] Testing with a small prompt...")
        with torch.no_grad():
            print("[INFO] Generating test image (64x64, 1 step)...")
            image = pipe(
                prompt="a red apple",
                negative_prompt="blurry, low quality, distorted",
                width=64,
                height=64,
                num_inference_steps=1,
                guidance_scale=7.5,
                num_images_per_prompt=1
            ).images[0]
        
        print("\n[SUCCESS] Model loaded and tested successfully!")
        print(f"Generated image size: {image.size if hasattr(image, 'size') else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load or test the model: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Stable Diffusion Load Test")
    print("="*50)
    
    success = test_sd_loading()
    
    print("\n" + "="*50)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed. Please check the error messages above.")
    print("="*50)
