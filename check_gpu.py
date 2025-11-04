import torch
import sys
import platform
import subprocess

def get_nvidia_driver_version():
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
    except:
        pass
    return "Not available"

def check_gpu():
    # Set console encoding to UTF-8 for Windows
    if platform.system() == "Windows":
        import io
        import sys
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # System Information
    print("="*70)
    print("🐍 WormGPT - System Information".center(70))
    print("="*70)
    print(f"💻 System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔧 CUDA Available: {'✅ Yes' if torch.cuda.is_available() else '❌ No'}")
    
    # GPU Information
    if torch.cuda.is_available():
        print("\n" + "🎮 GPU Information".center(70, '-'))
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Driver: {get_nvidia_driver_version()}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB")
            print(f"  - CUDA Capability: {'.'.join(str(x) for x in torch.cuda.get_device_capability(i))}")
    else:
        print("\n" + "⚠️ No GPU Detected".center(70, '-'))
        print("Running in CPU mode. Performance will be limited.")
        print("For better performance, please ensure you have:")
        print("1. NVIDIA GPU with CUDA support")
        print("2. Latest NVIDIA drivers installed")
        print("3. PyTorch with CUDA support")
    
    # Performance Notes
    print("\n" + "⚡ Performance Notes".center(70, '-'))
    if torch.cuda.is_available():
        print("✅ GPU acceleration is enabled")
        print("✅ Image generation will be faster")
    else:
        print("⚠️  Running in CPU mode (slow)")
        print("⚠️  Image generation will be very slow")
    
    print("="*70)
    
    # Return status code (0 for success with GPU, 1 for CPU-only)
    return 0 if torch.cuda.is_available() else 1

if __name__ == "__main__":
    try:
        sys.exit(check_gpu())
    except Exception as e:
        print(f"\n❌ Error checking GPU: {str(e)}")
        sys.exit(1)
