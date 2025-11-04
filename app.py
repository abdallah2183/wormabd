import os
import sys
import io
import time
import json
import atexit
import signal
import torch
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from dotenv import load_dotenv
import threading
import time
import secrets # Import secrets for image file naming
from huggingface_hub import hf_hub_download # <-- NEW IMPORT

# Force CPU usage and disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cuda.enabled = False
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

# Set device to CPU
DEVICE = 'cpu'

# Disable unnecessary logging
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('diffusers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# =======================================================
# 📌 إعدادات النماذج على Hugging Face Hub (يجب تعديلها!)
# =======================================================
# الرجاء تغيير هذا المتغير ليطابق اسم المستخدم واسم المستودع على Hugging Face
HF_REPO_ID = os.environ.get("HF_REPO_ID", "YourName/YourModelRepo") # <--- عدّل هذا!
# =======================================================

# Configuration
CONFIG = {
    'HOST': '127.0.0.1',
    'PORT': 5000,
    'API_KEYS': [os.getenv('AI_SERVER_API_KEY', 'My_Website_Secure_Key_123456')],
    # المسارات هنا ستكون مسارات مؤقتة يتم تحديثها بعد التحميل من Hugging Face
    'MODEL_PATHS': { 
        'text': 'model_files/dolphin-2.9-llama3-8b-q8_0.gguf',
        'image': 'model_files/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'
    },
    'MAX_TOKENS': 1024,
    'DEFAULT_IMAGE_SIZE': 512,
    'MAX_IMAGE_SIZE': 1024
}

# Initialize Flask app (Single initialization, use 'app' variable)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # For proper Arabic text support

# Load environment variables
YOUR_API_KEY = os.getenv('AI_SERVER_API_KEY', 'My_Website_Secure_Key_123456')
# Fix Windows console encoding (might not be strictly needed on Linux, but harmless)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize global variables
llama_model = None
sd_pipeline = None

# Ensure model directory exists (No longer needed since we use cache)
# os.makedirs('model_files', exist_ok=True) 

# Disable gradient calculation for inference
torch.set_grad_enabled(False)

# Force CPU usage
torch.set_default_tensor_type(torch.FloatTensor)
DEVICE = 'cpu'

# --- استيراد النواة والتبعيات ---
try:
    from llama_cpp import Llama
    # from diffusers import StableDiffusionPipeline # Already imported at the top
    # import torch # Already imported at the top
except ImportError as e:
    print(f"🚨 خطأ في استيراد المكتبات: {e}")
    pass 

# تحميل متغيرات البيئة (مثل مفتاحك السري)
load_dotenv(".env")

# --- متغيرات السرية والمسارات ---
# المسارات التالية لم تعد تستخدم للتحميل، بل تستخدم كمعرفات للملفات فقط
YOUR_API_KEY = os.environ.get("AI_SERVER_API_KEY")
TEXT_MODEL_FILENAME = os.environ.get("TEXT_MODEL", "dolphin-2.9-llama3-8b-q8_0.gguf")
IMAGE_MODEL_FILENAME = os.environ.get("IMAGE_MODEL", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors")

# تم حذف مسارات التحقق المحلية
# if not os.path.exists(D_MODEL_PATH): ...

# --- دالة المصادقة (الحارس) ---
def authenticate_request():
    """Verify API key from request headers"""
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key not in CONFIG['API_KEYS']:
        return False, jsonify({
            "status": "error",
            "message": "Invalid or missing API key"
        }), 401
    return True, None, None

def cleanup_models():
    """Clean up model resources properly"""
    global llama_model, sd_pipeline
    
    if llama_model is not None:
        try:
            # Simple dereference (llama_cpp should handle its own memory)
            llama_model = None
        except Exception as e:
            print(f"[WARNING] Error cleaning up text model: {e}")
    
    if sd_pipeline is not None:
        try:
            # Dereference the pipeline object
            sd_pipeline = None
            # Aggressively clear PyTorch cache if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARNING] Error cleaning up image model: {e}")


# --- تحميل النواتين عند بدء تشغيل الخادم ---
def load_ai_cores():
    """Load all AI models with proper error handling"""
    global llama_model, sd_pipeline
    
    models_loaded = True
    
    # Clean up existing models
    cleanup_models()
    
    print(f"\n{'='*50}")
    print(f"WORMGPT Server - Running on {DEVICE.upper()}")
    print(f"Hugging Face Repo ID: {HF_REPO_ID}")
    print(f"{'='*50}\n")
    
    # =============================================
    # ⬇️ التحميل من Hugging Face Hub (للنموذج النصي)
    # =============================================
    text_model_local_path = None
    try:
        print(f"[TEXT] Downloading {TEXT_MODEL_FILENAME} from Hugging Face...")
        text_model_local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=TEXT_MODEL_FILENAME,
            revision="main", # يمكن تغييرها إذا كان نموذجك على فرع آخر
            cache_dir="./hf_cache" # مسار للتخزين المؤقت
        )
        CONFIG['MODEL_PATHS']['text'] = text_model_local_path # تحديث المسار للقراءة من القرص
        print(f"[TEXT] ✅ Model downloaded successfully to: {text_model_local_path}")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to download text model from Hugging Face: {str(e)}")
        models_loaded = False
        
    # =============================================
    # ⬇️ التحميل من Hugging Face Hub (لنموذج الصور)
    # =============================================
    image_model_local_path = None
    try:
        print(f"\n[IMAGE] Downloading {IMAGE_MODEL_FILENAME} from Hugging Face...")
        image_model_local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=IMAGE_MODEL_FILENAME,
            revision="main",
            cache_dir="./hf_cache"
        )
        CONFIG['MODEL_PATHS']['image'] = image_model_local_path # تحديث المسار
        print(f"[IMAGE] ✅ Model downloaded successfully to: {image_model_local_path}")
    except Exception as e:
        print(f"[ERROR] ❌ Failed to download image model from Hugging Face: {str(e)}")
        # السماح للخادم بالاستمرار بدون نموذج الصور
    
    # Load text generation model
    if text_model_local_path:
        try:
            from llama_cpp import Llama
            print("\n[TEXT] Loading language model...")
            llama_model = Llama(
                model_path=CONFIG['MODEL_PATHS']['text'],
                n_ctx=4096,
                n_threads=os.cpu_count() // 2,
                verbose=False,
                n_gpu_layers=0 # Force CPU
            )
            print("[TEXT] ✅ Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load text model: {str(e)}")
            models_loaded = False
    else:
        print("[TEXT] ℹ️ Skipping text model loading due to prior download failure.")
        models_loaded = False
        
    # Load image generation model (optional)
    if image_model_local_path:
        try:
            print("\n[IMAGE] Loading image generation model...")
            
            # Configure model loading
            load_kwargs = {
                'torch_dtype': torch.float32,
                'safety_checker': None,
                'requires_safety_checker': False,
                # local_files_only=True remains, but now points to the downloaded cache path
                'local_files_only': True, 
                'use_safetensors': True,
                'variant': 'fp32',
            }
            
            # Load the model on CPU
            with torch.device('cpu'):
                # Force CPU even if CUDA is detected
                torch.cuda.is_available = lambda: False
                sd_pipeline = StableDiffusionXLPipeline.from_single_file(
                    CONFIG['MODEL_PATHS']['image'],
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                    use_safetensors=True,
                    variant='fp32'
                ).to('cpu')
                
                # Test the model with a simple prompt
                print("[IMAGE] Testing model with simple prompt...")
                with torch.no_grad():
                    # Reduce test size to prevent memory crash during startup
                    test_output = sd_pipeline(
                        prompt="test",
                        num_inference_steps=1,
                        width=64,
                        height=64,
                        output_type="pil",
                        generator=torch.Generator(device=DEVICE)
                    )
                
                print("[IMAGE] ✅ Model loaded and tested successfully!")
                
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load image model: {str(e)}")
            print("[IMAGE] ℹ️  Image generation will be disabled")
            import traceback
            traceback.print_exc()
            sd_pipeline = None
    else:
        print("[IMAGE] ℹ️ Skipping image model loading due to prior download failure.")
        sd_pipeline = None
    
    return models_loaded


# Execute model loading once when the module is imported by Gunicorn
# This replaces the logic that was inside if __name__ == '__main__':
print("[INFO] Starting AI server setup (Gunicorn launch)...")
try:
    models_loaded = load_ai_cores()
    if models_loaded:
        print("[SUCCESS] AI Core Initialization Complete.")
    else:
        print("[WARNING] Not all models loaded. Server will run with limited functionality.")
except Exception as e:
    print(f"[FATAL ERROR] Critical failure during initial model loading: {e}")
    sys.exit(1)


# --- Middlewares and Routes (Rest of the code remains the same) ---

# =======================================================
# 📝 المسار الأول: توليد النصوص (بدون قيود)
# =======================================================
@app.route('/ai/generate_text', methods=['POST'])
def generate_text_unrestricted():
    # 1. التحقق من مفتاحك السري
    is_authenticated, response, status = authenticate_request()
    if not is_authenticated:
        return response, status

    if llama_model is None:
        return jsonify({
            "status": "error",
            "message": "نواة النصوص غير متوفرة. يرجى التأكد من تثبيت النموذج النصي بشكل صحيح."
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                "status": "error",
                "message": "الرجاء إدخال نص للتحليل."
            }), 400
            
        user_prompt = data.get('prompt', 'Generate a detailed narrative.')
        max_tokens = min(int(data.get('max_tokens', 512)), 1024)  # حد أقصى 1024 رمز
        temperature = max(0.1, min(float(data.get('temperature', 0.7)), 1.0))  # نطاق 0.1 إلى 1.0

        try:
            output = llama_model.create_completion(
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,
                stop=["\n###", "\n---", "\nالمساعد:"]
            )
            
            if not output or 'choices' not in output or not output['choices']:
                return jsonify({
                    "status": "error",
                    "message": "لم يتم الحصول على استجابة صالحة من النموذج."
                }), 500
                
            ai_result = output['choices'][0]['text'].strip()
            
            # إزالة أي نصوص زائدة قد تظهر أحياناً
            stop_sequences = ["\n###", "\n---", "\nالمساعد:", "\nHuman:", "\n###"]
            for seq in stop_sequences:
                if seq in ai_result:
                    ai_result = ai_result.split(seq)[0].strip()
            
            return jsonify({
                "status": "success",
                "output": ai_result,
                "model": "Dolphin-2.9-Llama3-8B",
                "tokens_generated": len(ai_result.split())
            }), 200
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"حدث خطأ أثناء معالجة الطلب: {str(e)}",
                "error_type": str(type(e).__name__)
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"حدث خطأ غير متوقع: {str(e)}",
            "error_type": str(type(e).__name__)
        }), 500

# =======================================================
# 🖼️ المسار الثاني: توليد الصور (بدون قيود)
# =======================================================
@app.route('/ai/generate_image', methods=['POST'])
def generate_image_unrestricted():
    # 1. التحقق من مفتاحك السري
    is_authenticated, response, status = authenticate_request()
    if not is_authenticated: 
        return response, status

    if sd_pipeline is None:
        print("[ERROR] تم استدعاء نواة الصور ولكنها غير متوفرة")
        return jsonify({
            "status": "error",
            "message": "نواة الصور غير متوفرة حاليًا. يرجى التأكد من تثبيت النموذج بشكل صحيح.",
            "error_type": "ModelNotLoaded",
            "details": "The image generation model failed to load. Please check the server logs for more information."
        }), 503
        
    # 2. التحقق من أن النموذج جاهز للاستخدام
    try:
        print("[DEBUG] التحقق من حالة نموذج توليد الصور...")
        if sd_pipeline is None:
            raise RuntimeError("نموذج توليد الصور غير محمل")
            
        # التحقق من أن النموذج لديه السمات الأساسية
        required_attrs = ['device', 'scheduler', 'text_encoder', 'vae', 'unet']
        for attr in required_attrs:
            if not hasattr(sd_pipeline, attr):
                raise RuntimeError(f"نموذج توليد الصور غير مكتمل: مفقود {attr}")
                
        print(f"[DEBUG] نموذج توليد الصور جاهز على الجهاز: {getattr(sd_pipeline, 'device', 'unknown')}")
        
    except Exception as e:
        error_msg = f"فشل التحقق من حالة النموذج: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "حدث خطأ في تهيئة نموذج توليد الصور",
            "error_type": "ModelInitializationError",
            "details": str(e),
            "debug_info": {
                "pipeline_loaded": sd_pipeline is not None,
                "pipeline_attrs": dir(sd_pipeline) if sd_pipeline else []
            }
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                "status": "error",
                "message": "الرجاء إدخال وصف للصورة (prompt)."
            }), 400
            
        user_prompt = data.get('prompt', 'A detailed, photorealistic image.')
        num_inference_steps = min(int(data.get('num_inference_steps', 30)), 50)  # Limit to 50 steps max
        width = min(int(data.get('width', 512)), 1024)  # Max width 1024
        height = min(int(data.get('height', 512)), 1024)  # Max height 1024
        
        print(f"[INFO] Generating image with prompt: {user_prompt}")
        print(f"[INFO] Image dimensions: {width}x{height}, Steps: {num_inference_steps}")
        
        # توليد الصورة مع معالجة الأخطاء
        try:
            print("[INFO] Starting image generation...")
            # Safely get device information
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] Using device: {device}")
            
            # تعطيل xformers في وضع CPU لتفادي المشاكل
            try:
                if hasattr(sd_pipeline, 'enable_xformers_memory_efficient_attention'):
                    if torch.cuda.is_available():
                        sd_pipeline.enable_xformers_memory_efficient_attention()
                        print("[INFO] تم تفعيل الذاكرة الفعالة لـ xformers")
                    else:
                        print("[INFO] تم تعطيل xformers في وضع CPU")
            except Exception as e:
                print(f"[WARNING] تعذر تكوين xformers: {e}")
                
            # تحسينات إضافية لـ CPU
            if not torch.cuda.is_available():
                print("[INFO] تطبيق تحسينات وضع CPU...")
                try:
                    import torch
                    torch.set_num_threads(1)  # تقليل عدد الخيوط لتجنب استنزاف الموارد
                    if hasattr(sd_pipeline, 'enable_attention_slicing'):
                        sd_pipeline.enable_attention_slicing(slice_size='auto')
                        print("[INFO] تم تفعيل تقطيع الانتباه لتوفير الذاكرة")
                except Exception as e:
                    print(f"[WARNING] تعذر تطبيق تحسينات CPU: {e}")
            
            with torch.no_grad():
                try:
                    print("[INFO] Starting image generation...")
                    
                    # ضبط حجم الصورة ليكون مناسبًا للـ CPU
                    # We are in a constrained CPU environment, force small size for safety.
                    max_size = 512 if torch.cuda.is_available() else 384 
                    gen_width = min(width, max_size)
                    gen_height = min(height, max_size)
                    
                    # تأكد من أن الأبعاد من مضاعفات 8
                    gen_width = (gen_width // 8) * 8
                    gen_height = (gen_height // 8) * 8
                    
                    print(f"[INFO] إنشاء صورة بحجم {gen_width}x{gen_height} مع {num_inference_steps} خطوة")
                    
                    # إنشاء الصورة مع معالجة الأخطاء المحسنة
                    generation_kwargs = {
                        "prompt": user_prompt,
                        "negative_prompt": "blurry, low quality, distorted, bad anatomy, text, watermark, lowres, error",
                        "num_inference_steps": min(num_inference_steps, 30),  # تقليل الخطوات لتسريع العملية
                        "guidance_scale": 7.5,
                        "width": gen_width,
                        "height": gen_height,
                        "num_images_per_prompt": 1,
                        "output_type": "pil"
                    }
                    
                    # تنفيذ توليد الصورة مع كشف الأخطاء المحسن
                    try:
                        result = sd_pipeline(**generation_kwargs)
                    except Exception as gen_error:
                        print(f"[ERROR] فشل توليد الصورة: {str(gen_error)}")
                        raise RuntimeError(f"فشل توليد الصورة: {str(gen_error)}") from gen_error
                    
                    # Check if we got a valid result
                    if not result or not hasattr(result, 'images') or not result.images:
                        raise ValueError("No images were generated - invalid result format")
                    
                    image = result.images[0]
                    
                    # Verify the image
                    if not image:
                        raise ValueError("Generated image is empty")
                    
                    # Convert to RGB if needed
                    if hasattr(image, 'mode') and image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    print("[INFO] Image generated successfully")
                    
                except Exception as e:
                    error_msg = f"Image generation failed: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({
                        "status": "error",
                        "message": f"فشل توليد الصورة: {str(e)}",
                        "error_type": type(e).__name__,
                        "details": str(e)
                    }), 500
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Failed to generate image: {str(e)}")
            print(f"Error details: {error_details}")
            return jsonify({
                "status": "error",
                "message": f"حدث خطأ أثناء توليد الصورة: {str(e)}",
                "error_type": str(type(e).__name__),
                "details": str(e)
            }), 500
        
        # تحويل الصورة إلى base64
        import io
        import base64
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # حفظ الصورة للاحتفاظ بسجل (اختياري)
        try:
            if not os.path.exists("output_images"):
                os.makedirs("output_images")
            filename = f"image_{secrets.token_urlsafe(8)}.png"
            image_path = os.path.join("output_images", filename)
            image.save(image_path)
        except Exception as e:
            print(f"Warning: Could not save image: {e}")
        
        return jsonify({
            "status": "success",
            "output": img_str,
            "message": "تم توليد الصورة بنجاح"
        }), 200

    except Exception as e:
        return jsonify({
            "message": f"حدث خطأ غير متوقع: {str(e)}",
            "error_type": str(type(e).__name__)
        }), 500

# =======================================================
# 🚀 مسارات الخدمة
# =======================================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the server is running"""
    status = {
        "status": "running",
        "text_model_loaded": llama_model is not None,
        "image_model_loaded": sd_pipeline is not None,
        "system": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    return jsonify(status), 200

# Add a root route for basic testing
@app.route('/')
def home():
    return """
    <h1>AI Server is Running</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>GET /health - Check server status</li>
        <li>POST /ai/generate_text - Generate text</li>
        <li>POST /ai/generate_image - Generate images</li>
    </ul>
    """