import os
import sys
import json
import requests
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:5000"
API_KEY = os.getenv("AI_SERVER_API_KEY")

def test_text_generation():
    """Test text generation endpoint"""
    url = f"{BASE_URL}/ai/generate_text"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    data = {
        "prompt": "اكتب لي قصيدة قصيرة عن الذكاء الاصطناعي",
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print("✅ Text Generation Test Passed!")
        print(f"Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Text Generation Test Failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return False

def test_image_generation():
    """Test image generation endpoint"""
    url = f"{BASE_URL}/ai/generate_image"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    data = {
        "prompt": "صورة لشروق الشمس في الصحراء",
        "num_inference_steps": 30,
        "width": 512,
        "height": 512
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'output' in result and result['output']:
            print("✅ Image Generation Test Passed!")
            print("Image generated successfully.")
            # Save the image
            import base64
            from PIL import Image
            from io import BytesIO
            
            img_data = base64.b64decode(result['output'])
            img = Image.open(BytesIO(img_data))
            img.save("test_output.png")
            print(f"Image saved as 'test_output.png'")
            return True
        else:
            print("❌ Image Generation Test Failed: No image data in response")
            print(f"Response: {json.dumps(result, indent=2)}")
            return False
            
    except Exception as e:
        print(f"❌ Image Generation Test Failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return False

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: AI_SERVER_API_KEY not found in .env file")
        exit(1)
        
    print("[INFO] Starting WormGPT API Tests...\n")
    
    # Test text generation
    print("1. Testing Text Generation...")
    text_success = test_text_generation()
    
    # Test image generation
    print("\n2. Testing Image Generation...")
    image_success = test_image_generation()
    
    # Print summary
    print("\n[TEST SUMMARY]")
    print(f"Text Generation: {'[PASSED]' if text_success else '[FAILED]'}")
    print(f"Image Generation: {'[PASSED]' if image_success else '[FAILED]'}")
    
    if text_success and image_success:
        print("\n[SUCCESS] All tests passed successfully!")
    else:
        print("\n[ERROR] Some tests failed. Please check the logs above for details.")
