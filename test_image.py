import os
import sys
import json
import requests
import traceback
from dotenv import load_dotenv

# Enable detailed tracebacks
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:5000"
API_KEY = os.getenv("AI_SERVER_API_KEY")

# Configure requests to be more verbose
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
requests_log = logging.getLogger("urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

def test_simple_image():
    """Test image generation with a simple prompt"""
    url = f"{BASE_URL}/ai/generate_image"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "Accept": "application/json"
    }
    
    # Simple prompt in English to avoid any encoding issues
    data = {
        "prompt": "a simple red apple on a white background, high quality, photorealistic",
        "num_inference_steps": 10,  # Reduced steps for faster testing
        "width": 384,  # Reduced size for faster processing
        "height": 384,
        "guidance_scale": 7.5
    }
    
    # Log the request details
    logger.info(f"Sending request to: {url}")
    logger.info(f"Headers: {json.dumps(headers, indent=2)}")
    logger.info(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        logger.info("Sending request to generate image...")
        
        # First, verify the server is reachable
        try:
            health_check = requests.get(f"{BASE_URL}/health", timeout=5)
            logger.info(f"Health check status: {health_check.status_code}")
            if health_check.status_code != 200:
                logger.error(f"Server health check failed: {health_check.text}")
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
        
        # Send the actual request
        response = requests.post(
            url, 
            headers=headers, 
            json=data, 
            timeout=300,  # 5 minute timeout
            verify=False  # Disable SSL verification for local testing
        )
        
        # Log raw response
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        # Try to parse JSON response
        try:
            result = response.json()
            logger.info(f"Response JSON: {json.dumps(result, indent=2)}")
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response.text[:1000]}...")  # Log first 1000 chars
            return False
        
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        
        if 'output' in result and result['output']:
            logger.info("Image generated successfully!")
            try:
                # Save the image
                import base64
                from PIL import Image
                from io import BytesIO
                
                img_data = base64.b64decode(result['output'])
                img = Image.open(BytesIO(img_data))
                output_path = "generated_image.png"
                img.save(output_path)
                logger.info(f"Image saved as '{output_path}'")
                return True
            except Exception as img_error:
                logger.error(f"Failed to save image: {img_error}")
                return False
        else:
            logger.error("No image data in response")
            logger.error(f"Response: {json.dumps(result, indent=2)}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Status Code: {e.response.status_code}")
            try:
                error_response = e.response.json()
                logger.error(f"Error response: {json.dumps(error_response, indent=2)}")
            except ValueError:
                logger.error(f"Raw error response: {e.response.text[:1000]}...")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if not API_KEY:
        print("[ERROR] AI_SERVER_API_KEY not found in .env file")
        exit(1)
        
    print("[INFO] Testing image generation with a simple prompt...")
    success = test_simple_image()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed. Please check the logs above for details.")
