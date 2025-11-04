import requests
import json

# تأكد من استبدال هذه القيم بقيمك الحقيقية
NGROK_URL = "http://127.0.0.1:5000" # استخدم 127.0.0.1 للاختبار على جهازك، أو رابط Ngrok
API_KEY = "your_chosen_secret_key_12345" # المفتاح الذي وضعته في .env.local

# مسار توليد النصوص
url = f"{NGROK_URL}/ai/generate_text" 

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY  # إرسال المفتاح السري
}

payload = {
    "prompt": "ما هي الخطوة الأخيرة لتشغيل سيرفر الذكاء الاصطناعي؟",
    "max_tokens": 100
}

try:
    print(f"إرسال طلب إلى: {url}")
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # التحقق من حالة الطلب
    if response.status_code == 200:
        print("\n✅ نجح الاتصال. النتيجة:")
        print(response.json().get('output', 'لم يتم العثور على مخرج'))
    else:
        print(f"\n❌ فشل الاتصال. رمز الحالة: {response.status_code}")
        print("الاستجابة:", response.text)

except Exception as e:
    print(f"\n❌ فشل الاتصال: {e}")