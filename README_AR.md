# WormGPT - نموذج ذكاء اصطناعي متقدم

## المتطلبات
- Python 3.8 أو أحدث
- pip (مدير حزم بايثون)

## طريقة التثبيت

1. استنسخ المشروع:
```bash
git clone [رابط_المستودع]
cd wormgpt-the-obsidian-flame
```

2. قم بتثبيت المتطلبات:
```bash
chmod +x setup.sh
./setup.sh
```

3. أضف ملف `.env` في مجلد المشروع:
```
AI_SERVER_API_KEY=your_api_key_here
```

## طريقة التشغيل

لتشغيل الخادم محلياً:
```bash
python app.py
```

## النشر على Render

1. ارفع الكود إلى GitHub
2. سجل الدخول إلى [Render](https://render.com/)
3. اختر "New" ثم "Web Service"
4. اختر المستودع الخاص بك
5. استخدم الإعدادات التالية:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 120`
6. أضف متغيرات البيئة المطلوبة

## ملاحظات هامة
- تأكد من أن النماذج متاحة في مجلد `model_files`
- استخدم متغيرات البيئة للبيانات الحساسة
- النسخة المجانية من Render قد لا تكون كافية للنماذج الكبيرة
