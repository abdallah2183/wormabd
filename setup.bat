@echo off
REM إنشاء بيئة افتراضية
python -m venv venv

call venv\Scripts\activate

REM تثبيت المتطلبات
pip install --upgrade pip
pip install -r requirements.txt

REM إنشاء ملف .env.local إذا لم يكن موجوداً
if not exist .env.local (
    echo AI_SERVER_API_KEY=My_Website_Secure_Key_123456 > .env.local
    echo تم إنشاء ملف .env.local مع مفتاح افتراضي. يرجى تغييره لاحقاً.
) else (
    echo ملف .env.local موجود بالفعل.
)

echo.
echo ✅ تم إعداد المشروع بنجاح!
echo.
echo لتشغيل السيرفر:
echo 1. قم بتشغيل: python app.py
echo 2. في نافذة جديدة، قم بتشغيل: ngrok http 5000
echo 3. افتح ملف website/index.html في المتصفح
echo.
pause
