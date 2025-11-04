# 1. الصورة الأساسية: نستخدم إصدار Python 3.10 (نسخة slim أخف)
FROM python:3.10-slim

# **التصحيح الحاسم:** تثبيت أدوات التجميع (مثل GCC و CMake)
# هذه الأدوات ضرورية لتجميع مكتبات مثل llama-cpp-python من المصدر
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# 3. نسخ ملف المتطلبات أولاً
COPY requirements.txt .

# 4. تثبيت المتطلبات (مهم جداً للنماذج)
# نثبت PyTorch لـ CPU فقط (للتشغيل المجاني على الخادم) لتجنب أخطاء GPU
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# تثبيت باقي المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# 5. نسخ باقي ملفات المشروع (الكود والنماذج)
# سيتم نسخ app.py و model_files وكل شيء آخر (بدون النماذج الكبيرة التي حذفتها)
COPY . .

# 6. تعيين المنفذ (Port) الافتراضي
EXPOSE 5000

# 7. الأمر النهائي لتشغيل السيرفر باستخدام Gunicorn
# Gunicorn هو الخادم الإنتاجي الموصى به لـ Flask/Python
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]