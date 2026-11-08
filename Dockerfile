FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_DIR=/models/sentiment_model

WORKDIR /app

# 1) torch (CPU فقط): طبقة ثقيلة، بتتخزّن
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2) باقي المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) الكود (بيتغيّر كتير، فآخر شي)
COPY src ./src

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "sentiment_api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]