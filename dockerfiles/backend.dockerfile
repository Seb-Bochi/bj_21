FROM python:3.12-slim

WORKDIR /app

COPY samples/frontend_backend/backend.txt requirements.txt

RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY samples/frontend_backend/backend.py backend.py

ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port \"$PORT\""]
