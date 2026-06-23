FROM python:3.12-slim

WORKDIR /app

COPY samples/frontend_backend/requirements_frontend.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY samples/frontend_backend/frontend.py frontend.py

ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "streamlit run frontend.py --server.port=$PORT --server.address=0.0.0.0"]
