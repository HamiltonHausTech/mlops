FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY api.py .
RUN bentoml build
CMD ["bentoml", "serve", "IrisService:latest", "--port", "5000"]
