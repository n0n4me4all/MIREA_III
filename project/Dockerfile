FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/

ENV CONFIG_PATH=configs/config.yaml
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
