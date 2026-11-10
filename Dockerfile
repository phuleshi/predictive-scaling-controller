FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predictive_controller.py .
COPY rps_prediction_model_v2.pkl .
COPY feature_config_v2.pkl .

CMD ["python", "predictive_controller.py"]