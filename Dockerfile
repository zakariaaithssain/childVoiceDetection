FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/child_detector_calibrated.joblib model/
COPY model.py app.py ./

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py"]
