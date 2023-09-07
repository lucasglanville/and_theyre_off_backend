FROM tensorflow/tensorflow:2.10.0

# WORKDIR /api

COPY requirements-Docker.txt requirements.txt
COPY api/ api/


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt



HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD uvicorn api.fast2:app --host 0.0.0.0 --port $PORT
