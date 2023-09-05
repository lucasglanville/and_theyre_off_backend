FROM python:3.10.6-buster

WORKDIR /interface

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY interface interface
# COPY setup.py setup.py
# RUN pip install .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD uvicorn interface.fast:app --host 0.0.0.0 --port $PORT
