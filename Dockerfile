FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY instance/ ./instance

ENV FLASK_APP=app/main.py
EXPOSE 5000

CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2"]
