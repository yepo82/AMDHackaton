FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json

CMD ["python", "main.py"]
