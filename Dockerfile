FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python run.py train

EXPOSE 8000
CMD ["sh", "-c", "python run.py serve --host 0.0.0.0 --port ${PORT:-8000}"]
