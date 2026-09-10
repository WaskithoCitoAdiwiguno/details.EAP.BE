FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# HF Spaces runs containers as a non-root user on port 7860;
# make sure the app can write its artifacts and .env.local
RUN mkdir -p artifacts data && chmod -R 777 artifacts data

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
