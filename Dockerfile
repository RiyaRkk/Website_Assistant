# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional but helpful for builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install system packages + Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg wget ca-certificates unzip \
    fonts-liberation libasound2 libatk1.0-0 libatk-bridge2.0-0 libc6 \
    libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 \
    libgbm1 libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 \
    libpangocairo-1.0-0 libxcb1 libxcomposite1 libxcursor1 libxdamage1 \
    libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (helps with caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Load environment variables from .env at runtime (optional but common)
ENV PYTHONUNBUFFERED=1

# Expose the port your app runs on (Flask = 5000, FastAPI = 8000)
EXPOSE 5000

# Default command to run your app
CMD ["python", "app.py"]
