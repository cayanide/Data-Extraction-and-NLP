# Use official slim Python image (lightweight and stable)
FROM python:3.10-slim

# Set environment variables (good for Docker consistency)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required to build wheels and work with wordcloud
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (recommended)
RUN pip install --upgrade pip

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port for web apps (e.g., Flask or FastAPI)
EXPOSE 5000

# Default command to run your app
CMD ["python", "main.py"]
