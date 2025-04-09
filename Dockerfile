# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages like wordcloud
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Expose port (e.g., for Flask apps)
EXPOSE 5000

# Run the application (adjust as needed)
CMD ["python", "main.py"]
