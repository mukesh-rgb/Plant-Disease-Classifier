FROM python:3.12-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller, no CUDA needed on HF free tier)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HF Spaces listens on 7860
EXPOSE 7860

CMD ["python", "app.py"]
