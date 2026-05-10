# 1. Start with a lightweight Python base image
FROM python:3.10-slim

# 2. Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies required by OpenCV
# (These are C++ libraries that cv2 needs to process images)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements file and install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project into the container 
# (This includes api.py, your .pt YOLO weights, and your .onnx OCR model)
COPY . .

# 7. Expose port 8001 so your Laravel frontend can reach it
EXPOSE 8001

# 8. The command to start the FastAPI server when the container runs
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]