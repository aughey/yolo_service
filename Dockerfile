# Use the analytics/analytics image as the base
FROM ultralytics/ultralytics:latest

# Install Python dependencies (Flask, Ultralytics YOLO, optional opencv-python)
RUN pip install --no-cache-dir flask

# Create a working directory
WORKDIR /app

# Copy your Python script (the YOLO Flask service) into the container
COPY server.py .

# Expose the port Flask will run on (default 5000)
EXPOSE 5000

# Start the Flask service
ENTRYPOINT ["python", "server.py"]
