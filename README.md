# YOLO Flask Service

This repository (and Docker image) provides a Flask-based web service for performing object detection using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The service runs a single YOLO model “hot” in a background worker thread, handling multiple HTTP endpoints to accept image data in different formats.

---

## Docker Image

A prebuilt Docker image is available at:

```
ghcr.io/aughey/yolo_flask_service:latest
```

### Running the Container

To run the service as a Docker container:

```bash
docker run --rm -it -p 5000:5000 ghcr.io/aughey/yolo_flask_service:latest
```

This will start the Flask service on port `5000`. You can then make requests to the various endpoints (described below).

---

## Endpoints

### 1. `GET /info`

- **Purpose**: Returns metadata describing the raw format expected by the `/predict_raw` endpoint.  
- **Response**: JSON containing `color_order`, `height`, `width`, `channels`, `dtype`, etc.  

Example (assuming the container is running locally on port 5000):
```bash
curl http://localhost:5000/info
```

**Sample JSON Response**:
```json
{
  "color_order": "BGR",
  "height": 640,
  "width": 640,
  "channels": 3,
  "dtype": "float32",
  "pixel_range": "[0..1]",
  "layout": "row-major (H x W x C)",
  "note": "Send exactly 640 x 640 x 3 BGR float32 pixels..."
}
```

### 2. `POST /predict_raw`

- **Purpose**: Accepts **raw bytes** in the format described by `/info` (e.g., a shape of `640×640×3`, `float32`, BGR).  
- **Behavior**:  
  - No decoding or color conversions occur.  
  - The data is sent directly to the YOLO model.  
- **Request Body**: The raw bytes of a NumPy array in the exact shape and datatype returned by `/info`.  

#### Example in Python

```python
import requests
import numpy as np

# Suppose you have a BGR float32 array of shape (640,640,3) in [0..1]
fake_image = np.random.rand(640,640,3).astype("float32")

resp = requests.post(
    "http://localhost:5000/predict_raw",
    data=fake_image.tobytes()
)
print(resp.status_code, resp.json())
```

### 3. `POST /predict`

- **Purpose**: Accepts **encoded image** data (PNG/JPEG) in the **request body**.  
- **Behavior**:  
  - The service decodes the image with OpenCV (`cv2`) into BGR format.  
  - This is the standard approach if you have, for example, a PNG or JPEG file.  

#### Example with `curl`

```bash
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: image/png" \
     --data-binary "@/path/to/local_image.png"
```

**Sample Response** (JSON):
```json
{
  "boxes": [
    {
      "x1": 77.11,
      "y1": 52.22,
      "x2": 100.89,
      "y2": 120.01,
      "confidence": 0.891,
      "class_id": 0,
      "class_name": "person"
    },
    ...
  ]
}
```

### 4. `GET /predict_url`

- **Purpose**: Fetches the image from a remote URL, decodes it, and performs YOLO inference.  
- **Query Param**: `?url=<IMAGE_URL>`  
- **Usage**:
```bash
curl "http://localhost:5000/predict_url?url=https://ultralytics.com/images/bus.jpg"
```

---

## Project Overview

1. **YOLO Model**  
   The script loads a single YOLO model (`YOLO("yolo11n.pt")` by default) at startup. This model remains in memory (the “hot” model) and is shared by all requests.

2. **Worker Thread**  
   A dedicated worker thread (`worker_thread()`) continuously monitors a queue (`prediction_queue`) for incoming tasks. Each task has a `task_id` and a NumPy array representing the image.  
   - The worker runs `model.predict(...)` and stores results in a global `results_dict[task_id]`.  
   - The Flask endpoints enqueue tasks, then wait (`queue.join()`) until they’re processed.

3. **Endpoints**  
   - **`/info`**: Describes the raw format for `/predict_raw`.  
   - **`/predict_raw`**: Expects raw NumPy bytes (no decoding).  
   - **`/predict`**: Standard PNG/JPEG decode using OpenCV.  
   - **`/predict_url`**: Fetches an image from a remote URL for inference.  

4. **Color Space**  
   By default, `IDEAL_COLOR_SPACE` is “BGR” to align with typical OpenCV usage. If your model was trained on RGB images (without conversion), you can set `IDEAL_COLOR_SPACE = "RGB"` and adjust your inference pipeline accordingly.

5. **Supported Python Packages**  
   - `Flask` for the web server  
   - `ultralytics` for YOLO  
   - `numpy` and `opencv-python` (cv2) for image handling  
   - `requests` to fetch images by URL

---

## Example Usage

1. **Pull the Docker Image**:
   ```bash
   docker pull ghcr.io/aughey/yolo_flask_service:latest
   ```

2. **Run the Container**:
   ```bash
   docker run -p 5000:5000 ghcr.io/aughey/yolo_flask_service:latest
   ```
   The service starts on port `5000`.

3. **Check the Raw Input Info**:
   ```bash
   curl http://localhost:5000/info
   ```

4. **Predict from a Remote URL**:
   ```bash
   curl "http://localhost:5000/predict_url?url=https://ultralytics.com/images/bus.jpg"
   ```

5. **Predict from Raw Data** (Python snippet):
   ```python
   import requests
   import numpy as np

   # Create random test data
   data = np.random.rand(640, 640, 3).astype("float32")

   r = requests.post("http://localhost:5000/predict_raw", data=data.tobytes())
   print(r.status_code, r.json())
   ```

6. **Predict from a Local PNG/JPEG**:
   ```bash
   curl -X POST "http://localhost:5000/predict" \
        -H "Content-Type: image/png" \
        --data-binary "@/path/to/local_image.png"
   ```

---

## Limitations & Notes

- **Memory Constraints**: Large images or many concurrent requests can impact memory usage.  
- **Thread Safety**: Currently, the script uses **one** YOLO model instance in **one** worker thread. This handles requests sequentially. If you need parallel inference, you can launch multiple containers or manage multiple model instances.  
- **Timeouts**: The example uses `requests.get(image_url, timeout=10)` in `/predict_url`. Adjust as needed.  
- **Production Readiness**: This is a reference design. For production, consider rate limiting, request size limits, and more robust error handling.  

---

## License & Acknowledgments

- **Ultralytics YOLO** is licensed under the [GPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).  
- The Docker image is provided as-is, and you should ensure compliance with all respective licenses in your production environment.

---
