import io
import uuid
import queue
import threading

from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2  # If you want to handle encoded images in /predict
import requests

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Configuration: specify the "ideal" raw input format
# -----------------------------------------------------------------------------
IDEAL_HEIGHT = 640
IDEAL_WIDTH = 640
IDEAL_CHANNELS = 3
IDEAL_DTYPE = np.float32  # or np.float16, etc., depending on your pipeline
IDEAL_COLOR_SPACE = "BGR"  # set to "BGR" or "RGB" (or another if truly needed)
# Pixel range often [0..1], but some users keep [0..255]. Must match your training.


# -----------------------------------------------------------------------------
# Global objects
# -----------------------------------------------------------------------------
model = None            # YOLO model
prediction_queue = queue.Queue()
results_dict = {}
stop_thread = False


# -----------------------------------------------------------------------------
# Worker thread: processes tasks (task_id, image_array) -> runs model -> stores results
# -----------------------------------------------------------------------------
def worker_thread():
    while not stop_thread:
        try:
            task_id, img_array = prediction_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Run YOLO
        results = model.predict(img_array)
        detections = results[0]

        # Convert bounding boxes to JSON-friendly data
        boxes_json = []
        if detections.boxes is not None:
            boxes = detections.boxes
            names = detections.names

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                score = float(conf[i])
                class_id = int(clss[i])
                class_name = names.get(class_id, str(class_id))
                boxes_json.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": score,
                    "class_id": class_id,
                    "class_name": class_name
                })

        results_dict[task_id] = {"boxes": boxes_json}
        prediction_queue.task_done()

@app.route("/predict_url", methods=["GET"])
def predict_url():
    """
    GET /predict_url?url=<IMAGE_URL>
      1) Fetches the image from <IMAGE_URL>
      2) Decodes into a numpy array
      3) Enqueues it for YOLO inference
      4) Returns detection results as JSON
    """
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({"error": "No 'url' query parameter provided"}), 400

    # Fetch the image from the URL
    try:
        resp = requests.get(image_url, timeout=10)
        if resp.status_code != 200:
            return jsonify({
                "error": f"Failed to fetch image. HTTP status code {resp.status_code}"
            }), 400
        image_bytes = resp.content
    except Exception as e:
        return jsonify({"error": f"Could not retrieve image from URL. {str(e)}"}), 400

    # Decode the bytes into a NumPy array (assuming PNG/JPEG)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data from URL"}), 400

    # Enqueue for YOLO inference
    task_id = str(uuid.uuid4())
    prediction_queue.put((task_id, img_bgr))
    prediction_queue.join()  # Wait for worker thread

    # Retrieve result
    result = results_dict.pop(task_id, None)
    if not result:
        return jsonify({"error": "Unknown error occurred"}), 500

    return jsonify(result), 200

# -----------------------------------------------------------------------------
# 1) GET /info
#    Returns JSON describing exactly what raw format we expect
# -----------------------------------------------------------------------------
@app.route("/info", methods=["GET"])
def info():
    data = {
        "color_order": IDEAL_COLOR_SPACE,
        "height": IDEAL_HEIGHT,
        "width": IDEAL_WIDTH,
        "channels": IDEAL_CHANNELS,
        "dtype": str(IDEAL_DTYPE),
        "pixel_range": "[0..1]" if IDEAL_DTYPE == np.float32 else "[0..255] or other",
        "layout": "row-major (H x W x C)",
        "note": (
            f"Send exactly {IDEAL_HEIGHT} x {IDEAL_WIDTH} x {IDEAL_CHANNELS} "
            f"{IDEAL_COLOR_SPACE} float32 pixels in row-major order to /predict_raw. "
            "No decoding or conversion is done."
        )
    }
    return jsonify(data), 200


# -----------------------------------------------------------------------------
# 2) POST /predict_raw
#    Accepts raw bytes of shape (HEIGHT, WIDTH, CHANNELS) in the chosen color order, dtype, etc.
#    The server does NO color or format conversion.
# -----------------------------------------------------------------------------
@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    raw_bytes = request.data
    if not raw_bytes:
        return jsonify({"error": "No raw data found"}), 400

    # Convert raw bytes -> np array
    expected_size = IDEAL_HEIGHT * IDEAL_WIDTH * IDEAL_CHANNELS
    np_arr = np.frombuffer(raw_bytes, dtype=IDEAL_DTYPE)
    if np_arr.size != expected_size:
        return jsonify({
            "error": "Incorrect data size",
            "expected": expected_size,
            "received": np_arr.size
        }), 400

    # Reshape to (H,W,C). We assume row-major order
    np_arr = np_arr.reshape((IDEAL_HEIGHT, IDEAL_WIDTH, IDEAL_CHANNELS))

    # Enqueue
    task_id = str(uuid.uuid4())
    prediction_queue.put((task_id, np_arr))
    prediction_queue.join()

    # Retrieve
    result = results_dict.pop(task_id, None)
    if not result:
        return jsonify({"error": "Unknown error"}), 500

    return jsonify(result), 200


# -----------------------------------------------------------------------------
# 3) POST /predict
#    Standard approach: accept an encoded (JPEG, PNG, etc.) image, decode it (BGR by cv2).
#    Then we pass that to YOLO. This may do color ordering conversion internally
#    (i.e., YOLO might just treat it as BGR) or you can manually convert to RGB if you wish.
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    image_bytes = request.data
    if not image_bytes:
        return jsonify({"error": "No image data found in request body"}), 400

    # Decode with OpenCV -> BGR
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Enqueue
    task_id = str(uuid.uuid4())
    prediction_queue.put((task_id, img_bgr))
    prediction_queue.join()

    # Retrieve
    result = results_dict.pop(task_id, None)
    if not result:
        return jsonify({"error": "Unknown error"}), 500

    return jsonify(result), 200


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    global model

    # Load YOLO (your choice of weights)
    model = YOLO("yolo11n.pt")

    # Start worker
    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()

    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False)

    # Clean shutdown
    global stop_thread
    stop_thread = True
    t.join()


if __name__ == "__main__":
    main()

