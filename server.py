import io
import uuid
import queue
import threading
import sys

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
def worker_thread(model):
    while not stop_thread:
        try:
            event, task_id, img_array = prediction_queue.get(timeout=1.0)
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
        event.set()

def predict_bgr(img_bgr):
    # Enqueue
    task_id = str(uuid.uuid4())
    event = threading.Event()
    prediction_queue.put((event,task_id, img_bgr))
    event.wait()

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
    
    return predict_bgr(img_bgr)


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

    return predict_bgr(img_bgr)

@app.route("/predict_two_squares", methods=["GET"])
def predict_two_squares():
    """
    GET /predict_two_squares_queue?url=<IMAGE_URL>

    1) Fetch an image from <IMAGE_URL>
    2) Decode into BGR via OpenCV
    3) Assume the image is "wide" (width > height).
       Let H = image height, W = image width.
    4) Create two crops of shape (H x H):
       - left_img: columns [0 : H]
       - right_img: columns [W - H : W]
    5) Enqueue each crop for YOLO detection on the worker thread.
    6) Wait for both tasks to finish, then retrieve results.
    7) Offset the right boxes' x-coordinates by (W - H) so they map
       back to the original wide image.
    8) Combine boxes and return JSON.
    """
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({"error": "No 'url' query parameter provided"}), 400

    # 1) Fetch the remote image
    try:
        resp = requests.get(image_url, timeout=10)
        if resp.status_code != 200:
            return jsonify({
                "error": f"Failed to fetch image. HTTP status code {resp.status_code}"
            }), 400
        image_bytes = resp.content
    except Exception as e:
        return jsonify({"error": f"Could not retrieve image from URL. {str(e)}"}), 400

    # 2) Decode with OpenCV -> BGR array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid or unreadable image data"}), 400

    H, W, _ = img_bgr.shape
    if W <= H:
        return jsonify({"error": "Image width must be greater than height"}), 400

    # 3) Make two HxH crops
    left_img = img_bgr[:, 0:H, :]
    right_img = img_bgr[:, W - H:W, :]

    print(f"left_img shape: {left_img.shape}")
    print(f"right_img shape: {right_img.shape}")

    # 4) Enqueue both crops for YOLO inference
    left_task_id = str(uuid.uuid4())
    right_task_id = str(uuid.uuid4())
    left_event = threading.Event()
    right_event = threading.Event()
    prediction_queue.put((left_event,left_task_id, left_img))
    prediction_queue.put((right_event,right_task_id, right_img))

    # wait
    left_event.wait()
    right_event.wait()

    # 6) Retrieve results
    left_result = results_dict.pop(left_task_id, None)
    right_result = results_dict.pop(right_task_id, None)

    if left_result is None or right_result is None:
        return jsonify({"error": "Failed to retrieve inference results."}), 500

    left_boxes = left_result.get("boxes", [])
    right_boxes = right_result.get("boxes", [])

    # 7) Offset the right boxes' x-coordinates by (W - H)
    #    so they map back into the original coordinate space
    offset_x = W - H
    for box in right_boxes:
        box["x1"] += offset_x
        box["x2"] += offset_x

    # 8) Combine both sets of boxes
    combined_boxes = left_boxes + right_boxes

    return jsonify({"boxes": combined_boxes}), 200

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
# Main
# -----------------------------------------------------------------------------
def main():
    global model

    # Load YOLO (your choice of weights)
    # use argv to pass the path to the weights file
    # or "yolo11m.pt" if no argument is provided

    # start up 1 workers (it doesn't seem to make it better)
    for i in range(1):
        weights_file  = "yolo11m.pt" if len(sys.argv) < 2 else sys.argv[1]
        model = YOLO(weights_file)

        # Start worker
        t = threading.Thread(target=worker_thread, daemon=True, args=(model,))
        t.start()

    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False)

    # Clean shutdown
    global stop_thread
    stop_thread = True
    t.join()


if __name__ == "__main__":
    main()

