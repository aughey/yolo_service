import cv2
import json
import argparse
import queue
import threading
from ultralytics import YOLO


def read_frames_thread(video_path, detection_queue, max_frames, stop_signal):
    """
    Reads frames from 'video_path'. For each frame:
      1) Split into left_img, right_img if width > height (forming two squares).
      2) Put (frame_id, left_img, right_img, original_width, original_height) into detection_queue.

    Sends a sentinel (None) when done.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        stop_signal[0] = True  # notify others
        detection_queue.put(None)
        return

    frame_id = 0
    while not stop_signal[0]:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        # If a max_frames limit is set, we can stop after that many
        if max_frames is not None and frame_id >= max_frames:
            break

        H, W, _ = frame.shape
        if W <= H:
            # If the width <= height, we won't do a left-right split
            # For simplicity, let's skip or treat the entire frame as "left" with no "right"
            left_img = frame
            right_img = None
        else:
            # Use H x H squares:
            #  left = columns [0 : H]
            #  right = columns [W - H : W]
            left_img = frame[:, 0:H, :]
            right_img = frame[:, W-H:W, :]

        # Put them in the queue
        # Blocks if queue is full (up to maxsize) to keep YOLO hot
        detection_queue.put((frame_id, left_img, right_img, W, H))

        frame_id += 1

    cap.release()
    # Signal the detection worker that we're done
    detection_queue.put(None)
    print("Reader thread: finished reading video")


def detection_worker_thread(model, detection_queue, aggregator_queue, stop_signal):
    """
    Pulls (frame_id, left_img, right_img, W, H) from detection_queue.
    Runs YOLO inference on each sub-image (if not None).
    Then pushes (frame_id, left_result, right_result, W, H) into aggregator_queue.
    Sends a sentinel (None) when done.
    """

    while not stop_signal[0]:
        item = detection_queue.get()
        if item is None:
            # No more frames to process
            detection_queue.task_done()
            break

        frame_id, left_img, right_img, W, H = item

        # Run YOLO on left sub-image
        left_result = None
        if left_img is not None:
            left_result = model_predict(model, left_img)

        # Run YOLO on right sub-image
        right_result = None
        if right_img is not None:
            right_result = model_predict(model, right_img)

        # Push detection data to aggregator
        aggregator_queue.put((frame_id, left_result, right_result, W, H))

        detection_queue.task_done()

    # Signal aggregator that we are finished
    aggregator_queue.put(None)
    print("Detection worker: finished all frames")


def aggregator_thread(aggregator_queue, output_json_path, stop_signal):
    """
    Pulls (frame_id, left_result, right_result, W, H) from aggregator_queue.
    - Adjusts the bounding boxes of the right_result by (W - H) in x-coordinates.
    - Collects results in a dictionary keyed by frame_id.
    When a sentinel (None) is received, it writes all results to 'output_json_path'.
    """

    # We'll store final results in a dict: frame_id -> list of box dicts
    all_results = {}

    while not stop_signal[0]:
        item = aggregator_queue.get()
        if item is None:
            aggregator_queue.task_done()
            break

        frame_id, left_result, right_result, W, H = item

        # Combine bounding boxes
        combined_boxes = []
        if left_result is not None:
            combined_boxes.extend(left_result)

        if right_result is not None:
            offset_x = W - H
            # shift x coords for right side
            for box in right_result:
                box["x1"] += offset_x
                box["x2"] += offset_x
            combined_boxes.extend(right_result)

        all_results[frame_id] = combined_boxes

        aggregator_queue.task_done()

    # Write all results to JSON
    # We'll store as a list of { "frame_id": X, "boxes": [...] } or do a dict
    final_list = []
    for fid in sorted(all_results.keys()):
        final_list.append({"frame_id": fid, "boxes": all_results[fid]})

    with open(output_json_path, "w") as f:
        json.dump(final_list, f, indent=2)

    print(f"Aggregator: wrote {len(final_list)} frames to {output_json_path}")


def model_predict(model, image):
    """
    Helper to run YOLO detection and return a list of boxes in JSON-friendly format.
    (x1, y1, x2, y2, confidence, class_id, class_name)
    """
    # Run inference
    results = model.predict(image)  # returns a list of length 1
    det = results[0]
    boxes_json = []
    if det.boxes is not None:
        boxes = det.boxes
        names = det.names

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            score = float(confs[i])
            cid = int(clss[i])
            cname = names.get(cid, str(cid))
            boxes_json.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": score,
                "class_id": cid,
                "class_name": cname
            })
    return boxes_json


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process a video with YOLO, splitting each frame into left & right squares."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--weights", type=str, required=True, help="Weights File")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional limit on number of frames to process")
    parser.add_argument("--queue_size", type=int, default=10,
                        help="Max queue size for detection pipeline to keep YOLO hot")
    args = parser.parse_args()

    # 1) Create queues
    #    'detection_queue' has limited size so we keep about 'queue_size' frames ready,
    #     ensuring YOLO can run at full speed if reading is faster than detection.
    detection_queue = queue.Queue(maxsize=args.queue_size)
    aggregator_queue = queue.Queue()

    # 2) A simple "stop" mechanism
    stop_signal = [False]  # store boolean in a list to mutate from threads

    # 3) Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.weights)  # or your custom weights
    print("Model loaded.")

    # 4) Spawn 3 threads:
    #    - Thread A: read video frames -> detection_queue
    #    - Thread B: detection worker -> aggregator_queue
    #    - Thread C: aggregator -> writes JSON
    t_reader = threading.Thread(
        target=read_frames_thread,
        args=(args.video, detection_queue, args.max_frames, stop_signal),
        daemon=True
    )
    t_detector = threading.Thread(
        target=detection_worker_thread,
        args=(model, detection_queue, aggregator_queue, stop_signal),
        daemon=True
    )
    t_aggregator = threading.Thread(
        target=aggregator_thread,
        args=(aggregator_queue, args.output, stop_signal),
        daemon=True
    )

    # 5) Start threads
    t_reader.start()
    t_detector.start()
    t_aggregator.start()

    # 6) Wait until reader finishes
    t_reader.join()

    # 7) Wait until detection_queue is empty (and sentinel consumed) => detection done
    detection_queue.join()

    # 8) Wait until aggregator_queue is empty (and sentinel consumed) => aggregator done
    aggregator_queue.join()

    # 9) Indicate we can stop threads (if they haven't already)
    stop_signal[0] = True
    t_detector.join()
    t_aggregator.join()

    print("All done!")


if __name__ == "__main__":
    main()
