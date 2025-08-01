# CCTV-Face-Detection
This project processes CCTV footage to detect and extract faces using the YOLOv8 object detection model. It identifies persons in video frames, crops the upper portion of each bounding box as the face region, and saves these face images at regular intervals (every 2 seconds) to reduce redundancy and applies perceptual hashing (phash) to identify and eliminate duplicate faces. Each saved face is checked against the last 100 saved face hashes to determine uniqueness. If it's not a duplicate, both the cropped face and full person image are saved. Duplicate detections are logged and stored separately for further analysis or debugging.

Features:
1. YOLOv8n model for efficient person detection
2. Extracts face regions from detected persons (top 25% of bounding box)
3. Perceptual hashing to filter out duplicate face images
4. Frame sampling (every 48 frames) for optimal performance on 24fps video
5. Outputs unique faces, full-body person crops, and duplicates

Requirements: 
  Python 3.8+
  Ultralytics YOLOv8
  OpenCV (cv2)
  Pillow (PIL)
  ImageHash

How it works:
  Loads YOLOv8 (yolov8n.pt) for person detection.
  Reads frames from cctv_footage.mp4.
  Detects people in each frame.
  Crops the top 25% of each person’s bounding box as a "face" image.
  Every 48 frames (~2 seconds), the cropped face is:
   1. Hashed using perceptual hashing, 
   2. Compared against recent saved hashes (last 100)
   3. Saved if unique (faces/ & persons/), or moved to duplicate/

▶ How to Run:
  Make sure you have a video named cctv_footage.mp4 in the same directory. Then run:
  python your_script_name.py
  Press q during video playback to quit early.

Parameters You Can Tune:
  frame_num % 48 == 0: Adjust frame skip interval for more/fewer face checks.
  hash_threshold = 15: Lower = stricter duplicate detection; higher = more lenient.
  saved_hashes = deque(maxlen=100): Number of previous hashes to remember.

Example Use Cases:
  Generating face/person datasets from CCTV
  Reducing duplicate face captures in real-time feeds

