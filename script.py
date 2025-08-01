from ultralytics import YOLO
import cv2
import os
import shutil
from PIL import Image
import imagehash
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
model = YOLO('yolov8n.pt')

face_dir = 'faces'
person_dir = 'persons'
dup = 'duplicate'
cap = cv2.VideoCapture("cctv_footage.mp4")

for path in [face_dir, person_dir, dup]: # checking if the directory exists
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

face_id = 0
frame_num = 0
saved_hashes = deque(maxlen = 100)  # Keeps only the last 100 hashes
hash_threshold = 15  # 5 eliminated only 1 image,  10 eliminated 47 duplicates, so proceeded with 15 which deleted 96 duplicates
dup_id = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            # Class 0 is 'person' in dataset
            if cls_id == 0 and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_y2 = y1 + int((y2 - y1) * 0.25)
                face = frame[y1:face_y2, x1:x2]
                if face.size == 0:
                    continue
                if face.shape[0] < 10 or face.shape[1] < 10: # if face that are tiny crops
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, face_y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Face ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


                #  Since the video is 24fps. 48 will be a good frame interval (Capturing once for every 2 seconds). Which helps in removing more duplicates
                if frame_num % 48 == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, face_y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    #print(face_pil)
                    current_hash = imagehash.phash(face_pil)
                    #print(current_hash)

                    is_duplicate = False
                    for prev_hash in saved_hashes :
                        if abs(current_hash - prev_hash) <= hash_threshold:
                            is_duplicate = True
                            break

                    if is_duplicate :
                        dup_id += 1
                        person = frame[y1:y2, x1:x2]
                        person_path = os.path.join(dup, f"dup_person_{dup_id}.jpg") #saving the detected duplicates here for verification
                        cv2.imwrite(person_path, person)
                        logging.info(f"Saved {person_path}")
                    else :
                        face_path = os.path.join(face_dir, f"face_{face_id}.jpg") # saving the faces detected in video
                        cv2.imwrite(face_path, face)
                        logging.info(f"Saved {face_path}")
                        saved_hashes.append(current_hash)

                        person = frame[y1:y2, x1:x2]
                        person_path = os.path.join(person_dir, f"person_{face_id}.jpg") # saving the persons detected in the video
                        cv2.imwrite(person_path, person)
                        logging.info(f"Saved {person_path}")
                        face_id += 1
        frame_num += 1
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #video window gets off when we press q
            break
except Exception as e:
    logging.error(f"face detection failed due to this error: {e}")
finally :
    logging.info(f"Total number of frames : {frame_num}")
    logging.info(f"Total number of duplicates detected : {dup_id}")
    logging.info(f"Total number of faces saved : {face_id}")
    cap.release()
    cv2.destroyAllWindows() # closes the video window