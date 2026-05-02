import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
video = cv2.VideoCapture('videos/fsf_ep36.mp4')

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 30 != 0:
        continue
    
    results = model(frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.5:
                print(f"Frame {frame_count}: {cls} ({conf:.2f})")

video.release()
print("Terminé")