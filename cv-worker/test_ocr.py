import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='ar')
video = cv2.VideoCapture('../videos/fsf_ep36.mp4')

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 90 != 0:
        continue

    h, w = frame.shape[:2]
    bandeau = frame[int(h*0.75):h, 0:w]
    
    result = ocr.predict(bandeau)
    
    if result:
        for item in result:
            if hasattr(item, 'rec_texts'):
                for text, conf in zip(item.rec_texts, item.rec_scores):
                    if conf > 0.7 and len(text.strip()) > 2:
                        print(f"Frame {frame_count}: {text} ({conf:.2f})")

video.release()
print("Terminé")

