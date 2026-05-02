import cv2

def blur_faces(frame, detections):
    for det in detections:
        if det["class"] == "person":
            x1, y1, x2, y2 = det["bbox"]
            face_y2 = y1 + max(1, int((y2 - y1) * 0.25))
            roi = frame[y1:face_y2, x1:x2]
            if roi.size > 0:
                frame[y1:face_y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 20)
    return frame