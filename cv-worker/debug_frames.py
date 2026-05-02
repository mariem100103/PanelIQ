import cv2

video = cv2.VideoCapture('C:/Users/user/Desktop/quizlive/videos/fsf_ep36.mp4')

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 900 != 0:
        continue

    h, w = frame.shape[:2]
    bandeau = frame
    
    cv2.imwrite(f'C:/Users/user/Desktop/quizlive/videos/frame_{frame_count}.jpg', bandeau)
    print(f"Frame {frame_count} sauvegardée")
    
    if frame_count > 27000:
        break

video.release()
print("Terminé")