import cv2
from ultralytics import YOLO
import time

model = YOLO('best_model.pt')
cap = cv2.VideoCapture('Video7-Scene1.mp4')

fps_counter = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar para processar mais rápido
    # frame_resized = cv2.resize(frame, (640, 480))
    
    # Inferência
    results = model(
        frame,
        verbose=False,
        conf=0.5,
        device='cpu'
    )
    
    annotated_frame = results[0].plot()
    
    # Calcular FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps = fps_counter / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
    
    cv2.imshow('YOLOv8', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()