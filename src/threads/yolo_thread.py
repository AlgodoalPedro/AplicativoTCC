"""
Thread para processamento YOLO em imagens
"""
import os
import cv2
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal


class YOLOThread(QThread):
    """Thread para processar detecção YOLO em imagens estáticas"""
    finished = pyqtSignal(str, list)
    progress = pyqtSignal(int)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit(15)
            model = YOLO(self.model_path)
            self.progress.emit(45)
            results = model(
                self.image_path,
                verbose=False,
                conf=0.5,
                device='0',
                half=True  # Usar FP16 na GPU
            )
            self.progress.emit(75)

            save_dir = "resultados"
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, "saida.jpg")

            img_result = results[0].plot()
            cv2.imwrite(output_path, cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

            detections = []
            for box in results[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                nome = results[0].names[cls]
                detections.append((nome, conf))

            self.progress.emit(100)
            self.finished.emit(output_path, detections)
        except Exception as e:
            print("Erro:", e)
            self.finished.emit("", [])
