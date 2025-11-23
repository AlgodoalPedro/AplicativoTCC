"""
Thread para processamento YOLO em vídeo/webcam
"""
import time
import cv2
import torch
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class WebcamThread(QThread):
    """Thread para processar detecção YOLO em tempo real (webcam/vídeo)"""
    frame_updated = pyqtSignal(QImage, list, float)

    def __init__(self, model_path, source=0, max_size=1280):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.source = source
        self.max_size = max_size  # Tamanho máximo para processar

    def run(self):
        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                print(f"Erro: Não foi possível abrir o vídeo: {self.source}")
                return

            fps_counter = 0
            start_time = time.time()
            fps = 0.0

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Fim do vídeo ou erro ao ler frame")
                    break

                try:
                    # Redimensionar frame grande para economizar memória
                    h, w = frame.shape[:2]
                    if max(h, w) > self.max_size:
                        scale = self.max_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                        print(f"Frame redimensionado de {w}x{h} para {new_w}x{new_h}")

                    results = model(
                        frame,
                        verbose=False,
                        conf=0.5,
                        device='0',
                        half=True  # Usar FP16 para economizar VRAM
                    )

                    annotated = results[0].plot()
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape

                    # Criar QImage com cópia dos dados contínuos
                    bytes_per_line = ch * w
                    qt_img = QImage(rgb.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)

                    detections = []
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        nome = results[0].names[cls]
                        detections.append((nome, conf))

                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        fps = fps_counter / (time.time() - start_time)

                    self.frame_updated.emit(qt_img, detections, fps)

                    # Limpar cache da GPU periodicamente
                    if fps_counter % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Erro ao processar frame {fps_counter}: {e}")
                    continue

            cap.release()

            # Limpar memória da GPU ao finalizar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Erro crítico na thread de vídeo: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Para a thread de forma segura"""
        self.running = False
        # Aguardar a thread terminar (com timeout de 3 segundos)
        if self.isRunning():
            self.wait(3000)
            if self.isRunning():
                print("Aviso: Thread não parou no tempo esperado")
                self.terminate()
                self.wait(1000)
