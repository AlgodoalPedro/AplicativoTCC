# ================================================
# Auto-verificação de dependências do requirements.txt
# ================================================
import os
import sys
import subprocess

def instalar_dependencias():
    """
    Verifica e instala automaticamente as dependências do requirements.txt
    caso alguma esteja faltando.
    """
    try:
        import pkg_resources
        requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

        if os.path.exists(requirements_path):
            with open(requirements_path, "r") as f:
                dependencies = f.read().splitlines()

            try:
                pkg_resources.require(dependencies)
            except pkg_resources.DistributionNotFound as e:
                print(f"[INFO] Dependência ausente: {e}. Instalando todas as dependências...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            except pkg_resources.VersionConflict as e:
                print(f"[INFO] Conflito de versão: {e}. Atualizando dependências...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", requirements_path])
    except Exception as err:
        print(f"[AVISO] Erro ao verificar dependências: {err}")
        print("Tentando instalar dependências manualmente...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        except Exception as e:
            print(f"[ERRO] Falha ao instalar dependências: {e}")
            sys.exit(1)

# Executa a verificação antes de importar as libs principais
instalar_dependencias()

import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QListWidget, QProgressBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ======================
# THREAD YOLO IMAGEM
# ======================
class YOLOThread(QThread):
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
            results = model(self.image_path)
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


# ======================
# THREAD WEBCAM / VÍDEO
# ======================
class WebcamThread(QThread):
    frame_updated = pyqtSignal(QImage, list)

    def __init__(self, model_path, source=0):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.source = source

    def run(self):
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.source)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

            detections = []
            for box in results[0].boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                nome = results[0].names[cls]
                detections.append((nome, conf))

            self.frame_updated.emit(qt_img, detections)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


# ======================
# INTERFACE PRINCIPAL
# ======================
class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Vision Studio")
        self.setGeometry(100, 100, 1200, 720)
        self.setWindowIcon(QIcon.fromTheme("camera"))
        self.model_path = None
        self.image_path = None
        self.webcam_thread = None
        self.thread = None

        # Layout geral
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === Barra lateral ===
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #141414;
                border-right: 1px solid #222;
            }
        """)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setAlignment(Qt.AlignTop)

        title = QLabel("🧠 YOLO Vision Studio")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ffae; font-size: 18px; font-weight: bold;")
        side_layout.addWidget(title)

        subtitle = QLabel("Detecção de Objetos\ncom IA")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #aaa; font-size: 13px; margin-bottom: 20px;")
        side_layout.addWidget(subtitle)

        # Botões da barra lateral
        def make_button(text, color="#00ffae"):
            btn = QPushButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: none;
                    border-radius: 8px;
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    margin: 5px 15px;
                }}
                QPushButton:hover {{
                    background-color: #00e69c;
                }}
            """)
            return btn

        self.btn_model = make_button("Selecionar Modelo")
        self.btn_model.clicked.connect(self.load_model)

        self.btn_image = make_button("Carregar Imagem", "#1e90ff")
        self.btn_image.clicked.connect(self.load_image)

        self.btn_detect = make_button("Detectar", "#00ffae")
        self.btn_detect.clicked.connect(self.detect_image)

        self.btn_webcam = make_button("Webcam", "#ffb100")
        self.btn_webcam.clicked.connect(self.toggle_webcam)

        self.btn_video = make_button("Vídeo Local", "#ff5f5f")
        self.btn_video.clicked.connect(self.load_video)

        self.btn_save = make_button("Salvar Resultado", "#888")
        self.btn_save.clicked.connect(self.save_result)

        side_layout.addWidget(self.btn_model)
        side_layout.addWidget(self.btn_image)
        side_layout.addWidget(self.btn_detect)
        side_layout.addWidget(self.btn_webcam)
        side_layout.addWidget(self.btn_video)
        side_layout.addWidget(self.btn_save)
        side_layout.addStretch()

        # === Área de exibição ===
        content = QVBoxLayout()
        content.setContentsMargins(20, 20, 20, 20)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border-radius: 10px;")
        content.addWidget(self.image_label)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                background-color: #2a2a2a;
                color: white;
                border-radius: 6px;
                height: 14px;
            }
            QProgressBar::chunk {
                background-color: #00ffae;
                width: 10px;
            }
        """)
        content.addWidget(self.progress)

        # === Painel de resultados ===
        result_box = QFrame()
        result_box.setStyleSheet("""
            QFrame {
                background-color: #141414;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        result_layout = QVBoxLayout(result_box)

        label = QLabel("🔍 Objetos Detectados")
        label.setStyleSheet("font-weight: bold; color: #00ffae; font-size: 15px;")
        result_layout.addWidget(label)

        self.list = QListWidget()
        self.list.setStyleSheet("""
            QListWidget {
                background-color: #1f1f1f;
                border: none;
                color: #ddd;
                padding: 8px;
                border-radius: 6px;
            }
            QListWidget::item:hover { background-color: #2a2a2a; }
        """)
        result_layout.addWidget(self.list)
        content.addWidget(result_box)

        main_layout.addWidget(sidebar)
        main_layout.addLayout(content)

        # === Fonte global ===
        app_font = QFont("Montserrat", 10)
        QApplication.setFont(app_font)

        self.setStyleSheet("background-color: #101010; color: white;")

    # ======================
    # FUNÇÕES
    # ======================
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Modelo YOLO", "", "Modelos (*.pt)")
        if file_path:
            self.model_path = file_path
            QMessageBox.information(self, "Modelo Carregado", f"Modelo selecionado:\n{file_path}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Imagens (*.jpg *.png *.jpeg)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.list.clear()
            self.progress.setValue(0)

    def display_image(self, path):
        pix = QPixmap(path).scaled(850, 500, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    def detect_image(self):
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, "Aviso", "Carregue o modelo e a imagem primeiro.")
            return
        self.thread = YOLOThread(self.model_path, self.image_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.show_result)
        self.thread.start()

    def show_result(self, output_path, detections):
        if not output_path:
            QMessageBox.critical(self, "Erro", "Erro na inferência.")
            return
        self.display_image(output_path)
        self.list.clear()
        if not detections:
            self.list.addItem("Nenhum objeto detectado.")
        else:
            for nome, conf in detections:
                self.list.addItem(f"{nome} - Confiança: {conf:.2f}")
        QMessageBox.information(self, "Concluído", "Detecção finalizada!")

    def save_result(self):
        if not self.image_label.pixmap():
            QMessageBox.warning(self, "Aviso", "Nada para salvar.")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Salvar Resultado", "saida.jpg", "Imagens (*.jpg *.png)")
        if file:
            self.image_label.pixmap().save(file)
            QMessageBox.information(self, "Salvo", f"Imagem salva em:\n{file}")

    def toggle_webcam(self):
        if not self.model_path:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo YOLO primeiro.")
            return
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread = None
            self.btn_webcam.setText("Webcam")
        else:
            self.webcam_thread = WebcamThread(self.model_path, 0)
            self.webcam_thread.frame_updated.connect(self.update_frame)
            self.webcam_thread.start()
            self.btn_webcam.setText("Fechar Webcam")

    def load_video(self):
        if not self.model_path:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo YOLO primeiro.")
            return
        file, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if file:
            self.webcam_thread = WebcamThread(self.model_path, file)
            self.webcam_thread.frame_updated.connect(self.update_frame)
            self.webcam_thread.start()

    def update_frame(self, img, detections):
        pix = QPixmap.fromImage(img).scaled(850, 500, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)
        self.list.clear()
        for nome, conf in detections:
            self.list.addItem(f"{nome} ({conf:.2f})")


# ======================
# EXECUÇÃO
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())