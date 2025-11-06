import sys
import os
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QListWidget, QProgressBar, QFrame,
    QComboBox, QButtonGroup, QRadioButton, QSplitter, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDir


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
            results = model(
                self.image_path,
                verbose=False,
                conf=0.5,
                device='0',
                half=True
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


# ======================
# THREAD WEBCAM / VÍDEO
# ======================
class WebcamThread(QThread):
    frame_updated = pyqtSignal(QImage, list, float)

    def __init__(self, model_path, source=0):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.source = source

    def run(self):
        import time
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.source)

        fps_counter = 0
        start_time = time.time()
        fps = 0.0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                verbose=False,
                conf=0.5,
                device='0',
                half=True
            )

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

            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = fps_counter / (time.time() - start_time)

            self.frame_updated.emit(qt_img, detections, fps)
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
        self.setWindowTitle("FEI - Vision Studio")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(900, 600)  # Tamanho mínimo da janela - mais flexível
        self.model_path = None
        self.source_path = None
        self.webcam_thread = None
        self.thread = None
        self.is_detecting = False
        self.detection_mode = "image"  # image, video, camera
        self.current_image_path = None  # Armazena o caminho da imagem atual
        self.current_scale = 1.0  # Escala atual da interface
        self.ui_initialized = False  # Flag para indicar se a UI está completamente inicializada

        # Layout geral
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter para tornar redimensionável
        self.splitter = QSplitter(Qt.Horizontal)

        # === Barra lateral ===
        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(400)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e5e7eb;
            }
        """)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(20, 30, 20, 20)
        side_layout.setSpacing(20)
        side_layout.setAlignment(Qt.AlignTop)

        # Logo e título
        logo_container = QHBoxLayout()
        self.logo_label = QLabel("👷")
        self.logo_label.setStyleSheet("font-size: 28px;")
        self.title_label = QLabel("FEI Vision Studio")
        self.title_label.setStyleSheet("""
            color: #111827;
            font-size: 18px;
            font-weight: 600;
            margin-left: 8px;
        """)
        logo_container.addWidget(self.logo_label)
        logo_container.addWidget(self.title_label)
        logo_container.addStretch()
        side_layout.addLayout(logo_container)

        self.subtitle_label = QLabel("Detecção de Objetos")
        self.subtitle_label.setStyleSheet("""
            color: #6b7280;
            font-size: 13px;
            margin-bottom: 5px;
        """)
        side_layout.addWidget(self.subtitle_label)

        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #e5e7eb; max-height: 1px;")
        side_layout.addWidget(separator)

        # Seção Selecionar Modelo
        self.model_label = QLabel("Modelo YOLO")
        self.model_label.setStyleSheet("""
            color: #374151;
            font-size: 12px;
            font-weight: 600;
            margin-top: 5px;
            margin-bottom: 8px;
        """)
        side_layout.addWidget(self.model_label)

        # ComboBox para modelos disponíveis
        self.model_combo = QComboBox()
        self.load_available_models()
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #f9fafb;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px 12px;
                color: #111827;
                font-size: 13px;
            }
            QComboBox:hover {
                border-color: #3b82f6;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #6b7280;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                selection-background-color: #dbeafe;
                selection-color: #1e40af;
                padding: 4px;
            }
        """)
        side_layout.addWidget(self.model_combo)

        # Separador
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet("background-color: #e5e7eb; max-height: 1px; margin-top: 10px;")
        side_layout.addWidget(separator2)

        # Seção Tipo de Detecção
        self.source_label = QLabel("Tipo de Detecção")
        self.source_label.setStyleSheet("""
            color: #374151;
            font-size: 12px;
            font-weight: 600;
            margin-top: 5px;
            margin-bottom: 8px;
        """)
        side_layout.addWidget(self.source_label)

        # Radio buttons para escolher tipo
        self.source_group = QButtonGroup()
        
        # Opção: Imagem
        self.radio_image = QRadioButton("📷  Imagem")
        self.radio_image.setChecked(True)
        self.radio_image.setCursor(Qt.PointingHandCursor)
        self.radio_image.setStyleSheet("""
            QRadioButton {
                color: #374151;
                font-size: 13px;
                padding: 8px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #d1d5db;
                background-color: #ffffff;
            }
            QRadioButton::indicator:hover {
                border-color: #3b82f6;
            }
            QRadioButton::indicator:checked {
                border-color: #3b82f6;
                background-color: #3b82f6;
            }
            QRadioButton::indicator:checked:after {
                content: '';
                width: 8px;
                height: 8px;
                border-radius: 4px;
                background-color: white;
            }
        """)
        self.radio_image.toggled.connect(lambda: self.set_detection_mode("image"))
        
        # Opção: Vídeo
        self.radio_video = QRadioButton("🎬  Vídeo")
        self.radio_video.setCursor(Qt.PointingHandCursor)
        self.radio_video.setStyleSheet(self.radio_image.styleSheet())
        self.radio_video.toggled.connect(lambda: self.set_detection_mode("video"))
        
        # Opção: Câmera/Vídeo IRL
        self.radio_camera = QRadioButton("📹  Vídeo IRL ou Câmera")
        self.radio_camera.setCursor(Qt.PointingHandCursor)
        self.radio_camera.setStyleSheet(self.radio_image.styleSheet())
        self.radio_camera.toggled.connect(lambda: self.set_detection_mode("camera"))
        
        self.source_group.addButton(self.radio_image)
        self.source_group.addButton(self.radio_video)
        self.source_group.addButton(self.radio_camera)
        
        side_layout.addWidget(self.radio_image)
        side_layout.addWidget(self.radio_video)
        side_layout.addWidget(self.radio_camera)

        # Separador
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setStyleSheet("background-color: #e5e7eb; max-height: 1px; margin-top: 10px;")
        side_layout.addWidget(separator3)

        # Botão carregar fonte
        self.source_btn_label = QLabel("Carregar Fonte")
        self.source_btn_label.setStyleSheet("""
            color: #374151;
            font-size: 12px;
            font-weight: 600;
            margin-top: 5px;
            margin-bottom: 8px;
        """)
        side_layout.addWidget(self.source_btn_label)

        self.btn_load_source = self.create_primary_button("📂  Selecionar Arquivo", "#3b82f6")
        self.btn_load_source.clicked.connect(self.load_source)
        side_layout.addWidget(self.btn_load_source)

        # Separador
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.HLine)
        separator4.setStyleSheet("background-color: #e5e7eb; max-height: 1px; margin-top: 10px;")
        side_layout.addWidget(separator4)

        # Botão Salvar
        self.save_label = QLabel("Salvar Resultado")
        self.save_label.setStyleSheet("""
            color: #374151;
            font-size: 12px;
            font-weight: 600;
            margin-top: 5px;
            margin-bottom: 8px;
        """)
        side_layout.addWidget(self.save_label)

        self.btn_save = self.create_secondary_button("💾  Salvar")
        self.btn_save.clicked.connect(self.save_result)
        side_layout.addWidget(self.btn_save)

        side_layout.addStretch()

        # Botão Iniciar/Parar Detecção (no final)
        self.btn_detect = self.create_action_button("▶  Iniciar Detecção", "#10b981")
        self.btn_detect.clicked.connect(self.toggle_detection)
        side_layout.addWidget(self.btn_detect)

        # === Área de conteúdo principal ===
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("background-color: #f9fafb;")
        self.content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(20)

        # Área de exibição da imagem
        self.image_container = QFrame()
        self.image_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 12px;
            }
        """)
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)
        
        # Placeholder quando não há imagem
        self.setup_placeholder()
        
        image_layout.addWidget(self.image_label)

        self.content_layout.addWidget(self.image_container, stretch=1)

        # Barra de progresso
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #e5e7eb;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        self.content_layout.addWidget(self.progress)

        # === Painel de resultados ===
        result_container = QFrame()
        result_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(20, 20, 20, 20)
        result_layout.setSpacing(12)

        self.result_header = QLabel("Objetos Detectados")
        self.result_header.setStyleSheet("""
            font-weight: 600;
            color: #111827;
            font-size: 15px;
        """)
        result_layout.addWidget(self.result_header)

        self.list = QListWidget()
        self.list.setStyleSheet("""
            QListWidget {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                color: #374151;
                padding: 8px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 6px;
                margin: 2px 0px;
            }
            QListWidget::item:hover {
                background-color: #eff6ff;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
                color: #1e40af;
            }
        """)
        self.list.addItem("Nenhum objeto detectado ainda. Selecione uma fonte e clique em 'Iniciar Detecção'.")
        result_layout.addWidget(self.list)

        self.content_layout.addWidget(result_container)

        # Adicionar widgets ao splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.content_frame)
        self.splitter.setStretchFactor(0, 0)  # sidebar não se expande
        self.splitter.setStretchFactor(1, 1)  # content_frame se expande
        self.splitter.setSizes([280, 1120])  # Tamanho inicial

        main_layout.addWidget(self.splitter)

        # Aplicar estilo global
        self.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            }
        """)

        # Marca que a UI foi completamente inicializada
        self.ui_initialized = True

    def resizeEvent(self, event):
        """Evento chamado quando a janela é redimensionada"""
        super().resizeEvent(event)

        # Só aplica responsividade se a UI estiver completamente inicializada
        if not self.ui_initialized:
            return

        # Calcula a nova escala baseada no tamanho da janela
        self._update_responsive_scale()
        # Atualiza a imagem se houver uma carregada
        if self.current_image_path and self.image_label.pixmap():
            self._update_displayed_image()

    def _update_responsive_scale(self):
        """Calcula e aplica escala responsiva baseada no tamanho da janela"""
        width = self.width()
        height = self.height()

        # Define breakpoints e escalas
        # Base: 1400x900 (tamanho padrão) = escala 1.0
        base_width = 1400
        base_height = 900

        # Calcula escala baseada na largura e altura
        width_scale = width / base_width
        height_scale = height / base_height

        # Usa a menor escala para garantir que tudo caiba
        new_scale = min(width_scale, height_scale)

        # Limita a escala entre 0.7 e 1.3
        new_scale = max(0.7, min(1.3, new_scale))

        # Se a escala mudou significativamente, atualiza os componentes
        if abs(new_scale - self.current_scale) > 0.05:
            self.current_scale = new_scale
            self._apply_responsive_styles()

    def _apply_responsive_styles(self):
        """Aplica estilos responsivos a todos os componentes"""
        # Verifica se a UI está inicializada
        if not self.ui_initialized:
            return

        scale = self.current_scale

        # Ajusta tamanhos de fonte
        logo_size = int(28 * scale)
        title_size = int(18 * scale)
        subtitle_size = int(13 * scale)
        label_size = int(12 * scale)
        button_size = int(13 * scale)
        action_button_size = int(14 * scale)
        result_header_size = int(15 * scale)

        # Ajusta margens e paddings
        sidebar_margin_h = int(20 * scale)
        sidebar_margin_v = int(30 * scale)
        content_margin = int(30 * scale)
        spacing = int(20 * scale)

        # Atualiza largura da sidebar
        sidebar_min = int(250 * scale)
        sidebar_max = int(400 * scale)
        self.sidebar.setMinimumWidth(max(200, sidebar_min))
        self.sidebar.setMaximumWidth(sidebar_max)

        # Atualiza margens dos layouts
        self.sidebar.layout().setContentsMargins(sidebar_margin_h, sidebar_margin_v, sidebar_margin_h, int(20 * scale))
        self.sidebar.layout().setSpacing(int(20 * scale))
        self.content_layout.setContentsMargins(content_margin, content_margin, content_margin, content_margin)
        self.content_layout.setSpacing(spacing)

        # Atualiza logo
        self.logo_label.setStyleSheet(f"font-size: {logo_size}px;")

        # Atualiza título
        self.title_label.setStyleSheet(f"""
            color: #111827;
            font-size: {title_size}px;
            font-weight: 600;
            margin-left: 8px;
        """)

        # Atualiza subtitle
        self.subtitle_label.setStyleSheet(f"""
            color: #6b7280;
            font-size: {subtitle_size}px;
            margin-bottom: 5px;
        """)

        # Atualiza labels de seção
        label_style = f"""
            color: #374151;
            font-size: {label_size}px;
            font-weight: 600;
            margin-top: 5px;
            margin-bottom: 8px;
        """
        self.model_label.setStyleSheet(label_style)
        self.source_label.setStyleSheet(label_style)
        self.source_btn_label.setStyleSheet(label_style)
        self.save_label.setStyleSheet(label_style)

        # Atualiza header de resultados
        self.result_header.setStyleSheet(f"""
            font-weight: 600;
            color: #111827;
            font-size: {result_header_size}px;
        """)

        # Atualiza ComboBox
        combo_padding = int(10 * scale)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: #f9fafb;
                border: 1px solid #d1d5db;
                border-radius: {int(8 * scale)}px;
                padding: {combo_padding}px {int(12 * scale)}px;
                color: #111827;
                font-size: {button_size}px;
            }}
            QComboBox:hover {{
                border-color: #3b82f6;
                background-color: #ffffff;
            }}
            QComboBox::drop-down {{
                border: none;
                width: {int(25 * scale)}px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #6b7280;
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                selection-background-color: #dbeafe;
                selection-color: #1e40af;
                padding: 4px;
            }}
        """)

        # Atualiza radio buttons
        radio_style = f"""
            QRadioButton {{
                color: #374151;
                font-size: {button_size}px;
                padding: {int(8 * scale)}px;
                spacing: {int(8 * scale)}px;
            }}
            QRadioButton::indicator {{
                width: {int(18 * scale)}px;
                height: {int(18 * scale)}px;
                border-radius: {int(9 * scale)}px;
                border: 2px solid #d1d5db;
                background-color: #ffffff;
            }}
            QRadioButton::indicator:hover {{
                border-color: #3b82f6;
            }}
            QRadioButton::indicator:checked {{
                border-color: #3b82f6;
                background-color: #3b82f6;
            }}
            QRadioButton::indicator:checked:after {{
                content: '';
                width: {int(8 * scale)}px;
                height: {int(8 * scale)}px;
                border-radius: {int(4 * scale)}px;
                background-color: white;
            }}
        """
        self.radio_image.setStyleSheet(radio_style)
        self.radio_video.setStyleSheet(radio_style)
        self.radio_camera.setStyleSheet(radio_style)

        # Atualiza botões
        btn_padding_v = int(11 * scale)
        btn_padding_h = int(16 * scale)

        self.btn_load_source.setStyleSheet(f"""
            QPushButton {{
                background-color: #3b82f6;
                border: none;
                border-radius: {int(8 * scale)}px;
                color: white;
                font-weight: 500;
                font-size: {button_size}px;
                padding: {btn_padding_v}px {btn_padding_h}px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #2563eb;
            }}
            QPushButton:pressed {{
                background-color: #1d4ed8;
            }}
        """)

        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: {int(8 * scale)}px;
                color: #374151;
                font-weight: 500;
                font-size: {button_size}px;
                padding: {btn_padding_v}px {btn_padding_h}px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #e5e7eb;
                border-color: #9ca3af;
            }}
            QPushButton:pressed {{
                background-color: #d1d5db;
            }}
        """)

        # Atualiza botão de detecção
        action_padding_v = int(14 * scale)
        action_padding_h = int(16 * scale)

        if self.is_detecting:
            bg_color = "#ef4444"
            hover_color = "#dc2626"
            pressed_color = "#b91c1c"
        else:
            bg_color = "#10b981"
            hover_color = "#059669"
            pressed_color = "#047857"

        self.btn_detect.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                border-radius: {int(10 * scale)}px;
                color: white;
                font-weight: 600;
                font-size: {action_button_size}px;
                padding: {action_padding_v}px {action_padding_h}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """)

        # Atualiza lista de detecções
        self.list.setStyleSheet(f"""
            QListWidget {{
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: {int(8 * scale)}px;
                color: #374151;
                padding: {int(8 * scale)}px;
                font-size: {button_size}px;
            }}
            QListWidget::item {{
                padding: {int(8 * scale)}px;
                border-radius: {int(6 * scale)}px;
                margin: {int(2 * scale)}px 0px;
            }}
            QListWidget::item:hover {{
                background-color: #eff6ff;
            }}
            QListWidget::item:selected {{
                background-color: #dbeafe;
                color: #1e40af;
            }}
        """)

        # Atualiza altura mínima da imagem
        self.image_label.setMinimumHeight(int(300 * scale))

    def setup_placeholder(self):
        """Configura o placeholder visual sem cortar conteúdo"""
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        placeholder_container = QFrame()
        placeholder_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)

        placeholder_layout = QVBoxLayout(placeholder_container)
        placeholder_layout.setAlignment(Qt.AlignCenter)
        placeholder_layout.setSpacing(12)
        placeholder_layout.setContentsMargins(10, 10, 10, 10)

        icon_placeholder = QLabel("🖼")
        icon_placeholder.setAlignment(Qt.AlignCenter)
        icon_placeholder.setStyleSheet("font-size: 48px;")

        text_placeholder = QLabel("Nenhuma imagem carregada")
        text_placeholder.setAlignment(Qt.AlignCenter)
        text_placeholder.setStyleSheet("color: #111827; font-size: 14px; font-weight: 600;")

        subtext_placeholder = QLabel("Selecione uma imagem, vídeo ou ative a webcam para começar")
        subtext_placeholder.setAlignment(Qt.AlignCenter)
        subtext_placeholder.setWordWrap(True)
        subtext_placeholder.setStyleSheet("color: #6b7280; font-size: 12px;")
        subtext_placeholder.setMaximumWidth(300)

        placeholder_layout.addWidget(icon_placeholder)
        placeholder_layout.addWidget(text_placeholder)
        placeholder_layout.addWidget(subtext_placeholder)

        main_placeholder_layout = QVBoxLayout()
        main_placeholder_layout.setAlignment(Qt.AlignCenter)
        main_placeholder_layout.addWidget(placeholder_container)

        self.image_label.setLayout(main_placeholder_layout)


    def load_available_models(self):
        """Carrega modelos .pt disponíveis no diretório"""
        self.model_combo.clear()
        self.model_combo.addItem("Selecione um modelo...")
        
        # Procura por arquivos .pt no diretório atual e subdiretórios comuns
        search_dirs = ['.', './models', './weights', '../models', '../weights']
        found_models = []
        
        for directory in search_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.pt'):
                        model_path = os.path.join(directory, file)
                        found_models.append((file, model_path))
        
        # Remove duplicatas
        found_models = list(set(found_models))
        found_models.sort()
        
        for model_name, model_path in found_models:
            self.model_combo.addItem(model_name, model_path)
        
        if not found_models:
            self.model_combo.addItem("Nenhum modelo encontrado")

    def on_model_selected(self, model_name):
        """Callback quando um modelo é selecionado"""
        if model_name and model_name not in ["Selecione um modelo...", "Nenhum modelo encontrado"]:
            self.model_path = self.model_combo.currentData()
            print(f"Modelo selecionado: {self.model_path}")

    def set_detection_mode(self, mode):
        """Define o modo de detecção"""
        self.detection_mode = mode
        if mode == "camera":
            self.btn_load_source.setText("📹  Ativar Câmera")
        elif mode == "video":
            self.btn_load_source.setText("🎬  Selecionar Vídeo")
        else:
            self.btn_load_source.setText("📷  Selecionar Imagem")

    def create_primary_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                font-size: 13px;
                padding: 11px 16px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #2563eb;
            }}
            QPushButton:pressed {{
                background-color: #1d4ed8;
            }}
        """)
        return btn

    def create_secondary_button(self, text):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                color: #374151;
                font-weight: 500;
                font-size: 13px;
                padding: 11px 16px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #d1d5db;
            }
        """)
        return btn

    def create_action_button(self, text, color):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                font-size: 14px;
                padding: 14px 16px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
            QPushButton:pressed {{
                background-color: #047857;
            }}
        """)
        return btn

    # ======================
    # FUNÇÕES
    # ======================
    def load_source(self):
        """Carrega a fonte (imagem, vídeo ou câmera)"""
        if self.detection_mode == "camera":
            self.source_path = 0
            QMessageBox.information(self, "Câmera", "Câmera selecionada. Clique em 'Iniciar Detecção' para começar.")
        elif self.detection_mode == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.source_path = file_path
                self.display_placeholder_with_text("Vídeo carregado", "Clique em 'Iniciar Detecção' para processar")
        else:  # image
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Selecionar Imagem", "", "Imagens (*.jpg *.png *.jpeg *.bmp)"
            )
            if file_path:
                self.source_path = file_path
                self.display_image(file_path)
                self.list.clear()
                self.progress.setValue(0)

    def display_image(self, path):
        """Exibe uma imagem redimensionando dinamicamente"""
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        self.current_image_path = path
        self._update_displayed_image()
        self.image_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)

    def _update_displayed_image(self):
        """Atualiza a exibição da imagem com base no tamanho disponível"""
        if not self.current_image_path:
            return

        # Calcula o tamanho disponível
        available_width = self.image_label.width() - 40
        available_height = self.image_label.height() - 40

        if available_width <= 0 or available_height <= 0:
            available_width = 800
            available_height = 600

        pix = QPixmap(self.current_image_path).scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def display_placeholder_with_text(self, main_text, sub_text):
        """Exibe placeholder com texto customizado"""
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        self.current_image_path = None  # Limpa o caminho da imagem

        # Container principal com borda (tamanho flexível)
        placeholder_container = QFrame()
        placeholder_container.setMinimumSize(300, 250)
        placeholder_container.setMaximumSize(500, 400)
        placeholder_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 35px;
            }
        """)
        
        placeholder_layout = QVBoxLayout(placeholder_container)
        placeholder_layout.setAlignment(Qt.AlignCenter)
        placeholder_layout.setSpacing(10)
        placeholder_layout.setContentsMargins(25, 25, 25, 25)
        
        icon = QLabel("✓")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 52px; color: #10b981; background: transparent; border: none;")
        
        text = QLabel(main_text)
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("color: #111827; font-size: 15px; font-weight: 600; background: transparent; border: none; margin-top: 10px;")
        
        subtext = QLabel(sub_text)
        subtext.setAlignment(Qt.AlignCenter)
        subtext.setStyleSheet("color: #6b7280; font-size: 12px; background: transparent; border: none; line-height: 1.4;")
        subtext.setWordWrap(True)
        subtext.setMaximumWidth(360)
        
        placeholder_layout.addWidget(icon)
        placeholder_layout.addWidget(text)
        placeholder_layout.addWidget(subtext)
        
        # Layout para centralizar o container
        main_placeholder_layout = QVBoxLayout()
        main_placeholder_layout.setAlignment(Qt.AlignCenter)
        main_placeholder_layout.addWidget(placeholder_container)
        
        self.image_label.setLayout(main_placeholder_layout)

    def toggle_detection(self):
        """Inicia ou para a detecção"""
        if not self.model_path or self.model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo YOLO primeiro.")
            return
        
        if not self.source_path:
            QMessageBox.warning(self, "Aviso", "Carregue uma fonte primeiro (imagem, vídeo ou câmera).")
            return

        if self.is_detecting:
            # Parar detecção
            self.stop_detection()
        else:
            # Iniciar detecção
            self.start_detection()

    def start_detection(self):
        """Inicia a detecção"""
        self.is_detecting = True
        self.btn_detect.setText("⏸  Parar Detecção")
        self._apply_responsive_styles()  # Reaplica estilos com a nova cor

        if self.detection_mode == "image":
            self.detect_image()
        else:
            self.detect_video()

    def stop_detection(self):
        """Para a detecção"""
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread = None

        self.is_detecting = False
        self.btn_detect.setText("▶  Iniciar Detecção")
        self._apply_responsive_styles()  # Reaplica estilos com a nova cor

    def detect_image(self):
        """Detecta objetos em imagem"""
        self.thread = YOLOThread(self.model_path, self.source_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.show_result)
        self.thread.start()

    def detect_video(self):
        """Detecta objetos em vídeo ou câmera"""
        self.webcam_thread = WebcamThread(self.model_path, self.source_path)
        self.webcam_thread.frame_updated.connect(self.update_frame)
        self.webcam_thread.start()

    def show_result(self, output_path, detections):
        """Mostra resultado da detecção em imagem"""
        self.is_detecting = False
        self.btn_detect.setText("▶  Iniciar Detecção")
        self._apply_responsive_styles()  # Reaplica estilos com a nova cor
        
        if not output_path:
            QMessageBox.critical(self, "Erro", "Erro na inferência.")
            return
        
        self.display_image(output_path)
        self.list.clear()
        if not detections:
            self.list.addItem("Nenhum objeto detectado.")
        else:
            for nome, conf in detections:
                self.list.addItem(f"✓  {nome} - Confiança: {conf:.2%}")
        
        QMessageBox.information(self, "Concluído", "Detecção finalizada!")

    def save_result(self):
        """Salva o resultado"""
        if not self.image_label.pixmap():
            QMessageBox.warning(self, "Aviso", "Nada para salvar.")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Salvar Resultado", "saida.jpg", "Imagens (*.jpg *.png)")
        if file:
            self.image_label.pixmap().save(file)
            QMessageBox.information(self, "Salvo", f"Imagem salva em:\n{file}")

    def update_frame(self, img, detections, fps):
        """Atualiza frame do vídeo/câmera"""
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        # Calcula o tamanho disponível
        available_width = self.image_label.width() - 40
        available_height = self.image_label.height() - 40

        if available_width <= 0 or available_height <= 0:
            available_width = 800
            available_height = 600

        pix = QPixmap.fromImage(img).scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)
        self.list.clear()

        if not detections:
            self.list.addItem("Nenhum objeto detectado no frame.")
        else:
            for nome, conf in detections:
                self.list.addItem(f"✓  {nome} ({conf:.2%})")


# ======================
# EXECUÇÃO
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())