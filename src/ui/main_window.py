"""
Janela principal do aplicativo FEI Vision Studio
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QListWidget, QProgressBar, QFrame,
    QComboBox, QButtonGroup, QRadioButton, QSplitter, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ..threads import YOLOThread, VideoThread
from ..utils.image_utils import display_image_scaled, create_placeholder, create_custom_placeholder
from . import styles


class YOLOApp(QWidget):
    """Janela principal da aplicação de detecção YOLO"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEI - Vision Studio")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(900, 600)

        # Atributos de estado
        self.model_path = None
        self.source_path = None
        self.video_thread = None
        self.thread = None
        self.is_detecting = False
        self.detection_mode = "image"
        self.current_image_path = None
        self.current_scale = 1.0
        self.ui_initialized = False

        # Construir interface
        self._setup_ui()
        self.ui_initialized = True

    def _setup_ui(self):
        """Configura a interface completa"""
        # Layout geral
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter para tornar redimensionável
        self.splitter = QSplitter(Qt.Horizontal)

        # Criar sidebar e área de conteúdo
        self._create_sidebar()
        self._create_content_area()

        # Adicionar ao splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.content_frame)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([280, 1120])

        main_layout.addWidget(self.splitter)

        # Aplicar estilo global
        self.setStyleSheet(styles.GLOBAL_STYLE)

    def _create_sidebar(self):
        """Cria a barra lateral com controles"""
        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(400)
        self.sidebar.setStyleSheet(styles.SIDEBAR_STYLE)

        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(20, 30, 20, 20)
        side_layout.setSpacing(20)
        side_layout.setAlignment(Qt.AlignTop)

        # Logo e título
        self._add_header(side_layout)

        # Separador
        side_layout.addWidget(self._create_separator())

        # Seção Modelo
        self._add_model_section(side_layout)

        # Separador
        side_layout.addWidget(self._create_separator())

        # Seção Tipo de Detecção
        self._add_detection_type_section(side_layout)

        # Separador
        side_layout.addWidget(self._create_separator())

        # Botão carregar fonte
        self._add_load_source_section(side_layout)

        # Separador
        side_layout.addWidget(self._create_separator())

        # Botão Salvar
        self._add_save_section(side_layout)

        side_layout.addStretch()

        # Botão Iniciar/Parar
        self._add_action_button(side_layout)

    def _add_header(self, layout):
        """Adiciona cabeçalho com logo e título"""
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
        layout.addLayout(logo_container)

        self.subtitle_label = QLabel("Detecção de Objetos")
        self.subtitle_label.setStyleSheet("""
            color: #6b7280;
            font-size: 13px;
            margin-bottom: 5px;
        """)
        layout.addWidget(self.subtitle_label)

    def _add_model_section(self, layout):
        """Adiciona seção de seleção de modelo"""
        self.model_label = QLabel("Modelo YOLO")
        self.model_label.setStyleSheet(styles.get_label_style(12))
        layout.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self._load_available_models()
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        self.model_combo.setStyleSheet(styles.get_combo_box_style())
        layout.addWidget(self.model_combo)

    def _add_detection_type_section(self, layout):
        """Adiciona seção de tipo de detecção"""
        self.source_label = QLabel("Tipo de Detecção")
        self.source_label.setStyleSheet(styles.get_label_style(12))
        layout.addWidget(self.source_label)

        self.source_group = QButtonGroup()

        self.radio_image = QRadioButton("📷  Imagem")
        self.radio_image.setChecked(True)
        self.radio_image.setCursor(Qt.PointingHandCursor)
        self.radio_image.setStyleSheet(styles.get_radio_button_style())
        self.radio_image.toggled.connect(lambda: self._set_detection_mode("image"))

        self.radio_video = QRadioButton("🎬  Vídeo")
        self.radio_video.setCursor(Qt.PointingHandCursor)
        self.radio_video.setStyleSheet(styles.get_radio_button_style())
        self.radio_video.toggled.connect(lambda: self._set_detection_mode("video"))

        self.source_group.addButton(self.radio_image)
        self.source_group.addButton(self.radio_video)

        layout.addWidget(self.radio_image)
        layout.addWidget(self.radio_video)

    def _add_load_source_section(self, layout):
        """Adiciona seção de carregar fonte"""
        self.source_btn_label = QLabel("Carregar Fonte")
        self.source_btn_label.setStyleSheet(styles.get_label_style(12))
        layout.addWidget(self.source_btn_label)

        self.btn_load_source = QPushButton("📂  Selecionar Arquivo")
        self.btn_load_source.setCursor(Qt.PointingHandCursor)
        self.btn_load_source.setStyleSheet(styles.get_primary_button_style())
        self.btn_load_source.clicked.connect(self._load_source)
        layout.addWidget(self.btn_load_source)

    def _add_save_section(self, layout):
        """Adiciona seção de salvar"""
        self.save_label = QLabel("Salvar Resultado")
        self.save_label.setStyleSheet(styles.get_label_style(12))
        layout.addWidget(self.save_label)

        self.btn_save = QPushButton("💾  Salvar")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet(styles.get_secondary_button_style())
        self.btn_save.clicked.connect(self._save_result)
        layout.addWidget(self.btn_save)

    def _add_action_button(self, layout):
        """Adiciona botão de iniciar/parar"""
        self.btn_detect = QPushButton("▶  Iniciar Detecção")
        self.btn_detect.setCursor(Qt.PointingHandCursor)
        self.btn_detect.setStyleSheet(styles.get_action_button_style(False))
        self.btn_detect.clicked.connect(self._toggle_detection)
        layout.addWidget(self.btn_detect)

    def _create_content_area(self):
        """Cria área de conteúdo principal"""
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet("background-color: #f9fafb;")
        self.content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(20)

        # Área de imagem
        self._create_image_area()

        # Barra de progresso
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(styles.PROGRESS_BAR_STYLE)
        self.content_layout.addWidget(self.progress)

        # Painel de resultados
        self._create_results_panel()

    def _create_image_area(self):
        """Cria área de exibição de imagem"""
        self.image_container = QFrame()
        self.image_container.setStyleSheet(styles.IMAGE_CONTAINER_STYLE)
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        # Placeholder inicial
        self._setup_placeholder()

        image_layout.addWidget(self.image_label)
        self.content_layout.addWidget(self.image_container, stretch=1)

    def _create_results_panel(self):
        """Cria painel de resultados"""
        result_container = QFrame()
        result_container.setStyleSheet(styles.RESULT_CONTAINER_STYLE)

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
        self.list.setStyleSheet(styles.get_list_widget_style())
        self.list.addItem("Nenhum objeto detectado ainda. Selecione uma fonte e clique em 'Iniciar Detecção'.")
        result_layout.addWidget(self.list)

        self.content_layout.addWidget(result_container)

    def _create_separator(self):
        """Cria uma linha separadora"""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #e5e7eb; max-height: 1px; margin-top: 10px;")
        return separator

    def _setup_placeholder(self):
        """Configura o placeholder inicial"""
        placeholder_layout = create_placeholder()
        self.image_label.setLayout(placeholder_layout)

    def _load_available_models(self):
        """Carrega modelos .pt disponíveis"""
        self.model_combo.clear()
        self.model_combo.addItem("Selecione um modelo...")

        search_dirs = ['.', './models', './weights', '../models', '../weights']
        found_models = []

        for directory in search_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.pt'):
                        model_path = os.path.join(directory, file)
                        found_models.append((file, model_path))

        found_models = list(set(found_models))
        found_models.sort()

        for model_name, model_path in found_models:
            self.model_combo.addItem(model_name, model_path)

        if not found_models:
            self.model_combo.addItem("Nenhum modelo encontrado")

    def _on_model_selected(self, model_name):
        """Callback quando um modelo é selecionado"""
        if model_name and model_name not in ["Selecione um modelo...", "Nenhum modelo encontrado"]:
            self.model_path = self.model_combo.currentData()
            print(f"Modelo selecionado: {self.model_path}")

    def _set_detection_mode(self, mode):
        """Define o modo de detecção"""
        self.detection_mode = mode
        if mode == "video":
            self.btn_load_source.setText("🎬  Selecionar Vídeo")
        else:
            self.btn_load_source.setText("📷  Selecionar Imagem")

    def _load_source(self):
        """Carrega a fonte (imagem ou vídeo)"""
        if self.detection_mode == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mov *.mkv)"
            )
            if file_path:
                self.source_path = file_path
                self._display_placeholder_with_text("Vídeo carregado", "Clique em 'Iniciar Detecção' para processar")
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Selecionar Imagem", "", "Imagens (*.jpg *.png *.jpeg *.bmp)"
            )
            if file_path:
                self.source_path = file_path
                self._display_image(file_path)
                self.list.clear()
                self.progress.setValue(0)

    def _display_image(self, path):
        """Exibe uma imagem"""
        from PyQt5.QtWidgets import QWidget
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        self.current_image_path = path
        self._update_displayed_image()

    def _update_displayed_image(self):
        """Atualiza a exibição da imagem"""
        if not self.current_image_path:
            return

        display_image_scaled(self.image_label, self.current_image_path)

    def _display_placeholder_with_text(self, main_text, sub_text):
        """Exibe placeholder com texto customizado"""
        from PyQt5.QtWidgets import QWidget
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

        self.current_image_path = None
        placeholder_layout = create_custom_placeholder("✓", main_text, sub_text)
        self.image_label.setLayout(placeholder_layout)

    def _toggle_detection(self):
        """Inicia ou para a detecção"""
        if not self.model_path or self.model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo YOLO primeiro.")
            return

        if not self.source_path:
            QMessageBox.warning(self, "Aviso", "Carregue uma fonte primeiro (imagem ou vídeo).")
            return

        if self.is_detecting:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self):
        """Inicia a detecção"""
        self.is_detecting = True
        self.btn_detect.setText("⏸  Parar Detecção")
        self.btn_detect.setStyleSheet(styles.get_action_button_style(True, self.current_scale))

        if self.detection_mode == "image":
            self._detect_image()
        else:
            self._detect_video()

    def _stop_detection(self):
        """Para a detecção"""
        try:
            if self.video_thread:
                print("Parando thread de vídeo...")
                # Desconectar sinais para evitar problemas
                try:
                    self.video_thread.frame_updated.disconnect()
                except:
                    pass

                # Parar thread se estiver rodando
                if self.video_thread.isRunning():
                    self.video_thread.stop()

                # Garantir que thread foi parada
                self.video_thread.deleteLater()
                self.video_thread = None
                print("Thread de vídeo parada com sucesso")
        except Exception as e:
            print(f"Erro ao parar thread de vídeo: {e}")
            import traceback
            traceback.print_exc()

        self.is_detecting = False
        self.btn_detect.setText("▶  Iniciar Detecção")
        self.btn_detect.setStyleSheet(styles.get_action_button_style(False, self.current_scale))

    def _detect_image(self):
        """Detecta objetos em imagem"""
        self.thread = YOLOThread(self.model_path, self.source_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self._show_result)
        self.thread.start()

    def _detect_video(self):
        """Detecta objetos em vídeo"""
        # Parar thread anterior se existir
        if self.video_thread:
            print("Limpando thread anterior...")
            try:
                self.video_thread.frame_updated.disconnect()
            except:
                pass
            if self.video_thread.isRunning():
                self.video_thread.stop()
            self.video_thread.deleteLater()
            self.video_thread = None

        # Criar e iniciar nova thread
        print(f"Iniciando detecção de vídeo: {self.source_path}")
        self.video_thread = VideoThread(self.model_path, self.source_path, max_size=1280)
        self.video_thread.frame_updated.connect(self._update_frame)
        self.video_thread.start()

    def _show_result(self, output_path, detections):
        """Mostra resultado da detecção em imagem"""
        self.is_detecting = False
        self.btn_detect.setText("▶  Iniciar Detecção")
        self.btn_detect.setStyleSheet(styles.get_action_button_style(False, self.current_scale))

        if not output_path:
            QMessageBox.critical(self, "Erro", "Erro na inferência.")
            return

        self._display_image(output_path)
        self.list.clear()
        if not detections:
            self.list.addItem("Nenhum objeto detectado.")
        else:
            for nome, conf in detections:
                self.list.addItem(f"✓  {nome} - Confiança: {conf:.2%}")

        QMessageBox.information(self, "Concluído", "Detecção finalizada!")

    def _save_result(self):
        """Salva o resultado"""
        if not self.image_label.pixmap():
            QMessageBox.warning(self, "Aviso", "Nada para salvar.")
            return

        file, _ = QFileDialog.getSaveFileName(self, "Salvar Resultado", "saida.jpg", "Imagens (*.jpg *.png)")
        if file:
            self.image_label.pixmap().save(file)
            QMessageBox.information(self, "Salvo", f"Imagem salva em:\n{file}")

    def _update_frame(self, img, detections, fps):
        """Atualiza frame do vídeo"""
        from PyQt5.QtWidgets import QWidget
        if self.image_label.layout():
            QWidget().setLayout(self.image_label.layout())

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

    def closeEvent(self, event):
        """Evento de fechamento da janela - limpar threads"""
        try:
            # Parar thread de vídeo se estiver rodando
            if self.video_thread and self.video_thread.isRunning():
                print("Parando thread de vídeo...")
                self.video_thread.stop()

            # Parar thread de imagem se estiver rodando
            if self.thread and self.thread.isRunning():
                print("Parando thread de imagem...")
                self.thread.quit()
                self.thread.wait(2000)
        except Exception as e:
            print(f"Erro ao limpar threads: {e}")

        event.accept()

    def resizeEvent(self, event):
        """Evento de redimensionamento"""
        super().resizeEvent(event)
        if not self.ui_initialized:
            return

        if self.current_image_path and self.image_label.pixmap():
            self._update_displayed_image()
