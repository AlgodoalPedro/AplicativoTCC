"""
Utilitários para manipulação de imagens
"""
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


def display_image_scaled(image_label, image_path):
    """
    Exibe uma imagem no QLabel com redimensionamento dinâmico

    Args:
        image_label: QLabel onde a imagem será exibida
        image_path: Caminho da imagem
    """
    # Remove layout anterior se existir
    if image_label.layout():
        QWidget().setLayout(image_label.layout())

    # Calcula o tamanho disponível
    available_width = image_label.width() - 40
    available_height = image_label.height() - 40

    if available_width <= 0 or available_height <= 0:
        available_width = 800
        available_height = 600

    pix = QPixmap(image_path).scaled(
        available_width,
        available_height,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    image_label.setPixmap(pix)


def create_placeholder(icon="🖼", main_text="Nenhuma imagem carregada",
                      sub_text="Selecione uma imagem ou vídeo para começar"):
    """
    Cria um widget de placeholder para exibição quando não há imagem

    Args:
        icon: Emoji ou ícone para exibir
        main_text: Texto principal
        sub_text: Texto secundário

    Returns:
        QVBoxLayout: Layout contendo o placeholder
    """
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

    icon_placeholder = QLabel(icon)
    icon_placeholder.setAlignment(Qt.AlignCenter)
    icon_placeholder.setStyleSheet("font-size: 48px;")

    text_placeholder = QLabel(main_text)
    text_placeholder.setAlignment(Qt.AlignCenter)
    text_placeholder.setStyleSheet("color: #111827; font-size: 14px; font-weight: 600;")

    subtext_placeholder = QLabel(sub_text)
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

    return main_placeholder_layout


def create_custom_placeholder(icon, main_text, sub_text):
    """
    Cria um placeholder customizado com texto específico

    Args:
        icon: Emoji ou ícone
        main_text: Texto principal
        sub_text: Texto secundário

    Returns:
        QVBoxLayout: Layout do placeholder
    """
    placeholder_container = QFrame()
    placeholder_container.setStyleSheet("""
        QFrame {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 40px;
        }
    """)

    placeholder_layout = QVBoxLayout(placeholder_container)
    placeholder_layout.setAlignment(Qt.AlignCenter)
    placeholder_layout.setSpacing(15)
    placeholder_layout.setContentsMargins(40, 40, 40, 40)

    icon_label = QLabel(icon)
    icon_label.setAlignment(Qt.AlignCenter)
    icon_label.setStyleSheet("font-size: 52px; color: #10b981; background: transparent; border: none;")

    text = QLabel(main_text)
    text.setAlignment(Qt.AlignCenter)
    text.setStyleSheet("color: #111827; font-size: 16px; font-weight: 600; background: transparent; border: none; margin-top: 10px;")

    subtext = QLabel(sub_text)
    subtext.setAlignment(Qt.AlignCenter)
    subtext.setStyleSheet("color: #6b7280; font-size: 13px; background: transparent; border: none; line-height: 1.5;")
    subtext.setWordWrap(True)

    placeholder_layout.addWidget(icon_label)
    placeholder_layout.addWidget(text)
    placeholder_layout.addWidget(subtext)

    main_placeholder_layout = QVBoxLayout()
    main_placeholder_layout.setAlignment(Qt.AlignCenter)
    main_placeholder_layout.setContentsMargins(20, 20, 20, 20)
    main_placeholder_layout.addWidget(placeholder_container)

    return main_placeholder_layout
