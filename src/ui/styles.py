"""
Estilos CSS para a interface do aplicativo
"""

GLOBAL_STYLE = """
    QWidget {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
"""

SIDEBAR_STYLE = """
    QFrame {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
"""

IMAGE_CONTAINER_STYLE = """
    QFrame {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
    }
"""

RESULT_CONTAINER_STYLE = """
    QFrame {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
    }
"""

PROGRESS_BAR_STYLE = """
    QProgressBar {
        background-color: #e5e7eb;
        border: none;
        border-radius: 3px;
    }
    QProgressBar::chunk {
        background-color: #3b82f6;
        border-radius: 3px;
    }
"""


def get_combo_box_style(scale=1.0):
    """Retorna o estilo do ComboBox com escala"""
    button_size = int(13 * scale)
    combo_padding = int(10 * scale)
    border_radius = int(8 * scale)

    return f"""
        QComboBox {{
            background-color: #f9fafb;
            border: 1px solid #d1d5db;
            border-radius: {border_radius}px;
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
    """


def get_radio_button_style(scale=1.0):
    """Retorna o estilo dos radio buttons com escala"""
    button_size = int(13 * scale)

    return f"""
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


def get_primary_button_style(color="#3b82f6", scale=1.0):
    """Retorna o estilo de botão primário com escala"""
    button_size = int(13 * scale)
    btn_padding_v = int(11 * scale)
    btn_padding_h = int(16 * scale)

    return f"""
        QPushButton {{
            background-color: {color};
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
    """


def get_secondary_button_style(scale=1.0):
    """Retorna o estilo de botão secundário com escala"""
    button_size = int(13 * scale)
    btn_padding_v = int(11 * scale)
    btn_padding_h = int(16 * scale)

    return f"""
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
    """


def get_action_button_style(is_detecting=False, scale=1.0):
    """Retorna o estilo do botão de ação com escala"""
    action_button_size = int(14 * scale)
    action_padding_v = int(14 * scale)
    action_padding_h = int(16 * scale)

    if is_detecting:
        bg_color = "#ef4444"
        hover_color = "#dc2626"
        pressed_color = "#b91c1c"
    else:
        bg_color = "#10b981"
        hover_color = "#059669"
        pressed_color = "#047857"

    return f"""
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
    """


def get_list_widget_style(scale=1.0):
    """Retorna o estilo da lista de detecções com escala"""
    button_size = int(13 * scale)

    return f"""
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
    """


def get_label_style(font_size, color="#374151", weight=600, scale=1.0):
    """Retorna o estilo de um label"""
    scaled_size = int(font_size * scale)
    return f"""
        color: {color};
        font-size: {scaled_size}px;
        font-weight: {weight};
        margin-top: 5px;
        margin-bottom: 8px;
    """
