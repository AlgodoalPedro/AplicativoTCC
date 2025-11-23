"""
FEI Vision Studio - Aplicativo de Detecção de Objetos com YOLO

Ponto de entrada principal da aplicação
"""
import sys
from PyQt5.QtWidgets import QApplication
from src.ui import YOLOApp


def main():
    """Função principal para iniciar a aplicação"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = YOLOApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
