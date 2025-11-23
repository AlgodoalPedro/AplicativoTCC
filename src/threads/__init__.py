"""
Módulo de threads para processamento YOLO
"""

from .yolo_thread import YOLOThread
from .webcam_thread import WebcamThread

__all__ = ['YOLOThread', 'WebcamThread']
