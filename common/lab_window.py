# -*- coding: utf-8 -*-
"""Shared shell for laboratory apps.

New apps should build one fixed square canvas and install it into a
LabWindow; this keeps window size, background and save behaviour aligned.
"""
from common.qt_base import QMainWindow, QWidget


class SquareCanvas(QWidget):
    def __init__(self, parent=None, size=600):
        super().__init__(parent)
        self.setFixedSize(int(size), int(size))

    def save_image(self, path):
        return self.grab().save(path)


class LabWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #ffffff;")
        self.setMinimumSize(960, 640)
        self.resize(960, 640)
