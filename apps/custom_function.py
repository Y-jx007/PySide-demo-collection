from custom_import import *

def vectorize_func(func):
            """将mp函数向量化"""
            def vectorized_func(z_array,*args, **kwargs):
                if np.isscalar(z_array):
                    return complex(func(z_array,*args, **kwargs))
                
                result = np.empty(z_array.shape, dtype=complex)
                flat_z = z_array.flat
                flat_result = result.flat
                
                for i in range(len(flat_z)):
                    flat_result[i] = complex(func(flat_z[i],*args, **kwargs))
                
                return result
            return vectorized_func

class ColorButton(QWidget):
    """可点击的色块按钮，点击后弹出颜色选择对话框"""
    colorChanged = Signal(QColor)

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, self.width(), self.height(), self.color)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            color = QColorDialog.getColor(self.color, self, "选择颜色")
            if color.isValid():
                self.color = color
                self.update()
                self.colorChanged.emit(color)