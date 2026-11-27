"""Modal arbitrary-precision deep-zoom renderer for the Mandelbrot app."""

from __future__ import annotations

import os

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from support.fractals.deep_zoom import DeepZoomRequest, render_deep


class _DeepZoomThread(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(bytes, int, int)
    failed = Signal(str)

    def __init__(self, request: DeepZoomRequest, parent=None):
        super().__init__(parent)
        self.request = request
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            data = render_deep(
                self.request,
                progress=lambda done, total: self.progress.emit(done, total),
                cancelled=lambda: self._cancel,
            )
            if self._cancel:
                return
            self.finished_ok.emit(bytes(data), self.request.size, self.request.size)
        except Exception as exc:  # pragma: no cover - shown in the dialog
            self.failed.emit(str(exc))


class DeepZoomDialog(QDialog):
    """Render a square view with mpmath precision, independent of the GPU path."""

    def __init__(self, center_x: float, center_y: float, scale: float,
                 max_iter: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("深度缩放 — 任意精度 Mandelbrot")
        self.setMinimumSize(760, 720)
        self._thread = None
        self._image = None
        self._scale = float(scale)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        hint = QLabel(
            "GPU 画布使用 f64，约 10¹⁴ 倍后失真；这里用 Decimal 任意精度逐像素计算。"
            "支持 z² + c。分辨率越高、精度/迭代越高越慢。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        self.cx_edit = QLineEdit(f"{center_x:.17g}")
        self.cy_edit = QLineEdit(f"{center_y:.17g}")
        self.scale_edit = QLineEdit(f"{scale:.17g}")
        self.scale_edit.setToolTip("视野半高，例如 1e-30；数值越小放得越大")
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(20, 300)
        digits = 0.0
        if abs(scale) > 0:
            import math as _math
            digits = max(0.0, -_math.log10(abs(scale)))
        self.precision_spin.setValue(min(300, max(40, int(digits) + 20)))
        self.precision_spin.setSuffix(" 位")
        self.precision_spin.setToolTip("建议至少比放大位数多 10 位")
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 20000)
        self.iter_spin.setValue(min(20000, max(250, int(250 + 220 * digits))))
        self.iter_spin.setToolTip("深度越大，需要的迭代通常越高")
        self.size_combo = QComboBox()
        for size in (120, 160, 200, 240, 320):
            self.size_combo.addItem(f"{size} × {size}", size)
        self.size_combo.setCurrentIndex(2)
        form.addRow("中心实部:", self.cx_edit)
        form.addRow("中心虚部:", self.cy_edit)
        form.addRow("视野半高:", self.scale_edit)
        form.addRow("精度:", self.precision_spin)
        form.addRow("迭代:", self.iter_spin)
        form.addRow("分辨率:", self.size_combo)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.render_btn = QPushButton("开始渲染")
        self.render_btn.clicked.connect(self.start_render)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_render)
        self.save_btn = QPushButton("保存 PNG")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_png)
        buttons.addWidget(self.render_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.image_label = QLabel("渲染结果会显示在这里")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet("background:#111; color:#aaa; border:1px solid #333;")
        layout.addWidget(self.image_label, 1)

        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#666;")
        layout.addWidget(self.status)

    def _request(self) -> DeepZoomRequest | None:
        try:
            cx = self.cx_edit.text().strip()
            cy = self.cy_edit.text().strip()
            scale_text = self.scale_edit.text().strip()
            float(cx), float(cy), float(scale_text)  # validate numeric syntax
            scale = abs(float(scale_text))
            if scale <= 0:
                raise ValueError("视野必须大于 0")
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", f"中心/视野不是有效数字：\n{exc}")
            return None
        return DeepZoomRequest(
            center_x=cx,
            center_y=cy,
            scale=scale_text,
            size=int(self.size_combo.currentData()),
            max_iter=int(self.iter_spin.value()),
            precision=int(self.precision_spin.value()),
        )

    def start_render(self):
        if self._thread is not None and self._thread.isRunning():
            return
        request = self._request()
        if request is None:
            return
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("渲染中… 深度缩放是 CPU 计算，请耐心等待。")
        self._thread = _DeepZoomThread(request, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def cancel_render(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self.status.setText("正在取消…")

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(int(100 * done / max(1, total)))

    def _on_finished(self, data: bytes, width: int, height: int):
        image = QImage(data, width, height, width * 3, QImage.Format_RGB888).copy()
        self._image = image
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        self.progress.setValue(100)
        self.status.setText(f"完成：{width} × {height} 像素。可保存 PNG。")
        self.save_btn.setEnabled(True)

    def _on_failed(self, message: str):
        self.status.setText("渲染失败")
        QMessageBox.critical(self, "渲染失败", message)

    def _on_thread_finished(self):
        self.render_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def save_png(self):
        if self._image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存深度缩放图像", os.path.expanduser("~/mandelbrot_deep.png"),
            "PNG (*.png)")
        if path:
            self._image.save(path, "PNG")
            self.status.setText(f"已保存：{path}")

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(1500)
        super().closeEvent(event)
