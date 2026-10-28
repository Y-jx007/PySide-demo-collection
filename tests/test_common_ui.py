"""公共组件的行为测试：widgets / imaging / app / workers / utils。"""
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from common.app import STYLE_PATH, load_stylesheet, run_app, setup_gl_format  # noqa: E402
from common.imaging import (  # noqa: E402
    ImageLabel,
    ask_save_path,
    ndarray_to_qimage,
    ndarray_to_qpixmap,
)
from common.utils import FPSCounter  # noqa: E402
from common.widgets import (  # noqa: E402
    LabeledSlider,
    SquareContainer,
    caption,
    double_spinbox,
    form_group,
    group_box,
    hbox,
    separator,
    spinbox,
    vbox,
)
from common.workers import StoppableThread  # noqa: E402


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_labeled_slider_formats_and_binds(self):
        seen = []
        slider = LabeledSlider("大小", 1, 20, 5, formatter=lambda v: f"{v}px")
        slider.bind(seen.append)
        self.assertEqual(seen, [5])
        self.assertEqual(slider.value_label.text(), "5px")

        slider.setValue(12)
        self.assertEqual(slider.value(), 12)
        self.assertEqual(seen[-1], 12)
        self.assertEqual(slider.value_label.text(), "12px")

    def test_spinbox_defaults(self):
        ints = spinbox(1, 10, 3, 2)
        self.assertEqual((ints.value(), ints.minimum(), ints.maximum(), ints.singleStep()),
                         (3, 1, 10, 2))

        from PySide6.QtWidgets import QAbstractSpinBox

        floats = double_spinbox(0.0, 1.0, 0.25, 0.05, decimals=3, buttons=False)
        self.assertAlmostEqual(floats.value(), 0.25)
        self.assertEqual(floats.decimals(), 3)
        self.assertEqual(floats.buttonSymbols(), QAbstractSpinBox.NoButtons)

    def test_layout_helpers(self):
        row = hbox(separator(), None)
        self.assertEqual(row.count(), 2)          # 组件 + 弹簧
        column = vbox(QLabel("stub"), spacing=2)
        self.assertEqual(column.count(), 1)

    def test_group_helpers(self):
        box = group_box("标题", separator())
        self.assertEqual(box.title(), "标题")
        self.assertEqual(box.layout().count(), 1)

        group, form = form_group("参数")
        self.assertIs(group.layout(), form)

        wrapped = caption("标题", QWidget())
        self.assertEqual(wrapped.layout().count(), 2)

    def test_square_container_keeps_child_square(self):
        child = QWidget()
        container = SquareContainer(child)
        container.show()
        container.resize(400, 300)
        self.app.processEvents()
        self.assertEqual(child.width(), child.height())
        container.close()


class ImagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ndarray_to_qimage_shapes(self):
        rgb = ndarray_to_qimage(np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertEqual((rgb.width(), rgb.height()), (6, 4))

        rgba = ndarray_to_qimage(np.zeros((2, 3, 4), dtype=np.uint8))
        self.assertEqual((rgba.width(), rgba.height()), (3, 2))

        gray = ndarray_to_qimage(np.zeros((3, 5), dtype=np.uint8))
        self.assertEqual((gray.width(), gray.height()), (5, 3))

        with self.assertRaises(ValueError):
            ndarray_to_qimage(np.zeros((2, 2, 2), dtype=np.uint8))

    def test_ndarray_to_qpixmap(self):
        pixmap = ndarray_to_qpixmap(np.full((5, 5), 128, dtype=np.uint8))
        self.assertEqual((pixmap.width(), pixmap.height()), (5, 5))

    def test_image_label_shows_array(self):
        label = ImageLabel("空")
        self.assertTrue(label.set_array(np.zeros((16, 16, 3), dtype=np.uint8)))
        label.resize(200, 200)
        label.show()
        self.app.processEvents()
        self.assertFalse(label.pixmap().isNull())
        label.close()

    def test_image_label_reports_bad_shape(self):
        label = ImageLabel("空")
        self.assertFalse(label.set_array(np.zeros((2, 2, 2), dtype=np.uint8)))
        self.assertIn("显示错误", label.text())

    def test_save_path_helper_is_importable(self):
        self.assertTrue(callable(ask_save_path))


class AppBootstrapTests(unittest.TestCase):
    def test_stylesheet_path_and_loader(self):
        self.assertTrue(os.path.isfile(STYLE_PATH), STYLE_PATH)
        self.assertIn("QMainWindow", load_stylesheet())
        self.assertEqual(load_stylesheet(os.path.join(ROOT, "no_such.qss")), "")

    def test_bootstrap_helpers_are_callable(self):
        self.assertTrue(callable(run_app))
        self.assertTrue(callable(setup_gl_format))


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stop_flag(self):
        worker = StoppableThread()
        self.assertFalse(worker.stopping)
        worker.stop()
        self.assertTrue(worker.stopping)
        worker.reset_stop()
        self.assertFalse(worker.stopping)

    def test_stop_and_wait_on_idle_thread(self):
        worker = StoppableThread()
        self.assertTrue(worker.stop_and_wait(100))


class FpsCounterTests(unittest.TestCase):
    def test_counts_frames_per_window(self):
        counter = FPSCounter(interval=0.5)
        counter._since = 0.0            # 固定起点，避免依赖真实计时
        self.assertIsNone(counter.tick(now=0.2))
        self.assertAlmostEqual(counter.tick(now=0.5), 2 / 0.5)
        self.assertAlmostEqual(counter.fps, 4.0)

    def test_reset(self):
        counter = FPSCounter(interval=0.5)
        counter._since = 0.0
        counter.tick(now=1.0)
        counter.reset()
        self.assertEqual(counter.fps, 0.0)

if __name__ == "__main__":
    unittest.main()

class MplCanvasTests(unittest.TestCase):
    """common.mpl_canvas：画布尺寸、Gcf 登记、工具栏。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fixed_size_canvas(self):
        from common.mpl_canvas import MplCanvas

        canvas = MplCanvas(4, 3, 100, fixed_size=True)
        self.assertEqual((canvas.width(), canvas.height()), (400, 300))
        self.assertIsNotNone(canvas.figure.add_subplot(111))
        canvas.close()

    def test_register_and_unregister(self):
        from common.mpl_canvas import MplCanvas, unregister_figure_manager

        canvas = MplCanvas(2, 2, 100, register=True)
        self.assertIsNotNone(canvas.manager)
        unregister_figure_manager(canvas.manager)
        canvas.manager = None
        canvas.close()


class ArrayCanvasTests(unittest.TestCase):
    """common.field_canvas：等比绘制矩形、坐标映射、懒渲染。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_square_rect_and_mapping(self):
        from PySide6.QtCore import QPoint

        from common.field_canvas import ArrayCanvas

        canvas = ArrayCanvas(minimum=None)
        canvas.resize(400, 200)
        canvas.set_array(np.zeros((10, 10, 3), dtype=np.uint8))
        rect = canvas.image_rect()
        self.assertAlmostEqual(rect.width(), 200.0)
        self.assertAlmostEqual(rect.height(), 200.0)
        self.assertAlmostEqual(rect.x(), 100.0)
        self.assertEqual(canvas.map_pos(QPoint(200, 100)), (0.5, 0.5))
        self.assertIsNone(canvas.map_pos(QPoint(5, 5)))
        outside = canvas.map_normalized(QPoint(5, 5))
        self.assertLess(outside[0], 0.0)
        canvas.close()

    def test_renderer_runs_on_paint(self):
        from common.field_canvas import ArrayCanvas

        calls = []

        def render():
            calls.append(1)
            return np.full((8, 8, 3), 40, dtype=np.uint8)

        canvas = ArrayCanvas(minimum=None, renderer=render)
        canvas.resize(100, 100)
        canvas.show()
        self.app.processEvents()
        self.assertGreaterEqual(len(calls), 1)
        self.assertIsNotNone(canvas.image())
        canvas.close()

