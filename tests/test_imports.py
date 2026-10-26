"""冒烟测试：验证所有 app 模块能够被正常导入（无循环导入、无语法错误）"""
import sys
import os
import unittest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))

from PySide6.QtWidgets import QWidget


class TestAppImports(unittest.TestCase):
    """测试各 app 文件的导入是否正常"""

    def test_split_import_bases(self):
        """qt/numpy/taichi bases must be independently importable."""
        from common.qt_base import QApplication, QMainWindow, QTimer, OrbitCamera, ColorButton
        from common.numpy_base import np
        from common.taichi_base import ti
        from common.numba_base import jit, njit
        from common.mp_base import mp
        from common.custom_base import c64, csqr, cmul, BaseFractalWidget
        from common.reaction_diffusion import SimulationBase, SimulationViewer

        self.assertIsNotNone(np)
        self.assertIsNotNone(ti)
        self.assertIsNotNone(QApplication)
        self.assertTrue(issubclass(SimulationViewer, QWidget))
        self.assertTrue(hasattr(SimulationBase, "step"))

    def test_utils_import(self):
        """utils.py 应独立可导入"""
        from common import utils
        self.assertTrue(hasattr(utils, 'OrbitCamera'))
        self.assertTrue(hasattr(utils, 'ColorButton'))

    def test_custom_function_import(self):
        """custom_function.py 应独立可导入"""
        from common import custom_function
        self.assertTrue(hasattr(custom_function, 'c64'))
        self.assertTrue(hasattr(custom_function, 'csqr'))
        self.assertTrue(hasattr(custom_function, 'integrate_custom_python'))

    def test_reaction_diffusion_import(self):
        """reaction_diffusion.py 应独立可导入"""
        from common import reaction_diffusion
        self.assertTrue(hasattr(reaction_diffusion, 'SimulationBase'))
        self.assertTrue(hasattr(reaction_diffusion, 'SimulationViewer'))

    def test_dynamic_fractal_import(self):
        """dynamic_fractal.py 应独立可导入"""
        from common import dynamic_fractal
        self.assertTrue(hasattr(dynamic_fractal, 'BaseFractalWidget'))

    def test_no_circular_imports(self):
        """验证多个模块按顺序导入不会产生循环导入错误"""
        modules = ['common.utils', 'common.custom_function', 'common.reaction_diffusion',
                   'common.dynamic_fractal', 'common.qt_base', 'common.numpy_base', 'common.taichi_base']
        for mod_name in modules:
            # 从 sys.modules 中移除以便重新导入
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        imported = []
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=['*'])
            imported.append(mod)
        self.assertEqual(len(imported), 7)


if __name__ == '__main__':
    unittest.main()
