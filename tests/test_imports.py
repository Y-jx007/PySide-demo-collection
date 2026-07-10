"""冒烟测试：验证所有 app 模块能够被正常导入（无循环导入、无语法错误）"""
import sys
import os
import unittest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))


class TestAppImports(unittest.TestCase):
    """测试各 app 文件的导入是否正常"""

    def test_custom_import_clean(self):
        """custom_import.py 应无循环导入错误"""
        # 清除可能存在的模块缓存
        for mod in list(sys.modules.keys()):
            if 'apps.' in mod or mod in sys.modules:
                if mod.startswith('apps.') or mod in ('apps',):
                    del sys.modules[mod]
        from custom_import import (
            np, ti, QApplication, QMainWindow, QTimer,
            jit, njit, mp, ABC, abstractmethod,
            OrbitCamera, ColorButton,
            c64, csqr, cmul,
            SimulationBase, BaseFractalWidget
        )
        # 验证导入成功
        self.assertIsNotNone(np)
        self.assertIsNotNone(ti)
        self.assertIsNotNone(QApplication)

    def test_utils_import(self):
        """utils.py 应独立可导入"""
        import utils
        self.assertTrue(hasattr(utils, 'OrbitCamera'))
        self.assertTrue(hasattr(utils, 'ColorButton'))

    def test_custom_function_import(self):
        """custom_function.py 应独立可导入"""
        import custom_function
        self.assertTrue(hasattr(custom_function, 'c64'))
        self.assertTrue(hasattr(custom_function, 'csqr'))
        self.assertTrue(hasattr(custom_function, 'integrate_custom_python'))

    def test_reaction_diffusion_import(self):
        """reaction_diffusion.py 应独立可导入"""
        import reaction_diffusion
        self.assertTrue(hasattr(reaction_diffusion, 'SimulationBase'))
        self.assertTrue(hasattr(reaction_diffusion, 'SimulationViewer'))

    def test_dynamic_fractal_import(self):
        """dynamic_fractal.py 应独立可导入"""
        import dynamic_fractal
        self.assertTrue(hasattr(dynamic_fractal, 'BaseFractalWidget'))

    def test_no_circular_imports(self):
        """验证多个模块按顺序导入不会产生循环导入错误"""
        modules = ['utils', 'custom_function', 'reaction_diffusion',
                   'dynamic_fractal', 'custom_import']
        for mod_name in modules:
            # 从 sys.modules 中移除以便重新导入
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        imported = []
        for mod_name in modules:
            mod = __import__(mod_name)
            imported.append(mod)
        self.assertEqual(len(imported), 5)


if __name__ == '__main__':
    unittest.main()
