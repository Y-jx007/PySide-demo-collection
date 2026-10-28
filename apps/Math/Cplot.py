from common.qt_base import (QApplication, QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
                            QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QTextEdit,
                            QVBoxLayout, QWidget, Qt, double_spinbox, run_app, spinbox)
from common.numpy_base import (np)
from common.mp_base import (mp)
from common.custom_base import (vectorize_func)
from common.mpl_canvas import MplCanvas, unregister_figure_manager
from support.math.cplot_parallel import evaluate_grid, weierstrass_p, shutdown as shutdown_cplot_pool
from support.math.cplot_fast_contour import install_fast_contours
install_fast_contours()
mp.dps = 15  # 设置mpmath的精度

# Weierstrass ℘ 的预设代码放在模块级，避免类里散落魔法字符串
WEIERSTRASS_CODE = "weierstrass_p(z)"


class CPlotController(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("cplot 复数函数绘图器")
        self.setGeometry(100, 100, 1200, 550)
        self.setMinimumSize(1180, 530)

        # 预设函数配置 - 集中管理，方便添加新函数
        self.preset_functions = self.setup_preset_functions()

        self.setup_ui()
        self.setup_connections()
        self.set_default_values()
        self.init_functions()

        # 缓存上次绘图的函数/网格结果
        self._cached_function = None
        self._cached_function_str = ""
        self._plot_cache_key = None
        self._plot_cache_fz = None

    def init_functions(self):

        # vectorize_func imported from custom_function

        # 向量化函数
        self.functions = {
            'gamma': vectorize_func(mp.gamma),
            'zeta': vectorize_func(mp.zeta),
            'eta': vectorize_func(mp.eta),
            'sn': vectorize_func(lambda z,tau: mp.ellipfun('sn', z, tau)),
            'cn': vectorize_func(lambda z,tau: mp.ellipfun('cn', z, tau)),
            'dn': vectorize_func(lambda z,tau: mp.ellipfun('dn', z, tau)),
            'EllipticParameter': vectorize_func(lambda z: mp.mfrom(tau=z)),
            'Kleinj': vectorize_func(lambda z: mp.kleinj(tau=z)),
            'weierstrass_p': weierstrass_p,
            'wp': weierstrass_p,
        }
    def setup_preset_functions(self):
        """设置预设函数配置"""
        return [
            {"name": "自定义", "code": "", "description": "自定义函数"},
            {"name": "sin(z³)/z", "code": "sin(z**3) / z", "description": "正弦函数与立方"},
            {"name": "Γ(z) - Gamma函数", "code": "gamma(z)", "description": "Gamma函数"},
            {"name": "Weierstrass ℘函数", "code": WEIERSTRASS_CODE, "description": "Weierstrass ℘函数"},
            {"name": "Riemann ζ函数", "code": "zeta(z)", "description": "Riemann ζ函数"},
            {"name": "Dedekind η函数 (y>=0)", "code": "eta(z)", "description": "Dedekind η函数"},
            {"name": "雅可比椭圆函数 sn(z)", "code": "sn(z, 0.5)", "description": "雅可比椭圆正弦函数"},
            {"name": "雅可比椭圆函数 cn(z)", "code": "cn(z, 0.5)", "description": "雅可比椭圆余弦函数"},
            {"name": "雅可比椭圆函数 dn(z)", "code": "dn(z, 0.5)", "description": "雅可比椭圆正切函数"},
            {"name": "模函数 λ(τ) (y>=0)", "code": "EllipticParameter(z)", "description": "模函数 λ(τ)"},
            {"name": "模形式 j(τ) (y>=0)", "code": "Kleinj(z)", "description": "模形式j(τ)"}
        ]

    def setup_ui(self):
        """设置用户界面"""
        # 不再调用 setup_window_style，使用系统默认风格

        # 中央部件：左控制面板 + 右嵌入绘图画布
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 16, 10)

        # 左侧滚动控制区
        scroll_area = self.setup_scroll_area()
        scroll_area.setMinimumWidth(500)
        scroll_area.setMaximumWidth(520)
        main_layout.addWidget(scroll_area, 0)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        self.setup_function_group(layout)
        self.setup_range_group(layout)
        self.setup_contour_display_group(layout)
        self.setup_status_label(layout)

        # 右侧 Matplotlib 画布
        self.setup_plot_panel(main_layout)

    def setup_plot_panel(self, parent_layout):
        panel = QWidget()
        panel.setObjectName("plotPanel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setStyleSheet("#plotPanel { background-color: white; border: 1px solid #d0d0d0; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 4, 8, 4)
        panel_layout.setSpacing(2)
        # register=True 会把画布登记进 pyplot 的 Gcf：
        # 没有它，plt.sca()/plt.gca() 会另建一块画布，工具栏的缩放/平移也会失灵。
        self.canvas = MplCanvas(6.4, 4.8, 100, facecolor="white", fixed_size=True,
                                register=True, toolbar=True, toolbar_parent=panel)
        self.figure = self.canvas.figure
        self.toolbar = self.canvas.toolbar
        panel.setFixedWidth(660)
        panel_layout.addWidget(self.toolbar)
        panel_layout.addWidget(self.canvas, 0, Qt.AlignHCenter)
        panel_layout.setAlignment(Qt.AlignTop)
        parent_layout.addWidget(panel, 0)

    def setup_scroll_area(self):
        """设置滚动区域"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll_area

    def setup_function_group(self, layout):
        """设置函数定义组"""
        function_group = QGroupBox("函数定义")
        function_layout = QVBoxLayout(function_group)
        function_layout.setSpacing(8)

        # 第一行：预设函数和按钮
        top_row_layout = QHBoxLayout()

        # 预设函数下拉框
        self.setup_preset_combo(top_row_layout)

        # 添加弹性空间
        top_row_layout.addStretch()

        # 按钮区域
        self.setup_buttons(top_row_layout)

        function_layout.addLayout(top_row_layout)

        # 自定义函数输入框
        self.setup_function_input(function_layout)

        layout.addWidget(function_group, 1)

    def setup_preset_combo(self, layout):
        """设置预设函数下拉框"""
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设函数:"))

        self.function_combo = QComboBox()
        # 从预设函数配置中添加选项
        for func in self.preset_functions:
            self.function_combo.addItem(func["name"])

        preset_layout.addWidget(self.function_combo)
        layout.addLayout(preset_layout)

    def setup_function_input(self, layout):
        """设置函数输入框"""
        self.function_input = QTextEdit()
        self.function_input.setPlaceholderText(
            """输入复数函数，使用 z 作为变量。例如: sin(z**3) / z\n
            支持多行代码定义复杂函数，需命名为f(z):
            def f(z):
                if abs(z) < 1:
                    return z**2
                else:
                    return 1/z"""
        )
        self.function_input.setMinimumHeight(120)
        self.function_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.function_input)

    def setup_buttons(self, layout):
        """设置按钮"""
        self.defaults_button = QPushButton("恢复默认值")
        layout.addWidget(self.defaults_button)

        self.plot_button = QPushButton("绘制图形")
        layout.addWidget(self.plot_button)

    def setup_range_group(self, layout):
        """设置范围和基本参数组"""
        range_group = QGroupBox("绘图范围和基本参数")
        range_layout = QVBoxLayout(range_group)
        range_layout.setSpacing(6)

        # 范围设置
        self.setup_range_controls(range_layout)

        # 其他参数
        self.setup_other_parameters(range_layout)

        layout.addWidget(range_group)

    def setup_range_controls(self, layout):
        """设置范围控件"""
        ranges_row1 = QHBoxLayout()

        # 实部范围
        real_group = self.create_range_group("实部范围", "re")
        ranges_row1.addWidget(real_group)

        # 虚部范围
        imag_group = self.create_range_group("虚部范围", "im")
        ranges_row1.addWidget(imag_group)

        layout.addLayout(ranges_row1)

    def create_range_group(self, title, prefix):
        """创建范围设置组：从 / 到 / 点数，都是无按钮的紧凑输入框。"""
        group = QGroupBox(title)
        layout = QHBoxLayout(group)
        layout.setSpacing(4)

        min_spinbox = double_spinbox(-100, 100, 0.0, 0.1, width=50, buttons=False)
        max_spinbox = double_spinbox(-100, 100, 0.0, 0.1, width=50, buttons=False)
        points_spinbox = spinbox(10, 2000, 400, 10, width=50, buttons=False)
        setattr(self, f"{prefix}_min_spinbox", min_spinbox)
        setattr(self, f"{prefix}_max_spinbox", max_spinbox)
        setattr(self, f"{prefix}_points_spinbox", points_spinbox)

        for label, widget in (("从", min_spinbox), ("到", max_spinbox), ("点数", points_spinbox)):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        return group

    def setup_other_parameters(self, layout):
        """设置其他参数"""
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("饱和度:"))

        self.saturation_spinbox = double_spinbox(0.1, 5.0, 1.28, 0.1, width=50, buttons=False)
        params_row.addWidget(self.saturation_spinbox)

        params_row.addWidget(QLabel("绝对值缩放:"))
        self.abs_scaling_input = QLineEdit("lambda x: x/(x+1)")
        self.abs_scaling_input.setMinimumWidth(150)  # 缩短宽度
        params_row.addWidget(self.abs_scaling_input)

        params_row.addStretch()
        layout.addLayout(params_row)

    def setup_contour_display_group(self, layout):
        """设置轮廓和显示选项组合并的组"""
        contour_display_group = QGroupBox("轮廓和显示设置")
        contour_display_layout = QVBoxLayout(contour_display_group)
        contour_display_layout.setSpacing(6)

        # 轮廓值
        self.setup_contour_values(contour_display_layout)

        # 轮廓选项和显示选项合并到同一行
        self.setup_contour_display_options(contour_display_layout)

        layout.addWidget(contour_display_group)

    def setup_contour_values(self, layout):
        """设置轮廓值"""
        contours_row1 = QHBoxLayout()
        contours_row1.addWidget(QLabel("|f(z)|="))

        self.contours_abs_input = QLineEdit("2.0")
        self.contours_abs_input.setMinimumWidth(20)  # 缩短宽度
        contours_row1.addWidget(self.contours_abs_input)

        contours_row1.addWidget(QLabel("arg(f(z))="))
        self.contours_arg_input = QLineEdit("-pi/2,0,pi/2,pi")
        self.contours_arg_input.setMinimumWidth(100)  # 缩短宽度
        contours_row1.addWidget(self.contours_arg_input)
        contours_row1.addWidget(QLabel("最小轮廓长度:"))
        self.min_contour_length_input = QLineEdit("1000")
        self.min_contour_length_input.setMinimumWidth(50)  # 缩短宽度
        contours_row1.addWidget(self.min_contour_length_input)

        contours_row1.addWidget(QLabel("线宽:"))
        self.linewidth_input = QLineEdit("None")
        self.linewidth_input.setMinimumWidth(50)  # 缩短宽度
        contours_row1.addWidget(self.linewidth_input)

        contours_row1.addStretch()
        layout.addLayout(contours_row1)

    def setup_contour_display_options(self, layout):
        """设置轮廓和显示选项在同一行"""
        options_row = QHBoxLayout()

        # 显示选项
        self.emphasize_abs_contour_1_check = QCheckBox("强调|f(z)|=1")
        options_row.addWidget(self.emphasize_abs_contour_1_check)
        self.add_colorbars_check = QCheckBox("显示颜色条")
        self.add_colorbars_check.setChecked(True)
        options_row.addWidget(self.add_colorbars_check)

        self.add_axes_labels_check = QCheckBox("显示坐标轴标签")
        self.add_axes_labels_check.setChecked(True)
        options_row.addWidget(self.add_axes_labels_check)

        options_row.addStretch()
        layout.addLayout(options_row)

    def setup_status_label(self, layout):
        """设置状态标签"""
        self.status_label = QLabel("准备就绪")
        layout.addWidget(self.status_label)

    def setup_connections(self):
        """设置信号连接"""
        self.function_combo.currentTextChanged.connect(self.on_function_changed)
        self.defaults_button.clicked.connect(self.set_default_values)
        self.plot_button.clicked.connect(self.plot_function)

    def on_function_changed(self, text):
        """当预设函数选择改变时更新函数输入框"""
        if text == "自定义":
            return

        # 从预设函数配置中查找对应的代码
        for func in self.preset_functions:
            if func["name"] == text:
                self.function_input.setPlainText(func["code"])
                break

    def set_default_values(self):
        """设置默认参数值"""
        # 范围设置
        self.re_min_spinbox.setValue(-10.0)
        self.re_max_spinbox.setValue(10.0)
        self.re_points_spinbox.setValue(400)

        self.im_min_spinbox.setValue(-10.0)
        self.im_max_spinbox.setValue(10.0)
        self.im_points_spinbox.setValue(400)

        # 其他参数
        self.saturation_spinbox.setValue(1.28)

        # 复选框
        self.emphasize_abs_contour_1_check.setChecked(True)
        self.add_colorbars_check.setChecked(True)
        self.add_axes_labels_check.setChecked(True)

        # 文本输入
        self.function_combo.setCurrentText("sin(z³)/z")
        self.function_input.setPlainText("sin(z**3) / z")
        self.abs_scaling_input.setText("lambda x: x/(x+1)")
        self.contours_abs_input.setText("2.0")
        self.contours_arg_input.setText("-pi/2,0,pi/2,pi")
        self.min_contour_length_input.setText("1000")
        self.linewidth_input.setText("None")

        self.status_label.setText("已恢复默认值")

    def parse_contours_arg(self, arg_str):
        """解析相位轮廓字符串"""
        try:
            if arg_str.strip().lower() == 'none':
                return None
            contours = eval(f"[{arg_str}]", {"pi": np.pi, "np": np})
            return tuple(contours)
        except Exception:
            return None

    def parse_value(self, value_str, default=None):
        """解析数值字符串，处理None情况"""
        try:
            if value_str.strip().lower() == 'none':
                return None
            return eval(value_str, {"pi": np.pi, "np": np})
        except Exception:
            return default

    def create_function(self, func_str):
        """创建可执行的函数对象，支持多行代码定义复杂函数"""
        if func_str == self._cached_function_str and self._cached_function is not None:
            return self._cached_function

        # 准备函数环境
        math_env = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e,
            'z': None,
            'abs': np.abs, 'real': np.real, 'imag': np.imag,
            'conj': np.conj, 'angle': np.angle
        }

        # 添加mp函数
        math_env.update(self.functions)

        # 检查是否是多行代码
        if '\n' in func_str or 'def ' in func_str or 'lambda ' in func_str:
            try:
                exec(func_str, math_env)

                if 'f' in math_env and callable(math_env['f']):
                    f = math_env['f']
                else:
                    lines = [line.strip() for line in func_str.split('\n') if line.strip()]
                    last_line = lines[-1] if lines else ""

                    if not (last_line.startswith(('if ', 'for ', 'while ', 'def ', 'class ', 'import ', 'from '))):
                        def f(z):
                            math_env['z'] = z
                            return eval(last_line, math_env)
                    else:
                        raise ValueError("多行代码必须定义一个名为f的函数，或最后一行是有效的表达式")
            except Exception as e:
                raise ValueError(f"无法解析多行函数定义: {str(e)}")
        else:
            def f(z):
                math_env['z'] = z
                return eval(func_str, math_env)

        # 缓存函数
        self._cached_function = f
        self._cached_function_str = func_str

        return f

    def plot_function(self):
        from cplot._main import Plotter
        import matplotlib.pyplot as plt
        """执行绘图函数（函数值并行采样，嵌入右侧画布）"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            plt.sca(ax)

            self.status_label.setText("正在并行计算网格...")
            QApplication.processEvents()

            # 获取函数定义
            func_str = self.function_input.toPlainText().strip()
            if not func_str:
                func_str = "sin(z**3) / z"

            # 准备绘图参数
            x_range = (self.re_min_spinbox.value(),
                      self.re_max_spinbox.value(),
                      self.re_points_spinbox.value())

            y_range = (self.im_min_spinbox.value(),
                      self.im_max_spinbox.value(),
                      self.im_points_spinbox.value())

            # 准备可选关键字参数
            kwargs = {}

            # 绝对值缩放函数
            abs_scaling_str = self.abs_scaling_input.text().strip()
            if abs_scaling_str and abs_scaling_str.lower() != 'none':
                kwargs['abs_scaling'] = eval(abs_scaling_str, {"np": np})

            # 轮廓参数
            contours_abs = self.parse_value(self.contours_abs_input.text())
            if contours_abs is not None:
                kwargs['contours_abs'] = contours_abs

            contours_arg = self.parse_contours_arg(self.contours_arg_input.text())
            if contours_arg is not None:
                kwargs['contours_arg'] = contours_arg

            # 布尔参数
            kwargs['emphasize_abs_contour_1'] = self.emphasize_abs_contour_1_check.isChecked()
            kwargs['add_colorbars'] = self.add_colorbars_check.isChecked()
            kwargs['add_axes_labels'] = self.add_axes_labels_check.isChecked()

            # 其他参数
            kwargs['saturation_adjustment'] = self.saturation_spinbox.value()

            min_contour_length = self.parse_value(self.min_contour_length_input.text())
            if min_contour_length is not None:
                kwargs['min_contour_length'] = min_contour_length

            linewidth = self.parse_value(self.linewidth_input.text())
            if linewidth is not None:
                kwargs['linewidth'] = linewidth

            # 先用向量化/多进程算完函数值，再交给 cplot 的 Plotter。
            # 这样色彩方案、轮廓、colorbars 和坐标轴风格完全保持原样。
            plotter = Plotter(x_range, y_range)
            cache_key = (func_str, x_range, y_range)
            if self._plot_cache_key == cache_key and self._plot_cache_fz is not None:
                fz = self._plot_cache_fz
            else:
                fz = evaluate_grid(func_str, plotter.Z)
                self._plot_cache_key = cache_key
                self._plot_cache_fz = fz
            self.status_label.setText("正在绘制配色与轮廓...")
            QApplication.processEvents()
            plotter.plot(fz, **kwargs)
            self.canvas.draw_idle()

            self.status_label.setText("图形已显示")

        except Exception as e:
            error_msg = f"错误: {str(e)}"
            self.status_label.setText(error_msg)
            print(f"绘图错误: {e}")
            import traceback
            traceback.print_exc()
    def closeEvent(self, event):
        shutdown_cplot_pool()
        unregister_figure_manager(self.canvas.manager)
        self.canvas.manager = None
        super().closeEvent(event)


if __name__ == "__main__":
    raise SystemExit(run_app(CPlotController))
