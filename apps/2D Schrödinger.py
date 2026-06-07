from custom_import import *

# Taichi 初始化
ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=False, offline_cache=True)
# 使用结构体进行逻辑分组
PotentialParams = ti.types.struct(
    strength=ti.f32,
    preset=ti.i32,  # 0:双缝, 1:单缝, 2:势阱, 3:势垒, 4:自定义
    barrier_thickness=ti.f32,
    single_slit_width=ti.f32,
    double_slit_width=ti.f32,
    double_slit_separation=ti.f32,
    radius=ti.f32
)
WavePacketParams = ti.types.struct(
    momentum=ti.f32,
    width=ti.f32,
    center_x=ti.f32,
    center_y=ti.f32,
    direction=ti.f32
)
SimulationParams = ti.types.struct(
    time_step=ti.f32,
    steps_per_frame=ti.i32
)
# 创建参数字段
potential_params = PotentialParams.field(shape=())
wave_packet_params = WavePacketParams.field(shape=())
simulation_params = SimulationParams.field(shape=())
# 自定义势能场
resolution=(1080,540)
custom_potential_field = ti.field(dtype=ti.f32, shape=resolution)
# 自定义势能函数编译器
class CustomPotentialCompiler:
    def __init__(self):
        self.globals_dict = {
            'math': math,
            'np': np,
            'ti': ti,
            'tm': tm,
            'sin': math.sin,
            'cos': math.cos,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'abs': abs,
            'pi': math.pi,
            'e': math.e
        }
        
    def compile_potential(self, code_str, width, height):
        """编译自定义势能函数"""
        try:
            # 清除之前的变量
            for key in list(self.globals_dict.keys()):
                if key not in ['math', 'np', 'ti', 'tm', 'sin', 'cos', 'exp', 
                              'sqrt', 'abs', 'pi', 'e']:
                    del self.globals_dict[key]
            
            # 定义函数
            func_code = f"""
def potential_func(x, y, width={width}, height={height}):
    {code_str}
    return result
"""
            exec(func_code, self.globals_dict)
            
            # 计算势能场
            potential_func = self.globals_dict['potential_func']
            
            x_coords = np.linspace(0, width-1, width)
            y_coords = np.linspace(0, height-1, height)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            # 计算势能值
            potential_values = np.zeros((width, height))
            for i in range(width):
                for j in range(height):
                    try:
                        potential_values[i, j] = potential_func(X[i, j], Y[i, j])
                    except:
                        potential_values[i, j] = 0.0
            
            return potential_values, None  # 成功返回
            
        except Exception as e:
            return None, str(e)
# OKLAB转换矩阵
lms_matrix_inv = ti.Matrix([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480]
])
rgb_matrix_inv = ti.Matrix([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010]
])
# 参数设置
resolution = (1080, 540)
lut_size = 720
# 初始化势能编译器
potential_compiler = CustomPotentialCompiler()
# 场定义
psi = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k1 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k2 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k3 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
k4 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp1 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp2 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
y_temp3 = ti.Vector.field(2, dtype=ti.f32, shape=resolution)
potential_temp = ti.field(dtype=ti.f32, shape=resolution)
# 可视化输出
output = ti.Vector.field(4, dtype=ti.f32, shape=resolution)
# 色相查找表
hue_lut = ti.Vector.field(3, dtype=ti.f32, shape=lut_size)
# 初始化参数
@ti.kernel
def init_parameters():
    potential_params[None].strength = 1.0
    potential_params[None].preset = 0  # 默认双缝
    potential_params[None].barrier_thickness = 5.0
    potential_params[None].single_slit_width = 50.0
    potential_params[None].double_slit_width = 10.0
    potential_params[None].double_slit_separation = 50.0
    potential_params[None].radius = 50.0
    
    wave_packet_params[None].momentum = 250.0
    wave_packet_params[None].width = 0.5
    wave_packet_params[None].center_x = 0.3
    wave_packet_params[None].center_y = 0.5
    wave_packet_params[None].direction = 0.0
    
    simulation_params[None].time_step = 0.1
    simulation_params[None].steps_per_frame = 30
@ti.func
def potential(p):
    """势能函数"""
    result = 0.0
    preset = potential_params[None].preset
    
    if preset == 0:  # 双缝势垒
        barrier_center_x = resolution[0] / 2
        barrier_thickness = potential_params[None].barrier_thickness
        
        if ti.abs(p[0] - barrier_center_x) < barrier_thickness / 2:
            slit_separation = potential_params[None].double_slit_separation
            slit_width = potential_params[None].double_slit_width
            slit_center1 = resolution[1] / 2 - slit_separation / 2
            slit_center2 = resolution[1] / 2 + slit_separation / 2
            
            in_slit1 = ti.abs(p[1] - slit_center1) < slit_width / 2
            in_slit2 = ti.abs(p[1] - slit_center2) < slit_width / 2
            
            if not (in_slit1 or in_slit2):
                result = 1.0
                
    elif preset == 1:  # 单缝势垒
        barrier_center_x = resolution[0] / 2
        barrier_thickness = potential_params[None].barrier_thickness
        
        if ti.abs(p[0] - barrier_center_x) < barrier_thickness / 2:
            slit_center = resolution[1] / 2
            slit_width = potential_params[None].single_slit_width
            if ti.abs(p[1] - slit_center) > slit_width / 2:
                result = 1.0
                
    elif preset == 2:  # 势阱
        center = ti.Vector([resolution[0] / 2, resolution[1] / 2])
        rsqr = (p[0] - center[0])**2 + (p[1] - center[1])**2
        if rsqr < potential_params[None].radius**2:
            result = -1.0
            
    elif preset == 3:  # 势垒
        center = ti.Vector([resolution[0] / 2, resolution[1] / 2])
        rsqr = (p[0] - center[0])**2 + (p[1] - center[1])**2
        if rsqr < potential_params[None].radius**2:
            result = 1.0
    
    elif preset == 4:  # 自定义势能
        # 从自定义场读取势能值
        result = custom_potential_field[int(p[0]), int(p[1])]
    
    return result * potential_params[None].strength
@ti.func
def psi0(p):
    """初始波函数"""
    x = (p[0] - resolution[0] * wave_packet_params[None].center_x) / resolution[1]
    y = (p[1] - resolution[1] * wave_packet_params[None].center_y) / resolution[1]
    
    sigma = wave_packet_params[None].width * 0.1
    r2 = x*x + y*y
    gaussian = 0.1 * ti.exp(-r2 / (2.0 * sigma *sigma)) / (sigma * ti.sqrt(math.pi))
    
    angle = wave_packet_params[None].direction * math.pi / 180.0
    kx = wave_packet_params[None].momentum * ti.cos(angle)
    ky = wave_packet_params[None].momentum * ti.sin(angle)
    
    plane_wave = tm.cexp(tm.vec2(0,kx * x + ky * y))
    
    return gaussian * plane_wave
@ti.func
def compute_k(i, j, field, potential):
    laplacian = field[i+1, j] + field[i-1, j] + field[i, j+1] + field[i, j-1] - 4.0 * field[i, j]
    return tm.cdiv(-laplacian + potential[i,j] * field[i,j],tm.vec2(0,1))
# RK4计算函数
@ti.kernel
def update_psi():
    # 缓存势能 + 计算k1
    for i,j in ti.ndrange((1,resolution[0]-1),(1,resolution[1]-1)):
        p = ti.Vector([i, j])
        potential_temp[i,j] = potential(p)
        k1[i,j] = compute_k(i,j,psi,potential_temp)
        y_temp1[i,j] = psi[i,j] + 0.5 * simulation_params[None].time_step * k1[i,j]
    # 计算k2 
    for i,j in ti.ndrange((1,resolution[0]-1),(1,resolution[1]-1)):
        k2[i,j] = compute_k(i,j,y_temp1,potential_temp)
        y_temp2[i,j] = psi[i,j] + 0.5 * simulation_params[None].time_step * k2[i,j]
    
    # 计算k3
    for i,j in ti.ndrange((1,resolution[0]-1),(1,resolution[1]-1)):
        k3[i,j] = compute_k(i,j,y_temp2,potential_temp)
        y_temp3[i,j] = psi[i,j] + simulation_params[None].time_step * k3[i,j]
    
    # 计算k4 + 更新psi
    for i,j in ti.ndrange((1,resolution[0]-1),(1,resolution[1]-1)):
        k4[i,j] = compute_k(i,j,y_temp3,potential_temp)
        psi[i,j] = psi[i,j] + simulation_params[None].time_step * (k1[i,j] + 2.0 * k2[i,j] + 2.0 * k3[i,j] + k4[i,j]) / 6.0
# OKLAB 颜色转换函数
@ti.func
def srgb_from_linear_srgb(x):
    """将线性sRGB转换为sRGB"""
    xlo = 12.92 * x
    xhi = 1.055 * ti.pow(x, 1.0/2.4) - 0.055
    result = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        if x[k] <= 0.0031308:
            result[k] = xlo[k]
        else:
            result[k] = xhi[k]
    return result
@ti.func
def linear_srgb_from_oklab(c):
    """将OKLAB转换为线性sRGB"""
    lms_nonlinear = lms_matrix_inv @ c
    lms_cubed = ti.Vector([lms_nonlinear[0] * lms_nonlinear[0] * lms_nonlinear[0],
                          lms_nonlinear[1] * lms_nonlinear[1] * lms_nonlinear[1],
                          lms_nonlinear[2] * lms_nonlinear[2] * lms_nonlinear[2]])
    return rgb_matrix_inv @ lms_cubed
@ti.kernel
def init_hue_lut():
    # 生成色相转换为OKLAB RGB颜色表
    for i in range(lut_size):
        h = i / lut_size  # 0~1
        angle = h * 2 * math.pi
        L = 0.8
        a = 0.4 * ti.cos(angle)
        b = 0.4 * ti.sin(angle)
        oklab_color = ti.Vector([L, a, b])
        linear_rgb = linear_srgb_from_oklab(oklab_color)
        srgb_color = srgb_from_linear_srgb(linear_rgb)
        # 限制在0~1
        for k in ti.static(range(3)):
            if srgb_color[k] < 0.0:
                srgb_color[k] = 0.0
            elif srgb_color[k] > 1.0:
                srgb_color[k] = 1.0
        hue_lut[i] = srgb_color
@ti.kernel
def visualize():
    """可视化波函数"""
    for i, j in output:
        v = psi[i,j]
        magnitude = ti.sqrt(v[0] * v[0] + v[1] * v[1])
        phase = ti.atan2(v[1], v[0])
        
        hue = phase / (2.0 * math.pi)
        idx = int(hue * lut_size) % lut_size
        rgb = hue_lut[idx]
        
        pot = potential_temp[i,j]
        color = 1.5 * magnitude * rgb + 0.25 * pot
        
        output[i,j] = ti.Vector([color[0], color[1], color[2], 1.0])
@ti.kernel
def initialize():
    """初始化波函数"""
    for i, j in psi:
        p = ti.Vector([i, j])
        psi[i,j] = psi0(p)
@ti.kernel
def update_custom_potential_field(potential_array: ti.types.ndarray()):
    """更新自定义势能场"""
    for i, j in custom_potential_field:
        custom_potential_field[i, j] = potential_array[i, j]
class TaichiCanvas(QWidget):
    """用于显示Taichi模拟画面的QWidget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(resolution[0], resolution[1]))
        self.cached_image = None
        init_hue_lut()
        
    def update_simulation(self):
        """更新Taichi模拟并刷新显示"""
        steps = simulation_params[None].steps_per_frame
        for _ in range(steps):
            update_psi()
        
        visualize()
        self.update_image_cache()
        self.update()
    def update_image_cache(self):
        """更新图像缓存"""
        np_image = output.to_numpy()
        np_image = np.clip(np_image, 0.0, 1.0)
        np_image = (np_image * 255).astype(np.uint8)
        np_image = np.transpose(np_image, (1, 0, 2))
        
        height, width, channels = np_image.shape
        bytes_per_line = channels * width
        
        self.cached_image = QImage(
            np.ascontiguousarray(np_image).data, 
            width, height, bytes_per_line, 
            QImage.Format_RGBA8888
        ).copy()
    def paintEvent(self, event):
        if self.cached_image is not None:
            painter = QPainter(self)
            painter.drawImage(0, 0, self.cached_image)
            painter.end()
class ControlPanel(QWidget):
    """控制面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_running = True
        self.custom_potential_code = """result = 1 if ((x+33.75) % 135 < 67.5 and (y-33.75) % 135 > 67.5) \
or ((x+33.75) % 135 > 67.5 and (y-33.75) % 135 < 67.5) else -1"""
        
        self.init_ui()
        
    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(450)
        scroll_area.setMinimumWidth(380)
        
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        
        grid_layout = QGridLayout()
        
        # 势能参数 (左侧)
        row = 0
        grid_layout.addWidget(QLabel("势能类型:"), row, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["双缝势垒", "单缝势垒", "圆形势阱", "圆形势垒", "自定义势能"])
        self.preset_combo.setCurrentIndex(0)
        grid_layout.addWidget(self.preset_combo, row, 1)
        row += 1
        
        # 使用字典映射slider和label
        self.sliders = {}
        self.labels = {}
        
        # 势能参数
        grid_layout.addWidget(QLabel("势能强度:"), row, 0)
        self.sliders["strength"] = QSlider(Qt.Horizontal)
        self.sliders["strength"].setRange(0, 100)
        self.sliders["strength"].setValue(10)
        grid_layout.addWidget(self.sliders["strength"], row, 1)
        self.labels["strength"] = QLabel("1.0")
        grid_layout.addWidget(self.labels["strength"], row, 2)
        row += 1
        
        grid_layout.addWidget(QLabel("单缝宽度:"), row, 0)
        self.sliders["single_slit_width"] = QSlider(Qt.Horizontal)
        self.sliders["single_slit_width"].setRange(0, 100)
        self.sliders["single_slit_width"].setValue(50)
        grid_layout.addWidget(self.sliders["single_slit_width"], row, 1)
        self.labels["single_slit_width"] = QLabel("50")
        grid_layout.addWidget(self.labels["single_slit_width"], row, 2)
        row += 1
        
        grid_layout.addWidget(QLabel("双缝宽度:"), row, 0)
        self.sliders["double_slit_width"] = QSlider(Qt.Horizontal)
        self.sliders["double_slit_width"].setRange(0, 100)
        self.sliders["double_slit_width"].setValue(10)
        grid_layout.addWidget(self.sliders["double_slit_width"], row, 1)
        self.labels["double_slit_width"] = QLabel("10")
        grid_layout.addWidget(self.labels["double_slit_width"], row, 2)
        row += 1
        
        grid_layout.addWidget(QLabel("双缝间距:"), row, 0)
        self.sliders["double_slit_separation"] = QSlider(Qt.Horizontal)
        self.sliders["double_slit_separation"].setRange(0, 200)
        self.sliders["double_slit_separation"].setValue(50)
        grid_layout.addWidget(self.sliders["double_slit_separation"], row, 1)
        self.labels["double_slit_separation"] = QLabel("50")
        grid_layout.addWidget(self.labels["double_slit_separation"], row, 2)
        row += 1
        grid_layout.addWidget(QLabel("圆势半径:"), row, 0)
        self.sliders["radius"] = QSlider(Qt.Horizontal)
        self.sliders["radius"].setRange(0, 200)
        self.sliders["radius"].setValue(50)
        grid_layout.addWidget(self.sliders["radius"], row, 1)
        self.labels["radius"] = QLabel("50")
        grid_layout.addWidget(self.labels["radius"], row, 2)
        row += 1
        
        grid_layout.addWidget(QLabel("墙壁厚度:"), row, 0)
        self.sliders["barrier_thickness"] = QSlider(Qt.Horizontal)
        self.sliders["barrier_thickness"].setRange(1, 50)
        self.sliders["barrier_thickness"].setValue(5)
        grid_layout.addWidget(self.sliders["barrier_thickness"], row, 1)
        self.labels["barrier_thickness"] = QLabel("5")
        grid_layout.addWidget(self.labels["barrier_thickness"], row, 2)
        row += 1
        
        # 波包参数 (右侧)
        row = 1
        grid_layout.addWidget(QLabel("初始动量:"), row, 3)
        self.sliders["momentum"] = QSlider(Qt.Horizontal)
        self.sliders["momentum"].setRange(0, 1000)
        self.sliders["momentum"].setValue(250)
        grid_layout.addWidget(self.sliders["momentum"], row, 4)
        self.labels["momentum"] = QLabel("250.0")
        grid_layout.addWidget(self.labels["momentum"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("波包半径:"), row, 3)
        self.sliders["width"] = QSlider(Qt.Horizontal)
        self.sliders["width"].setRange(10, 200)
        self.sliders["width"].setValue(50)
        grid_layout.addWidget(self.sliders["width"], row, 4)
        self.labels["width"] = QLabel("0.50")
        grid_layout.addWidget(self.labels["width"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("中心X:"), row, 3)
        self.sliders["center_x"] = QSlider(Qt.Horizontal)
        self.sliders["center_x"].setRange(0, 100)
        self.sliders["center_x"].setValue(30)
        grid_layout.addWidget(self.sliders["center_x"], row, 4)
        self.labels["center_x"] = QLabel("0.30")
        grid_layout.addWidget(self.labels["center_x"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("中心Y:"), row, 3)
        self.sliders["center_y"] = QSlider(Qt.Horizontal)
        self.sliders["center_y"].setRange(0, 100)
        self.sliders["center_y"].setValue(50)
        grid_layout.addWidget(self.sliders["center_y"], row, 4)
        self.labels["center_y"] = QLabel("0.50")
        grid_layout.addWidget(self.labels["center_y"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("传播方向:"), row, 3)
        self.sliders["direction"] = QSlider(Qt.Horizontal)
        self.sliders["direction"].setRange(0, 360)
        self.sliders["direction"].setValue(0)
        grid_layout.addWidget(self.sliders["direction"], row, 4)
        self.labels["direction"] = QLabel("0°")
        grid_layout.addWidget(self.labels["direction"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("时间步长:"), row, 3)
        self.sliders["time_step"] = QSlider(Qt.Horizontal)
        self.sliders["time_step"].setRange(1, 30)
        self.sliders["time_step"].setValue(10)
        grid_layout.addWidget(self.sliders["time_step"], row, 4)
        self.labels["time_step"] = QLabel("0.1")
        grid_layout.addWidget(self.labels["time_step"], row, 5)
        row += 1
        
        grid_layout.addWidget(QLabel("每帧步数:"), row, 3)
        self.sliders["steps_per_frame"] = QSlider(Qt.Horizontal)
        self.sliders["steps_per_frame"].setRange(1, 180)
        self.sliders["steps_per_frame"].setValue(30)
        grid_layout.addWidget(self.sliders["steps_per_frame"], row, 4)
        self.labels["steps_per_frame"] = QLabel("30")
        grid_layout.addWidget(self.labels["steps_per_frame"], row, 5)
        row += 1
        
        main_layout.addLayout(grid_layout)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # 自定义势能函数编辑器
        custom_group = QGroupBox("自定义势能函数")
        custom_layout = QVBoxLayout(custom_group)
        
        # 说明标签
        help_label = QLabel("编辑自定义势能函数。必须设置变量 'result'。")
        help_label.setWordWrap(True)
        custom_layout.addWidget(help_label)
        
        # 代码编辑器
        self.code_editor = QTextEdit()
        self.code_editor.setPlainText(self.custom_potential_code)
        self.code_editor.setFont(QFont("Consolas", 10))
        self.code_editor.setMinimumHeight(100)
        custom_layout.addWidget(self.code_editor)
        
        # 应用和重置按钮
        button_layout = QHBoxLayout()
        self.apply_custom_btn = QPushButton("应用自定义势能")
        self.reset_custom_btn = QPushButton("重置为示例")
        button_layout.addWidget(self.apply_custom_btn)
        button_layout.addWidget(self.reset_custom_btn)
        custom_layout.addLayout(button_layout)
        
        main_layout.addWidget(custom_group)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line2)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_pause_btn = QPushButton("暂停")
        self.start_pause_btn.setCheckable(True)
        self.start_pause_btn.setChecked(True)
        button_layout.addWidget(self.start_pause_btn)
        
        self.reset_btn = QPushButton("重置波函数")
        button_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        
        content_widget.setLayout(main_layout)
        scroll_area.setWidget(content_widget)
        
        panel_layout = QVBoxLayout(self)
        panel_layout.addWidget(scroll_area)
        self.setLayout(panel_layout)
        
        # 连接信号
        self.connect_signals()
        
    def connect_signals(self):
        """连接所有信号"""
        # 使用通用更新函数绑定所有slider
        for param_name, slider in self.sliders.items():
            slider.valueChanged.connect(lambda value, pn=param_name: self.on_parameter_changed(pn, value))
        
        # 特殊控件单独连接
        self.preset_combo.currentIndexChanged.connect(self.update_preset)
        self.start_pause_btn.clicked.connect(self.toggle_simulation)
        self.reset_btn.clicked.connect(self.reset_simulation)
        
        # 自定义势能相关信号
        self.apply_custom_btn.clicked.connect(self.apply_custom_potential)
        self.reset_custom_btn.clicked.connect(self.reset_custom_code)
        
        # 初始化标签显示
        for param_name in self.sliders:
            self.update_label(param_name, self.sliders[param_name].value())
    def on_parameter_changed(self, param_name, value):
        """通用参数更新函数"""
        self.update_label(param_name, value)
        self.update_parameter(param_name, value)
    def update_label(self, param_name, value):
        """更新参数标签显示"""
        format_funcs = {
            "strength": lambda v: f"{v/10.0:.1f}",
            "momentum": lambda v: f"{float(v)}",
            "width": lambda v: f"{v/100.0:.2f}",
            "center_x": lambda v: f"{v/100.0:.2f}",
            "center_y": lambda v: f"{v/100.0:.2f}",
            "time_step": lambda v: f"{v/100.0:.2f}",
            "direction": lambda v: f"{v}°",
            "single_slit_width": lambda v: f"{v}",
            "double_slit_width": lambda v: f"{v}",
            "double_slit_separation": lambda v: f"{v}",
            "radius": lambda v: f"{v}",
            "barrier_thickness": lambda v: f"{v}",
            "steps_per_frame": lambda v: f"{v}"
        }
        
        if param_name in format_funcs:
            self.labels[param_name].setText(format_funcs[param_name](value))
    def update_parameter(self, param_name, value):
        """更新Taichi参数"""
        param_mapping = {
            "strength": (potential_params, "strength", lambda v: v / 10.0),
            "momentum": (wave_packet_params, "momentum", lambda v: float(v)),
            "width": (wave_packet_params, "width", lambda v: v / 100.0),
            "center_x": (wave_packet_params, "center_x", lambda v: v / 100.0),
            "center_y": (wave_packet_params, "center_y", lambda v: v / 100.0),
            "time_step": (simulation_params, "time_step", lambda v: v / 100.0),
            "direction": (wave_packet_params, "direction", lambda v: float(v)),
            "single_slit_width": (potential_params, "single_slit_width", lambda v: float(v)),
            "double_slit_width": (potential_params, "double_slit_width", lambda v: float(v)),
            "double_slit_separation": (potential_params, "double_slit_separation", lambda v: float(v)),
            "radius": (potential_params, "radius", lambda v: float(v)),
            "barrier_thickness": (potential_params, "barrier_thickness", lambda v: float(v)),
            "steps_per_frame": (simulation_params, "steps_per_frame", lambda v: v)
        }
        
        if param_name in param_mapping:
            field, field_name, convert_func = param_mapping[param_name]
            setattr(field[None], field_name, convert_func(value))
    def update_preset(self, index):
        """更新势能类型"""
        potential_params[None].preset = index
        # 如果切换到自定义势能，应用当前代码
        if index == 4:  # 自定义势能
            self.apply_custom_potential()
    def apply_custom_potential(self):
        """应用自定义势能函数"""
        code = self.code_editor.toPlainText()
        
        # 编译自定义势能函数
        potential_values, error = potential_compiler.compile_potential(
            code, resolution[0], resolution[1]
        )
        
        if error is not None:
            QMessageBox.warning(self, "编译错误", f"自定义势能函数编译失败:\n{error}")
            return
        
        # 更新自定义势能场
        update_custom_potential_field(potential_values)
        
        # 切换到自定义势能类型
        potential_params[None].preset = 4
        
        # 更新UI
        self.preset_combo.setCurrentIndex(4) 
    def reset_custom_code(self):
        """重置自定义代码为示例"""
        self.code_editor.setPlainText(self.custom_potential_code)
    def toggle_simulation(self):
        self.simulation_running = not self.simulation_running
        if self.simulation_running:
            self.start_pause_btn.setText("暂停")
        else:
            self.start_pause_btn.setText("开始")
    def reset_simulation(self):
        initialize()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("薛定谔方程模拟 - 自定义势能版")
        self.setGeometry(100, 100, 1200, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        self.control_panel = ControlPanel()
        self.taichi_canvas = TaichiCanvas()
        
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.taichi_canvas)
        
        main_layout.setStretchFactor(self.control_panel, 0)
        main_layout.setStretchFactor(self.taichi_canvas, 1)
        
        # 初始化参数和波函数
        init_parameters()
        initialize()
        
        # 初始化自定义势能场（默认全零）
        custom_potential_field.fill(0.0)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(0)
        
    def update_simulation(self):
        if self.control_panel.simulation_running:
            self.taichi_canvas.update_simulation()
            
    def closeEvent(self, event):
        self.timer.stop()  # 停止 Taichi kernel 调用
        ti.reset()
        event.accept()
def main():
    app = QApplication(sys.argv)
    
    font = app.font()
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
if __name__ == "__main__":
    main()

