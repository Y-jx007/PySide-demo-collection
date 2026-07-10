from custom_import import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm

# ================== 全局 Taichi 资源 ==================
MAX_PARTICLES = 5000
positions = None
velocities = None
types = None
forces = None
num_particles = None

init_particles_kernel = None
simulate_step = None

PRESET_FILE = "particle_life_presets.npz"


def init_taichi(max_particles=5000):
    global positions, velocities, types, forces, num_particles
    global init_particles_kernel, simulate_step, MAX_PARTICLES

    ti.reset()
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    MAX_PARTICLES = max_particles
    positions = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
    velocities = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
    types = ti.field(dtype=ti.i32, shape=MAX_PARTICLES)
    forces = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
    num_particles = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def init_particles_kernel(n: ti.i32, num_types: ti.i32):
        for i in range(n):
            positions[i] = ti.Vector([ti.random(), ti.random()])
            velocities[i] = ti.Vector([0.0, 0.0])
            types[i] = ti.random(ti.i32) % num_types
        num_particles[None] = n

    @ti.kernel
    def simulate_step(att_matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      r_min: ti.f32, r_max: ti.f32,
                      force_scale: ti.f32, damping: ti.f32,
                      dt: ti.f32, max_speed: ti.f32):
        n = num_particles[None]
        for i in range(n):
            forces[i] = ti.Vector([0.0, 0.0])
        for i, j in ti.ndrange(n, n):
            if i < j:
                pi = positions[i]
                pj = positions[j]
                dx = pj - pi
                dist = dx.norm()
                if dist < r_max and dist > 1e-6:
                    if dist < r_min:
                        force_mag = (r_min - dist) / r_min * 10.0
                    else:
                        u = (dist - r_min) / (r_max - r_min)
                        force_mag = u * 2.0 - 1.0
                    f = dx / dist * force_mag * att_matrix[types[i], types[j]] * force_scale
                    forces[i] += f
                    forces[j] -= f
        for i in range(n):
            vel = velocities[i] + forces[i] * dt
            vel *= (1.0 - damping)
            speed = vel.norm()
            if speed > max_speed:
                vel = vel / speed * max_speed
            velocities[i] = vel
            positions[i] += vel * dt
            for d in ti.static(range(2)):
                if positions[i][d] < 0.0:
                    positions[i][d] += 1.0
                if positions[i][d] > 1.0:
                    positions[i][d] -= 1.0


class ParticleCanvas(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas_size = 640
        self.setFixedSize(self.canvas_size, self.canvas_size)
        self.drawing = False
        self.brush_radius = 0.02
        self.current_type = 0
        self.type_colors = []
        self._display_img = QImage(self.canvas_size, self.canvas_size, QImage.Format_ARGB32)
        self._display_img.fill(Qt.black)

        self._draw_queue = []
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_draw_queue)

        self.update_display()

    def set_type_colors(self, colors):
        self.type_colors = colors

    def initializeGL(self):
        pass

    def paintGL(self):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._display_img)
        painter.end()

    def _make_display_image(self):
        img = QImage(self.canvas_size, self.canvas_size, QImage.Format_ARGB32)
        img.fill(Qt.black)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, False)

        n = num_particles[None]
        if n == 0:
            self._display_img = img
            return

        pos_np = positions.to_numpy()[:n]
        types_np = types.to_numpy()[:n]
        pixel_x = (pos_np[:, 0] * self.canvas_size).astype(int)
        pixel_y = (pos_np[:, 1] * self.canvas_size).astype(int)
        pixel_x = np.clip(pixel_x, 0, self.canvas_size - 1)
        pixel_y = np.clip(pixel_y, 0, self.canvas_size - 1)

        for t in range(len(self.type_colors)):
            mask = types_np == t
            if not mask.any():
                continue
            col = QColor(*self.type_colors[t])
            painter.setPen(col)
            painter.setBrush(col)
            for px, py in zip(pixel_x[mask], pixel_y[mask]):
                painter.drawPoint(px, py)
        painter.end()
        self._display_img = img

    def update_display(self):
        self._make_display_image()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
        elif event.button() == Qt.RightButton:
            self.drawing = True
            self.current_type = (self.current_type + 1) % len(self.type_colors)
        if self.drawing:
            self._enqueue_brush(event.pos())

    def mouseMoveEvent(self, event):
        if not self.drawing:
            return
        self._enqueue_brush(event.pos())

    def mouseReleaseEvent(self, event):
        self.drawing = False
        self._flush_timer.stop()
        self._flush_draw_queue()

    def _enqueue_brush(self, pos):
        nx = pos.x() / self.canvas_size
        ny = pos.y() / self.canvas_size
        num_new = 5
        for _ in range(num_new):
            angle = np.random.rand() * 2 * np.pi
            r = np.random.rand() * self.brush_radius
            x = np.clip(nx + r * np.cos(angle), 0.0, 1.0)
            y = np.clip(ny + r * np.sin(angle), 0.0, 1.0)
            self._draw_queue.append((x, y, self.current_type))
        if not self._flush_timer.isActive():
            self._flush_timer.start(10)

    def _flush_draw_queue(self):
        if not self._draw_queue:
            return
        n = num_particles[None]
        add_count = len(self._draw_queue)
        if n + add_count > MAX_PARTICLES:
            add_count = MAX_PARTICLES - n
            if add_count <= 0:
                self._draw_queue.clear()
                return
        pos_np = positions.to_numpy()
        vel_np = velocities.to_numpy()
        type_np = types.to_numpy()
        for i, (x, y, t) in enumerate(self._draw_queue[:add_count]):
            idx = n + i
            pos_np[idx, 0] = x
            pos_np[idx, 1] = y
            vel_np[idx, 0] = 0.0
            vel_np[idx, 1] = 0.0
            type_np[idx] = t
        positions.from_numpy(pos_np)
        velocities.from_numpy(vel_np)
        types.from_numpy(type_np)
        num_particles[None] = n + add_count
        self._draw_queue.clear()
        self.update_display()


class InteractionPreview(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(2.2, 2.2), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.axis('off')
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.setFixedSize(180, 180)

    def update_matrix(self, matrix):
        self.ax.clear()
        self.ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        self.ax.axis('off')
        self.draw()


class ParticleLifeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.running = False
        self.num_particle_input = 400
        self.num_types = 4
        self.r_min = 0.05
        self.r_max = 0.15
        self.force_scale = 1.0
        self.damping = 0.05
        self.dt = 0.1
        self.max_speed = 0.05

        self.att_matrix = np.random.uniform(-1, 1, (self.num_types, self.num_types)).astype(np.float32)

        self._batch_update = False
        self.sliders = {}
        self.inputs = {}

        init_taichi(MAX_PARTICLES)
        self.canvas = ParticleCanvas()
        self.init_ui()

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.run_simulation)

        self.generate_type_colors()
        self.reset_particles()
        self.canvas.update_display()
        self.interaction_preview.update_matrix(self.att_matrix)

    def generate_type_colors(self):
        cmap = cm.get_cmap('tab10', self.num_types)
        colors = []
        for i in range(self.num_types):
            rgba = cmap(i)
            colors.append((int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
        self.canvas.set_type_colors(colors)

    def init_ui(self):
        self.setWindowTitle("Particle Life 模拟器")
        self.setGeometry(100, 100, 1060, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 4, 4, 4)

        tool_bar = QHBoxLayout()
        for text, slot in [("开始", self.toggle_simulation),
                           ("重置", self.reset_particles),
                           ("清空", self.clear_particles)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tool_bar.addWidget(btn)
        tool_bar.addSpacing(10)
        for text, slot in [("保存预设", self.save_preset),
                           ("加载预设", self.load_preset),
                           ("重命名", self.rename_preset),
                           ("删除", self.delete_preset)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tool_bar.addWidget(btn)
        tool_bar.addStretch()
        main_layout.addLayout(tool_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(330)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(2, 2, 2, 2)

        particle_group = QGroupBox("粒子设置")
        pf = QFormLayout(particle_group)
        pf.setSpacing(4)
        pf.setContentsMargins(4, 4, 4, 4)

        # 修正：添加的是 row widget，而不是元组
        count_row = self._add_int_input_row("数量:", "num_particle", 10, 3000,
                                            self.num_particle_input, self.on_num_particle)
        pf.addRow(count_row[2])  # 提取 row widget

        types_row = self._add_int_input_row("种类:", "num_types", 1, 8,
                                            self.num_types, self.on_types_change)
        pf.addRow(types_row[2])

        self.interaction_preview = InteractionPreview()
        pf.addRow("矩阵:", self.interaction_preview)

        physics_group = QGroupBox("物理参数")
        phf = QFormLayout(physics_group)
        phf.setSpacing(4)
        phf.setContentsMargins(4, 4, 4, 4)
        params = [
            ("r_min:", "r_min", 1, 100, 0.001, self.on_physics, self.r_min, 3),
            ("r_max:", "r_max", 1, 300, 0.001, self.on_physics, self.r_max, 3),
            ("力标度:", "force_scale", 1, 500, 0.01, self.on_physics, self.force_scale, 2),
            ("阻尼:", "damping", 0, 100, 0.001, self.on_physics, self.damping, 3),
            ("Δt:", "dt", 1, 200, 0.001, self.on_physics, self.dt, 3),
            ("最大速度:", "max_speed", 1, 200, 0.001, self.on_physics, self.max_speed, 3),
        ]
        for label, key, minv, maxv, scale, callback, init, prec in params:
            slider, input_widget, row = self._add_float_input_row(label, key, minv, maxv, init, callback, scale, prec)
            self.sliders[key] = slider
            self.inputs[key] = input_widget
            phf.addRow(row)

        left_layout.addWidget(particle_group)
        left_layout.addWidget(physics_group)

        hint = QLabel("<i>左键放置粒子，右键切换种类<br>拖动连续生成</i>")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        left_layout.addStretch()

        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([330, 640])
        main_layout.addWidget(splitter, 1)

    def _add_int_input_row(self, title, key, minv, maxv, init, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setFixedWidth(55)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minv, maxv)
        slider.setValue(init)
        slider.valueChanged.connect(lambda: self._on_int_slider(slider, key))
        input_widget = QLineEdit()
        input_widget.setFixedWidth(45)
        input_widget.setText(str(init))
        input_widget.editingFinished.connect(lambda: self._on_int_input(input_widget, slider, key))
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(input_widget)
        return slider, input_widget, row

    def _add_float_input_row(self, title, key, minv, maxv, init, callback, scale, prec):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setFixedWidth(55)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minv, maxv)
        slider.setValue(int(round(init / scale)))
        slider.valueChanged.connect(lambda: self._on_float_slider(slider, key, scale, prec))
        input_widget = QLineEdit()
        input_widget.setFixedWidth(55)
        input_widget.setText(f"{init:.{prec}f}")
        input_widget.editingFinished.connect(lambda: self._on_float_input(input_widget, slider, key, scale, prec))
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(input_widget)
        return slider, input_widget, row

    def _on_int_slider(self, slider, key):
        self.inputs[key].setText(str(slider.value()))
        getattr(self, f"on_{key}")()

    def _on_int_input(self, input_widget, slider, key):
        try:
            v = int(input_widget.text())
            v = max(slider.minimum(), min(slider.maximum(), v))
            slider.setValue(v)
        except:
            input_widget.setText(str(slider.value()))
        getattr(self, f"on_{key}")()

    def _on_float_slider(self, slider, key, scale, prec):
        if self.inputs[key].hasFocus():
            return
        val = slider.value() * scale
        self.inputs[key].setText(f"{val:.{prec}f}")
        self.on_physics()

    def _on_float_input(self, input_widget, slider, key, scale, prec):
        try:
            v = float(input_widget.text())
            slider.setValue(int(round(v / scale)))
        except:
            self._on_float_slider(slider, key, scale, prec)
        self.on_physics()

    def on_num_particle(self):
        pass  # 粒子数仅由重置按钮应用

    def on_types_change(self):
        new_types = self.sliders["num_types"].value()
        if new_types != self.num_types:
            self.num_types = new_types
            self.att_matrix = np.random.uniform(-1, 1, (self.num_types, self.num_types)).astype(np.float32)
            self.interaction_preview.update_matrix(self.att_matrix)
            self.generate_type_colors()
            self.canvas.set_type_colors(self.canvas.type_colors)

    def on_physics(self):
        if self._batch_update:
            return
        self.r_min = self.sliders["r_min"].value() * 0.001
        self.r_max = self.sliders["r_max"].value() * 0.001
        self.force_scale = self.sliders["force_scale"].value() * 0.01
        self.damping = self.sliders["damping"].value() * 0.001
        self.dt = self.sliders["dt"].value() * 0.001
        self.max_speed = self.sliders["max_speed"].value() * 0.001

    def toggle_simulation(self):
        if self.running:
            self.sim_timer.stop()
            self.sender().setText("开始")
            self.running = False
        else:
            self.sim_timer.start(16)
            self.sender().setText("暂停")
            self.running = True

    def reset_particles(self):
        n = self.sliders["num_particle"].value()
        self.num_particle_input = n
        init_particles_kernel(n, self.num_types)
        self.canvas.update_display()

    def clear_particles(self):
        num_particles[None] = 0
        self.canvas.update_display()

    def run_simulation(self):
        if num_particles[None] == 0:
            return
        simulate_step(self.att_matrix, self.r_min, self.r_max,
                      self.force_scale, self.damping, self.dt, self.max_speed)
        self.canvas.update_display()

    # ================== 预设管理 ==================
    def _get_preset_dict(self):
        presets = {}
        try:
            data = np.load(PRESET_FILE, allow_pickle=True)
        except FileNotFoundError:
            return presets
        for name in data.files:
            presets[name] = data[name].item()
        return presets

    def _save_preset_dict(self, preset_dict):
        np.savez(PRESET_FILE, **preset_dict)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "保存预设", "输入预设名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = self._get_preset_dict()
        if name in presets:
            reply = QMessageBox.question(self, "覆盖确认",
                                         f"预设 “{name}” 已存在，是否覆盖？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        n = num_particles[None]
        save_state = {
            'positions': positions.to_numpy()[:n].copy(),
            'velocities': velocities.to_numpy()[:n].copy(),
            'types': types.to_numpy()[:n].copy(),
            'n': n
        }
        save_params = {
            'num_types': self.num_types,
            'att_matrix': self.att_matrix,
            'r_min': self.r_min,
            'r_max': self.r_max,
            'force_scale': self.force_scale,
            'damping': self.damping,
            'dt': self.dt,
            'max_speed': self.max_speed,
            'num_particle': self.num_particle_input
        }
        presets[name] = {'state': save_state, 'params': save_params}
        self._save_preset_dict(presets)

    def load_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有找到任何预设。")
            return
        names = sorted(presets.keys())
        name, ok = QInputDialog.getItem(self, "加载预设", "选择预设：", names, 0, False)
        if not ok or not name:
            return
        data = presets[name]
        self._apply_preset(data)

    def _apply_preset(self, data):
        was_running = self.running
        if self.running:
            self.sim_timer.stop()

        params = data['params']
        self.num_types = params['num_types']
        self.att_matrix = params['att_matrix']
        self.r_min = params['r_min']
        self.r_max = params['r_max']
        self.force_scale = params['force_scale']
        self.damping = params['damping']
        self.dt = params['dt']
        self.max_speed = params['max_speed']
        self.num_particle_input = params['num_particle']

        self._batch_update = True
        self.sliders["num_types"].setValue(self.num_types)
        self.sliders["r_min"].setValue(int(self.r_min * 1000))
        self.sliders["r_max"].setValue(int(self.r_max * 1000))
        self.sliders["force_scale"].setValue(int(self.force_scale * 100))
        self.sliders["damping"].setValue(int(self.damping * 1000))
        self.sliders["dt"].setValue(int(self.dt * 1000))
        self.sliders["max_speed"].setValue(int(self.max_speed * 1000))
        self.sliders["num_particle"].setValue(self.num_particle_input)
        self._batch_update = False

        for key in self.inputs:
            self.inputs[key].setText(str(getattr(self, key, "")))

        self.generate_type_colors()
        self.canvas.set_type_colors(self.canvas.type_colors)
        self.interaction_preview.update_matrix(self.att_matrix)

        s = data['state']
        n = s['n']
        if n > MAX_PARTICLES:
            n = MAX_PARTICLES
        pos_np = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
        vel_np = np.zeros((MAX_PARTICLES, 2), dtype=np.float32)
        type_np = np.zeros(MAX_PARTICLES, dtype=np.int32)
        pos_np[:n] = s['positions'][:n]
        vel_np[:n] = s['velocities'][:n]
        type_np[:n] = s['types'][:n]
        positions.from_numpy(pos_np)
        velocities.from_numpy(vel_np)
        types.from_numpy(type_np)
        num_particles[None] = n

        self.canvas.update_display()
        if was_running:
            self.sim_timer.start(16)

    def rename_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有预设可以重命名。")
            return
        names = sorted(presets.keys())
        old_name, ok = QInputDialog.getItem(self, "重命名预设", "选择预设：", names, 0, False)
        if not ok or not old_name:
            return
        new_name, ok = QInputDialog.getText(self, "重命名", "新名称：", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if new_name in presets:
            reply = QMessageBox.question(self, "覆盖确认",
                                         f"预设 “{new_name}” 已存在，是否覆盖？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        presets[new_name] = presets.pop(old_name)
        self._save_preset_dict(presets)

    def delete_preset(self):
        presets = self._get_preset_dict()
        if not presets:
            QMessageBox.information(self, "提示", "没有预设可以删除。")
            return
        names = sorted(presets.keys())
        name, ok = QInputDialog.getItem(self, "删除预设", "选择预设：", names, 0, False)
        if not ok or not name:
            return
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除预设 “{name}” 吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return
        del presets[name]
        self._save_preset_dict(presets)

    def closeEvent(self, event):
        self.sim_timer.stop()
        ti.reset()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParticleLifeApp()
    window.show()
    sys.exit(app.exec())