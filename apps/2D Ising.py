from custom_import import *

ti.init(arch=ti.gpu)

# ── 序参量曲线组件 ──────────────────────────
class OrderParameterWidget(QWidget):
    """显示磁化强度和能量演化曲线"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMaximumHeight(250)
        self.mag_history = []
        self.eng_history = []
        self.max_len = 300

    def add_data(self, magnetization, energy):
        self.mag_history.append(magnetization)
        self.eng_history.append(energy)
        if len(self.mag_history) > self.max_len:
            self.mag_history.pop(0)
            self.eng_history.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        if len(self.mag_history) < 2:
            return

        margin = 30
        plot_rect = self.rect().adjusted(margin, margin, -margin, -margin)

        # 坐标轴
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(plot_rect.left(), plot_rect.center().y(),
                         plot_rect.right(), plot_rect.center().y())
        painter.drawLine(plot_rect.left(), plot_rect.top(),
                         plot_rect.left(), plot_rect.bottom())

        # 通用曲线绘制方法
        def draw_curve(data, color, y_scale):
            painter.setPen(QPen(color, 2))
            step = plot_rect.width() / (len(data) - 1)
            for i in range(len(data) - 1):
                x1 = plot_rect.left() + i * step
                y1 = plot_rect.center().y() - data[i] * y_scale
                x2 = plot_rect.left() + (i + 1) * step
                y2 = plot_rect.center().y() - data[i + 1] * y_scale
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        y_scale = plot_rect.height() / 4
        draw_curve(self.mag_history, QColor(0, 100, 200), y_scale)  # 磁化 (蓝)
        draw_curve(self.eng_history, QColor(200, 0, 0), y_scale)    # 能量 (红)

        # 当前值
        painter.setPen(Qt.black)
        painter.drawText(60, 25, f"M: {self.mag_history[-1]:.4f}")
        painter.drawText(180, 25, f"E: {self.eng_history[-1]:.4f}")


# ── 伊辛模型 (Taichi 数据导向) ─────────────────
@ti.data_oriented
class IsingModel:
    def __init__(self, size=1024):
        self.size = size
        self.temperature = 2.269
        self.steps_per_frame = 1
        self.is_paused = False
        self.use_wolff = False
        self.use_checkerboard = True

        total_spins = self.size * self.size
        self.updates_per_step = total_spins // 4

        # Taichi 字段
        self.spins = ti.field(ti.i8, (size, size))
        self.temperature_field = ti.field(ti.f32, ())
        self.updates_per_step_field = ti.field(ti.i32, ())
        self.texture_field = ti.field(ti.f32, (size, size, 3))

        # 用于存储磁化和能量的输出字段
        self._mag_out = ti.field(ti.f32, ())
        self._eng_out = ti.field(ti.f32, ())

        self.temperature_field[None] = self.temperature
        self.updates_per_step_field[None] = self.updates_per_step
        self.reset_spins()

    @ti.kernel
    def update_texture(self):
        for i, j in self.spins:
            value = (ti.cast(self.spins[i, j], ti.f32) + 1.0) * 0.5
            self.texture_field[i, j, 0] = value
            self.texture_field[i, j, 1] = value
            self.texture_field[i, j, 2] = value

    @ti.kernel
    def taichi_checkerboard_step(self, mask_type: ti.i32):
        grid = self.spins.shape[0]
        T = self.temperature_field[None]
        ups = self.updates_per_step_field[None]
        half = (grid * grid + 1) // 2
        p = ti.min(1.0, ups / half)

        for i, j in ti.ndrange(grid, grid):
            if (mask_type == 0 and (i + j) % 2 == 0) or (mask_type == 1 and (i + j) % 2 == 1):
                if ti.random() < p:
                    nb_sum = (self.spins[(i - 1) % grid, j] + self.spins[(i + 1) % grid, j] +
                              self.spins[i, (j - 1) % grid] + self.spins[i, (j + 1) % grid])
                    dE = 2.0 * self.spins[i, j] * nb_sum
                    if dE <= 0.0 or ti.random() < ti.exp(-dE / T):
                        self.spins[i, j] = -self.spins[i, j]

    @ti.kernel
    def taichi_monte_carlo_step(self):
        grid = self.spins.shape[0]
        T = self.temperature_field[None]
        ups = self.updates_per_step_field[None]

        ti.loop_config(serialize=True)
        for _ in range(ups):
            i = ti.cast(ti.floor(ti.random() * grid), ti.i32)
            j = ti.cast(ti.floor(ti.random() * grid), ti.i32)
            nb_sum = (self.spins[(i - 1) % grid, j] + self.spins[(i + 1) % grid, j] +
                      self.spins[i, (j - 1) % grid] + self.spins[i, (j + 1) % grid])
            dE = 2.0 * self.spins[i, j] * nb_sum
            if dE <= 0.0 or ti.random() < ti.exp(-dE / T):
                self.spins[i, j] = -self.spins[i, j]

    def cpu_wolff_step(self):
        spins_np = self.spins.to_numpy()
        size = self.size
        cluster_prob = 1.0 - np.exp(-2.0 / self.temperature)

        i0, j0 = np.random.randint(0, size, 2)
        cluster_sign = spins_np[i0, j0]
        cluster = set()
        stack = [(i0, j0)]

        while stack:
            i, j = stack.pop()
            if (i, j) in cluster:
                continue
            cluster.add((i, j))
            for ni, nj in [((i-1)%size, j), ((i+1)%size, j),
                           (i, (j-1)%size), (i, (j+1)%size)]:
                if (ni, nj) not in cluster and spins_np[ni, nj] == cluster_sign:
                    if np.random.random() < cluster_prob:
                        stack.append((ni, nj))

        for i, j in cluster:
            spins_np[i, j] *= -1
        self.spins.from_numpy(spins_np)

    @ti.kernel
    def _calc_observables(self):
        grid = self.spins.shape[0]
        mag_sum = 0.0
        eng_sum = 0.0
        for i, j in ti.ndrange(grid, grid):
            s = self.spins[i, j]
            mag_sum += ti.cast(s, ti.f32)
            nb = (self.spins[(i-1)%grid, j] + self.spins[(i+1)%grid, j] +
                  self.spins[i, (j-1)%grid] + self.spins[i, (j+1)%grid])
            eng_sum += -s * nb / 4.0
        N = grid * grid
        self._mag_out[None] = mag_sum / N
        self._eng_out[None] = eng_sum / N + 1.0

    def calculate_observables(self):
        self._calc_observables()
        return self._mag_out[None], self._eng_out[None]

    def set_temperature(self, T):
        self.temperature = T
        self.temperature_field[None] = T

    def set_steps_per_frame(self, steps):
        self.steps_per_frame = steps

    def set_updates_per_step(self, ups):
        self.updates_per_step = ups
        self.updates_per_step_field[None] = ups

    def reset_spins(self):
        spins_np = np.random.choice([-1, 1], (self.size, self.size)).astype(np.int8)
        self.spins.from_numpy(spins_np)


# ── OpenGL 渲染组件 ─────────────────────────
class IsingModelGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_size = 1024
        self.sim = IsingModel(self.grid_size)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_model)
        self.timer.start(0)
        self.setMinimumSize(540, 540)

        self.last_mask = 0
        self.texture_id = None

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        for param in [GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER]:
            glTexParameteri(GL_TEXTURE_2D, param, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        self.sim.update_texture()
        tex_data = self.sim.texture_field.to_numpy()
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.grid_size, self.grid_size,
                     0, GL_RGB, GL_FLOAT, tex_data)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()

    def set_grid_size(self, size):
        if size == self.grid_size:
            return
        T = self.sim.temperature
        paused = self.sim.is_paused
        wolff = self.sim.use_wolff
        checker = self.sim.use_checkerboard
        steps = self.sim.steps_per_frame

        self.sim = IsingModel(size)
        self.grid_size = size

        self.sim.set_temperature(T)
        self.sim.is_paused = paused
        self.sim.use_wolff = wolff
        self.sim.use_checkerboard = checker
        self.sim.steps_per_frame = steps

        if hasattr(self, 'updates_callback'):
            self.updates_callback(self.sim.updates_per_step)
        self.update()

    def update_model(self):
        if self.sim.is_paused:
            return
        for _ in range(self.sim.steps_per_frame):
            if self.sim.use_wolff:
                self.sim.cpu_wolff_step()
            else:
                if self.sim.use_checkerboard:
                    self.last_mask = 1 - self.last_mask
                    self.sim.taichi_checkerboard_step(self.last_mask)
                else:
                    self.sim.taichi_monte_carlo_step()

        mag, eng = self.sim.calculate_observables()
        if hasattr(self, 'data_callback'):
            self.data_callback(mag, eng)
        self.update()

    def set_data_callback(self, cb):
        self.data_callback = cb
    def set_updates_callback(self, cb):
        self.updates_callback = cb
    def set_temperature(self, T):
        self.sim.set_temperature(T)
    def set_steps_per_frame(self, steps):
        self.sim.set_steps_per_frame(steps)
    def set_updates_per_step(self, ups):
        self.sim.set_updates_per_step(ups)
    def set_use_checkerboard(self, val):
        self.sim.use_checkerboard = val
    def reset_spins(self):
        self.sim.reset_spins()
    def toggle_pause(self):
        self.sim.is_paused = not self.sim.is_paused
    def set_use_wolff(self, val):
        self.sim.use_wolff = val

    @property
    def is_paused(self):
        return self.sim.is_paused
    @property
    def updates_per_step(self):
        return self.sim.updates_per_step


# ── 主窗口 ──────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("二维伊辛模型 (Taichi + OpenGL)")
        self.setGeometry(50, 50, 800, 540)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── 左侧控制面板 ──
        left = QWidget()
        left.setMaximumWidth(350)
        left_layout = QGridLayout(left)

        # 分辨率
        res_grp = QGroupBox("网格分辨率")
        res_vbox = QVBoxLayout(res_grp)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(64, 2048)
        self.grid_spin.setValue(1024)
        self.grid_spin.valueChanged.connect(self._on_grid_changed)
        res_vbox.addWidget(self.grid_spin)
        preset_row = QHBoxLayout()
        for s in [512, 1024, 2048]:
            btn = QPushButton(str(s))
            btn.clicked.connect(lambda _, v=s: self.grid_spin.setValue(v))
            preset_row.addWidget(btn)
        res_vbox.addLayout(preset_row)
        left_layout.addWidget(res_grp, 0, 0)

        # 温度
        temp_grp = QGroupBox("温度控制")
        temp_vbox = QVBoxLayout(temp_grp)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0001, 1000.0)
        self.temp_spin.setSingleStep(0.0001)
        self.temp_spin.setDecimals(4)
        self.temp_spin.setValue(2.269)
        self.temp_spin.setKeyboardTracking(False)
        self.temp_spin.valueChanged.connect(self._on_temp_spin)
        temp_vbox.addWidget(self.temp_spin)

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(1)
        self.temp_slider.setMaximum(100000)
        self.temp_slider.setValue(int(2.269 * 10000))
        self.temp_slider.valueChanged.connect(self._on_temp_slider)
        temp_vbox.addWidget(self.temp_slider)

        self.temp_label = QLabel("当前: 2.2690")
        temp_vbox.addWidget(self.temp_label)
        left_layout.addWidget(temp_grp, 1, 0)

        # 算法设置
        algo_grp = QGroupBox("算法设置")
        algo_vbox = QVBoxLayout(algo_grp)
        self.wolff_chk = QCheckBox("Wolff 算法")
        self.wolff_chk.toggled.connect(self._on_wolff_toggled)
        algo_vbox.addWidget(self.wolff_chk)
        self.checker_chk = QCheckBox("棋盘格优化")
        self.checker_chk.setChecked(True)
        self.checker_chk.toggled.connect(self._on_checker_toggled)
        algo_vbox.addWidget(self.checker_chk)
        self.perf_label = QLabel("性能: 棋盘格优化")
        algo_vbox.addWidget(self.perf_label)
        left_layout.addWidget(algo_grp, 0, 1)

        # 性能控制
        perf_grp = QGroupBox("性能控制")
        perf_vbox = QVBoxLayout(perf_grp)
        perf_vbox.addWidget(QLabel("每帧更新次数:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1000)
        self.steps_spin.setValue(1)
        self.steps_spin.valueChanged.connect(lambda v: self.gl_widget.set_steps_per_frame(v))
        perf_vbox.addWidget(self.steps_spin)

        perf_vbox.addWidget(QLabel("每次尝试次数:"))
        self.updates_spin = QSpinBox()
        self.updates_spin.setRange(1, 20000000)
        self.updates_spin.valueChanged.connect(lambda v: self.gl_widget.set_updates_per_step(v))
        perf_vbox.addWidget(self.updates_spin)
        left_layout.addWidget(perf_grp, 1, 1)

        # 状态控制
        ctrl_grp = QGroupBox("状态控制")
        ctrl_vbox = QVBoxLayout(ctrl_grp)
        row1 = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self._toggle_pause)
        row1.addWidget(self.pause_btn)
        self.reset_btn = QPushButton("随机重置")
        self.reset_btn.clicked.connect(self._reset)
        row1.addWidget(self.reset_btn)
        ctrl_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        ferro_btn = QPushButton("铁磁")
        ferro_btn.clicked.connect(lambda: self._set_all_spins(1))
        row2.addWidget(ferro_btn)
        anti_btn = QPushButton("反铁磁")
        anti_btn.clicked.connect(self._set_antiferro)
        row2.addWidget(anti_btn)
        ctrl_vbox.addLayout(row2)
        left_layout.addWidget(ctrl_grp, 2, 1)

        # 序参量曲线
        self.order_widget = OrderParameterWidget()
        left_layout.addWidget(self.order_widget, 3, 0, 1, 2)

        # ── 右侧 OpenGL 视图 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.gl_widget = IsingModelGLWidget()
        self.gl_widget.set_updates_callback(self._on_widget_updates_changed)
        self.gl_widget.set_data_callback(self.order_widget.add_data)
        right_layout.addWidget(self.gl_widget)
        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)

        # 初始同步
        self.updates_spin.setValue(self.gl_widget.updates_per_step)
        self.current_temp = 2.269

    # ── 内部辅助方法 ──
    def _set_temperature(self, T):
        self.current_temp = T
        self.gl_widget.set_temperature(T)
        self.temp_label.setText(f"当前: {T:.4f}")

    def _update_perf_label(self):
        if self.wolff_chk.isChecked():
            text = "性能: Wolff 算法"
        else:
            text = "性能: " + ("棋盘格优化" if self.checker_chk.isChecked() else "标准蒙特卡洛")
        self.perf_label.setText(text)

    # ── 信号槽 ──
    def _on_grid_changed(self, size):
        self.gl_widget.set_grid_size(size)
        self.order_widget.mag_history.clear()
        self.order_widget.eng_history.clear()
        self.order_widget.update()

    def _on_temp_spin(self, val):
        self._set_temperature(val)
        self.temp_slider.blockSignals(True)
        self.temp_slider.setValue(int(val * 10000))
        self.temp_slider.blockSignals(False)

    def _on_temp_slider(self, val):
        T = val / 10000.0
        self._set_temperature(T)
        self.temp_spin.blockSignals(True)
        self.temp_spin.setValue(T)
        self.temp_spin.blockSignals(False)

    def _on_wolff_toggled(self, checked):
        self.gl_widget.set_use_wolff(checked)
        self._update_perf_label()

    def _on_checker_toggled(self, checked):
        self.gl_widget.set_use_checkerboard(checked)
        if not self.wolff_chk.isChecked():
            self._update_perf_label()

    def _on_widget_updates_changed(self, val):
        self.updates_spin.setValue(val)

    def _toggle_pause(self):
        self.gl_widget.toggle_pause()
        self.pause_btn.setText("继续" if self.gl_widget.is_paused else "暂停")

    def _reset(self):
        self.gl_widget.reset_spins()
        self.order_widget.mag_history.clear()
        self.order_widget.eng_history.clear()
        self.order_widget.update()

    def _set_all_spins(self, val):
        arr = np.full((self.gl_widget.grid_size, self.gl_widget.grid_size), val, dtype=np.int8)
        self.gl_widget.sim.spins.from_numpy(arr)
        self.gl_widget.update()

    def _set_antiferro(self):
        sz = self.gl_widget.grid_size
        arr = np.ones((sz, sz), dtype=np.int8)
        arr[1::2, ::2] = -1
        arr[::2, 1::2] = -1
        self.gl_widget.sim.spins.from_numpy(arr)
        self.gl_widget.update()

    def closeEvent(self, event):
        self.gl_widget.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())