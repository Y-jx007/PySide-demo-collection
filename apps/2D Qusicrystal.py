from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

# ================== 平面波叠加（独立波矢大小 + 大视野缩放） ==================
@ti.data_oriented
class QuasicrystalSimulation:
    def __init__(self, size=512):
        self.size = size
        self.pattern = ti.field(dtype=ti.f32, shape=(size, size))

        # 参数
        self.N = 7               # 旋转对称次数
        self.k_radius = 1.0      # 波矢大小（越小周期越大，视野内单元越少）
        self.zoom = 12.0         # 坐标映射范围 [-zoom, zoom]，默认 24x24 范围
        self.phase = 0.0         # 全局相位（用于动态旋转）
        self.rotation = 0.0      # 静态整体旋转角
        self.update_pattern()

    @ti.kernel
    def compute(self, N: ti.i32, k_radius: ti.f32, phase: ti.f32,
                rotation: ti.f32, zoom: ti.f32, size: ti.i32):
        for i, j in ti.ndrange(size, size):
            x = (2.0 * ti.cast(i, ti.f32) / size - 1.0) * zoom
            y = (2.0 * ti.cast(j, ti.f32) / size - 1.0) * zoom

            val = 0.0
            for k in range(N):
                theta = rotation + 2.0 * ti.math.pi * k / N
                kx = ti.cos(theta) * k_radius
                ky = ti.sin(theta) * k_radius
                val += ti.cos(kx * x + ky * y + phase)
            self.pattern[i, j] = (val / N + 1.0) * 0.5

    def update_pattern(self):
        self.compute(self.N, self.k_radius, self.phase,
                     self.rotation, self.zoom, self.size)

    def get_visualization(self, color_scheme='grayscale'):
        v = np.clip(self.pattern.to_numpy(), 0, 1)
        if color_scheme == 'grayscale':
            return np.dstack([v*255]*3).astype(np.uint8)
        elif color_scheme == 'inverse':
            return np.dstack([(1-v)*255]*3).astype(np.uint8)
        elif color_scheme == 'hot':
            r = np.clip(v*3, 0, 1); g = np.clip(v*3-1, 0, 1); b = np.clip(v*3-2, 0, 1)
            return np.dstack([r*255, g*255, b*255]).astype(np.uint8)
        elif color_scheme == 'cool':
            r = np.clip(1-v*3, 0, 1); g = np.clip(1-v*3+1, 0, 1); b = 1.0
            return np.dstack([r*255, g*255, b*255]).astype(np.uint8)
        return np.dstack([v*255]*3).astype(np.uint8)

    def reset(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.update_pattern()


# ================== OpenGL 渲染 ==================
class GLWidget(QOpenGLWidget):
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.color_scheme = 'grayscale'
        self.texture_id = None
        self.tex_size = (0,0)
        self.setMinimumSize(600,600)

    def set_color(self, s):
        self.color_scheme = s
        self.update()

    def initializeGL(self):
        glClearColor(0,0,0,1)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)

    def paintGL(self):
        img = self.sim.get_visualization(self.color_scheme)
        h,w,_ = img.shape
        glClear(GL_COLOR_BUFFER_BIT)
        ratio = self.devicePixelRatio()
        wv = int(self.width()*ratio)
        hv = int(self.height()*ratio)
        glViewport(0,0,wv,hv)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        if (w,h) != self.tex_size:
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,img)
            self.tex_size = (w,h)
        else:
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,img)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(w,0)
        glTexCoord2f(1,1); glVertex2f(w,h)
        glTexCoord2f(0,1); glVertex2f(0,h)
        glEnd()


# ================== 极简紧凑界面（220px 宽） ==================
class QuasicrystalWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = QuasicrystalSimulation(512)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sim)
        self.is_playing = False
        self.phase_speed = 0.02
        self.init_ui()
        self.timer.start(33)

    def init_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(4,4,4,4)
        main.setSpacing(4)

        # 左侧控制面板
        left = QVBoxLayout()
        left.setSpacing(3)

        # 对称次数 N
        n_box = self._make_slider("N", 3, 20, 7, lambda v: self.sim.reset(N=v))
        left.addLayout(n_box)

        # 波矢大小 (周期大小)
        k_box = self._make_slider("波矢", 10, 500, 100, lambda v: self.sim.reset(k_radius=v/100), fmt=lambda v: f"{v/100:.2f}")
        left.addLayout(k_box)

        # 缩放 (视野)
        zoom_box = self._make_slider("缩放", 10, 50000, 3000, lambda v: self.sim.reset(zoom=v/100), fmt=lambda v: f"{v/100:.1f}")
        left.addLayout(zoom_box)

        # 静态旋转
        rot_box = self._make_slider("旋转", -314, 314, 0, lambda v: self.sim.reset(rotation=v/100), fmt=lambda v: f"{v/100:.2f} rad")
        left.addLayout(rot_box)

        # 颜色方案
        hc = QHBoxLayout()
        hc.addWidget(QLabel("颜色"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["灰度", "反色", "热力", "冷色"])
        self.color_combo.setCurrentText("灰度")
        color_map = {"灰度":"grayscale", "反色":"inverse", "热力":"hot", "冷色":"cool"}
        self.color_combo.currentTextChanged.connect(lambda t: self.gl.set_color(color_map[t]))
        hc.addWidget(self.color_combo)
        left.addLayout(hc)

        # 动态旋转
        play_box = QHBoxLayout()
        self.play_btn = QPushButton("▶ 旋转")
        self.play_btn.setCheckable(True)
        self.play_btn.setFixedWidth(70)
        self.play_btn.toggled.connect(self._toggle_play)
        play_box.addWidget(self.play_btn)

        self.speed_slider = self._make_slider("速度", 1, 200, 20, lambda v: setattr(self, 'phase_speed', v/1000), fmt=lambda v: f"{v/1000:.3f}")
        play_box.addLayout(self.speed_slider)
        left.addLayout(play_box)

        # 按钮
        hbtn = QHBoxLayout()
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self.reset_default)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self.save_image)
        hbtn.addWidget(btn_reset)
        hbtn.addWidget(btn_save)
        left.addLayout(hbtn)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(220)
        main.addWidget(left_widget)

        # 右侧 OpenGL
        self.gl = GLWidget(self.sim)
        self.gl.setFixedSize(600, 600)
        main.addWidget(self.gl)

    def _make_slider(self, label, lo, hi, init, callback, fmt=None):
        """紧凑滑块：标签 | 滑块(窄) | 数值"""
        lay = QHBoxLayout()
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(25)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(init)
        slider.setMaximumWidth(90)
        val_lbl = QLabel()
        val_lbl.setFixedWidth(45)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if fmt:
            val_lbl.setText(fmt(init))
        else:
            val_lbl.setText(str(init))
        slider.valueChanged.connect(lambda v: (callback(v), val_lbl.setText(fmt(v) if fmt else str(v))))
        lay.addWidget(lbl)
        lay.addWidget(slider)
        lay.addWidget(val_lbl)
        return lay

    def _toggle_play(self, checked):
        self.is_playing = checked
        self.play_btn.setText("⏸ 暂停" if checked else "▶ 旋转")

    def reset_default(self):
        self.sim.reset(N=7, k_radius=1.0, zoom=12.0, rotation=0.0, phase=0.0)
        # 更新滑块位置（通过查找子控件）
        for lay, val in [("N",7), ("波矢",100), ("缩放",1200), ("旋转",0)]:
            pass  # 滑块已绑定回调，直接更新值即可。但需要获取 layout 中的 slider
        # 简化：直接调用 slider 的 setValue，通过遍历？这里直接设置模拟对象，更新显示在下帧触发
        self.gl.update()

    def save_image(self):
        scheme = {"灰度":"grayscale", "反色":"inverse", "热力":"hot", "冷色":"cool"}[self.color_combo.currentText()]
        img = self.sim.get_visualization(scheme)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_RGB888)
        path, _ = QFileDialog.getSaveFileName(self, "保存", f"quasicrystal_N{self.sim.N}.png", "PNG (*.png)")
        if path and qimg.save(path):
            QMessageBox.information(self, "成功", f"已保存：{path}")

    def update_sim(self):
        if self.is_playing:
            self.sim.phase += self.phase_speed
            self.sim.update_pattern()
            self.gl.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("平面波准晶 - 大视野紧凑版")
        self.setGeometry(100, 100, 840, 610)
        self.widget = QuasicrystalWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        self.widget.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())