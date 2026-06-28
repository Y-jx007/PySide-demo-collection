from OpenGL.GL import shaders

from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32)

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

# ---------- 光线追踪仿真核心 (Taichi) ----------
@ti.data_oriented
class RayTracingSimulation:
    def __init__(self, num_rays=500):
        self.num_rays = num_rays

        self.sx = -2.0
        self.sy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.r = 1.0
        self.n = 1.5

        self.ray_points = ti.Vector.field(2, dtype=ti.f32, shape=(num_rays, 6))
        self.update_rays()

    @ti.func
    def intersect_circle(self, ro, rd, cx, cy, r):
        oc = ti.Vector([ro.x - cx, ro.y - cy])
        a = rd.dot(rd)
        b = 2.0 * oc.dot(rd)
        c = oc.dot(oc) - r * r
        disc = b * b - 4.0 * a * c

        t = -1.0
        normal = ti.Vector([0.0, 0.0])
        hit = ti.Vector([0.0, 0.0])

        if disc >= 0.0:
            sqrt_disc = ti.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)
            t = t0 if t0 > 1e-4 else t1
            if t >= 1e-4:
                hit = ro + t * rd
                normal = (hit - ti.Vector([cx, cy])).normalized()
            else:
                t = -1.0
        return t, normal, hit

    @ti.func
    def refract(self, wi, n, eta):
        cosi = wi.dot(n)
        sin2t = eta * eta * (1.0 - cosi * cosi)
        result = ti.Vector([0.0, 0.0])
        if sin2t <= 1.0:
            cost = ti.sqrt(1.0 - sin2t)
            result = eta * wi + (eta * cosi - cost) * n
        return result

    @ti.kernel
    def trace_rays_kernel(
        self,
        sx: ti.f32, sy: ti.f32,
        cx: ti.f32, cy: ti.f32,
        r: ti.f32,
        n: ti.f32
    ):
        pi = 3.1415926535
        for idx in range(self.num_rays):
            angle = ti.random() * 2.0 * pi
            dir_x = ti.cos(angle)
            dir_y = ti.sin(angle)

            ro = ti.Vector([sx, sy])
            rd = ti.Vector([dir_x, dir_y])

            # 预定义所有变量，避免作用域错误
            P1 = ti.Vector([0.0, 0.0])
            P2 = ti.Vector([0.0, 0.0])
            P3 = ti.Vector([0.0, 0.0])
            rd_in = ti.Vector([0.0, 0.0])
            valid = False

            # 第一次折射：空气 → 玻璃
            t1, n1, tmpP1 = self.intersect_circle(ro, rd, cx, cy, r)
            if t1 > 0.0:
                P1 = tmpP1
                rd_in = self.refract(rd, n1, 1.0 / n)
                if rd_in.norm() > 1e-6:
                    valid = True

            # 第二次折射：玻璃 → 空气
            if valid:
                ro2 = P1 + 1e-4 * rd_in
                t2, n2, tmpP2 = self.intersect_circle(ro2, rd_in, cx, cy, r)
                if t2 > 0.0:
                    P2 = tmpP2
                    outward_n = (P2 - ti.Vector([cx, cy])).normalized()
                    rd_out = self.refract(rd_in, outward_n, n)
                    if rd_out.norm() > 1e-6:
                        ext_len = 5.0
                        P3 = P2 + ext_len * rd_out
                    else:
                        valid = False
                else:
                    valid = False

            # 写入顶点
            if valid:
                self.ray_points[idx, 0] = ro
                self.ray_points[idx, 1] = P1
                self.ray_points[idx, 2] = P1
                self.ray_points[idx, 3] = P2
                self.ray_points[idx, 4] = P2
                self.ray_points[idx, 5] = P3
            else:
                for k in ti.static(range(6)):
                    self.ray_points[idx, k] = ti.Vector([0.0, 0.0])

    def update_rays(self):
        self.trace_rays_kernel(self.sx, self.sy, self.cx, self.cy, self.r, self.n)

    def get_ray_vertices(self):
        pts = self.ray_points.to_numpy()
        return pts.reshape(-1, 2)

# ---------- OpenGL 绘制组件（与之前完全相同）----------
class CausticGLWidget(QOpenGLWidget):
    def __init__(self, simulation, parent=None):
        super().__init__(parent)
        self.sim = simulation
        self.setMinimumSize(600, 600)

        self.vao = None
        self.vbo_rays = None
        self.shader_program = None
        self.num_ray_vertices = 0

        self.bg_color = (0.05, 0.05, 0.1, 1.0)
        self.circle_vao = None
        self.circle_vbo = None
        self.num_circle_vertices = 0

    def initializeGL(self):
        glClearColor(*self.bg_color)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        vertex_shader_source = """
        #version 130
        in vec2 position;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * vec4(position, 0.0, 1.0);
        }
        """
        fragment_shader_source = """
        #version 130
        uniform vec4 color;
        out vec4 fragColor;
        void main() {
            fragColor = color;
        }
        """
        vs = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vs, fs)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo_rays = glGenBuffers(1)

        circle_pts = []
        segments = 200
        for i in range(segments + 1):
            ang = 2.0 * np.pi * i / segments
            circle_pts.append([np.cos(ang), np.sin(ang)])
        circle_data = np.array(circle_pts, dtype=np.float32)
        self.num_circle_vertices = len(circle_data)

        self.circle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.circle_vao)
        self.circle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.circle_vbo)
        glBufferData(GL_ARRAY_BUFFER, circle_data.nbytes, circle_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        w = self.width()
        h = self.height()
        aspect = w / max(h, 1)
        view_size = 4.0
        left = -view_size * aspect
        right = view_size * aspect
        bottom = -view_size
        top = view_size

        proj = np.array([
            [2.0/(right-left), 0.0, 0.0, -(right+left)/(right-left)],
            [0.0, 2.0/(top-bottom), 0.0, -(top+bottom)/(top-bottom)],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        glUseProgram(self.shader_program)
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)

        glBindVertexArray(self.circle_vao)
        color_loc = glGetUniformLocation(self.shader_program, "color")
        glUniform4f(color_loc, 1.0, 1.0, 1.0, 0.8)
        glLineWidth(2.0)
        glDrawArrays(GL_LINE_STRIP, 0, self.num_circle_vertices)

        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(left, right, bottom, top, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glColor3f(1.0, 0.2, 0.2)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex2f(self.sim.sx, self.sim.sy)
        glEnd()

        pts = self.sim.get_ray_vertices()
        self.num_ray_vertices = len(pts)
        if self.num_ray_vertices > 0:
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_rays)
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            glUseProgram(self.shader_program)
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
            glUniform4f(color_loc, 0.3, 0.8, 1.0, 0.6)
            glLineWidth(1.0)
            glDrawArrays(GL_LINES, 0, self.num_ray_vertices)
            glBindVertexArray(0)

        glUseProgram(0)

    def update_simulation(self):
        self.update()

# ---------- 主界面（与之前完全相同）----------
class CausticWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.num_rays = 500
        self.simulation = RayTracingSimulation(num_rays=self.num_rays)

        self.init_ui()
        self.update_display()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()

        light_group = QGroupBox("点光源位置")
        light_layout = QVBoxLayout()
        self.sx_slider, self.sx_label = self.create_slider("光源 X:", -3.0, 3.0, -2.0, 100)
        self.sy_slider, self.sy_label = self.create_slider("光源 Y:", -3.0, 3.0, 0.0, 100)
        light_layout.addWidget(self.sx_slider)
        light_layout.addWidget(self.sy_slider)
        light_group.setLayout(light_layout)
        left_panel.addWidget(light_group)

        cylinder_group = QGroupBox("圆柱参数")
        cyl_layout = QVBoxLayout()
        self.r_slider, self.r_label = self.create_slider("半径:", 0.3, 2.0, 1.0, 100)
        self.n_slider, self.n_label = self.create_slider("折射率:", 1.1, 2.5, 1.5, 100)
        cyl_layout.addWidget(self.r_slider)
        cyl_layout.addWidget(self.n_slider)
        cylinder_group.setLayout(cyl_layout)
        left_panel.addWidget(cylinder_group)

        rays_group = QGroupBox("光线数")
        rays_layout = QVBoxLayout()
        self.rays_slider, self.rays_label = self.create_int_slider("光线数:", 100, 2000, 500, 1)
        rays_layout.addWidget(self.rays_slider)
        rays_group.setLayout(rays_layout)
        left_panel.addWidget(rays_group)

        ctrl_group = QGroupBox("操作")
        ctrl_layout = QVBoxLayout()

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["上方照射", "侧向照射", "近轴聚焦"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        ctrl_layout.addLayout(preset_layout)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存截图")
        self.save_btn.clicked.connect(self.save_image)
        self.reset_btn = QPushButton("重置默认")
        self.reset_btn.clicked.connect(self.reset_defaults)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.reset_btn)
        ctrl_layout.addLayout(btn_layout)

        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(280)
        main_layout.addWidget(left_widget)

        self.gl_widget = CausticGLWidget(self.simulation)
        self.gl_widget.setFixedSize(600, 600)
        main_layout.addWidget(self.gl_widget)

        self.setLayout(main_layout)

    def create_slider(self, name, min_v, max_v, default_v, scale=100):
        container = QWidget()
        layout = QHBoxLayout()
        label = QLabel(f"{name} {default_v:.2f}")
        label.setMinimumWidth(110)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_v * scale))
        slider.setMaximum(int(max_v * scale))
        slider.setValue(int(default_v * scale))

        def on_change(value):
            v = value / scale
            label.setText(f"{name} {v:.2f}")
            self.on_parameter_changed()
        slider.valueChanged.connect(on_change)
        layout.addWidget(label)
        layout.addWidget(slider)
        container.setLayout(layout)
        return container, label

    def create_int_slider(self, name, min_v, max_v, default_v, scale=1):
        container = QWidget()
        layout = QHBoxLayout()
        label = QLabel(f"{name} {default_v}")
        label.setMinimumWidth(110)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_v)
        slider.setMaximum(max_v)
        slider.setValue(default_v)

        def on_change(value):
            label.setText(f"{name} {value}")
            self.num_rays = value
            self.simulation.num_rays = value
            self.simulation.ray_points = ti.Vector.field(2, dtype=ti.f32, shape=(value, 6))
            self.simulation.update_rays()
            self.gl_widget.update()
        slider.valueChanged.connect(on_change)
        layout.addWidget(label)
        layout.addWidget(slider)
        container.setLayout(layout)
        return container, label

    def get_slider_value(self, slider_widget):
        slider = slider_widget.layout().itemAt(1).widget()
        return slider.value() / 100.0

    def on_parameter_changed(self):
        self.simulation.sx = self.get_slider_value(self.sx_slider)
        self.simulation.sy = self.get_slider_value(self.sy_slider)
        self.simulation.r = self.get_slider_value(self.r_slider)
        self.simulation.n = self.get_slider_value(self.n_slider)
        self.simulation.update_rays()
        self.gl_widget.update()

    def update_display(self):
        self.simulation.update_rays()
        self.gl_widget.update()

    def set_slider(self, slider_widget, value):
        slider = slider_widget.layout().itemAt(1).widget()
        slider.setValue(int(value * 100))

    def apply_preset(self, name):
        presets = {
            "上方照射": {"sx": 0.0, "sy": 2.5, "r": 1.0, "n": 1.5},
            "侧向照射": {"sx": -3.0, "sy": 0.0, "r": 1.0, "n": 1.5},
            "近轴聚焦": {"sx": 0.0, "sy": 3.0, "r": 0.8, "n": 1.5},
        }
        if name in presets:
            p = presets[name]
            self.set_slider(self.sx_slider, p["sx"])
            self.set_slider(self.sy_slider, p["sy"])
            self.set_slider(self.r_slider, p["r"])
            self.set_slider(self.n_slider, p["n"])
            self.on_parameter_changed()

    def reset_defaults(self):
        self.set_slider(self.sx_slider, -2.0)
        self.set_slider(self.sy_slider, 0.0)
        self.set_slider(self.r_slider, 1.0)
        self.set_slider(self.n_slider, 1.5)
        self.on_parameter_changed()

    def save_image(self):
        self.gl_widget.makeCurrent()
        w = self.gl_widget.width()
        h = self.gl_widget.height()
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        img = np.flipud(img)
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        path, _ = QFileDialog.getSaveFileName(self, "保存焦散光路图", "caustic_rays.png", "PNG (*.png)")
        if path and qimg.save(path):
            QMessageBox.information(self, "保存成功", f"已保存至:\n{path}")

class CausticWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("圆柱玻璃点光源焦散 - 光线追踪版 (Taichi + OpenGL)")
        self.setGeometry(100, 100, 920, 650)
        self.central = CausticWidget()
        self.setCentralWidget(self.central)

    def closeEvent(self, event):
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = CausticWindow()
    window.show()
    sys.exit(app.exec())