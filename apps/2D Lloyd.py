from scipy.spatial import Voronoi
from custom_import import *   # 保留你的自定义导入，不动

ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)
_EPS = 1e-20

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

class VoronoiWidget(QOpenGLWidget):
   
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.array([])
        self.bbox = (0.0, 1.0, 0.0, 1.0)
        self.show_points = True
        self.show_voronoi = True
        self.setMinimumSize(600, 600)
        self._cached_polygons = []
        self._cached_sides = []

    def set_data(self, points, regions, vertices, region_sides=None):
        self.points = points
        self._cached_polygons = [vertices[r] if len(r) >= 3 else None for r in regions]
        self._cached_sides = (region_sides if region_sides is not None
                              else [len(r) for r in regions])
        self.update()

    def set_show_options(self, show_points, show_voronoi):
        self.show_points = show_points
        self.show_voronoi = show_voronoi
        self.update()

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if len(self.points) == 0:
            return

        ratio = self.devicePixelRatio()
        w = int(self.width() * ratio)
        h = int(self.height() * ratio)
        if w <= 0 or h <= 0:
            return

        xmin, xmax, ymin, ymax = self.bbox
        world_w = xmax - xmin
        world_h = ymax - ymin
        aspect = w / h

        if world_w / world_h > aspect:
            new_h = world_w / aspect
            center_y = (ymin + ymax) / 2.0
            ymin_o, ymax_o = center_y - new_h / 2.0, center_y + new_h / 2.0
            xmin_o, xmax_o = xmin, xmax
        else:
            new_w = world_h * aspect
            center_x = (xmin + xmax) / 2.0
            xmin_o, xmax_o = center_x - new_w / 2.0, center_x + new_w / 2.0
            ymin_o, ymax_o = ymin, ymax

        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(xmin_o, xmax_o, ymin_o, ymax_o, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.show_voronoi:
            for poly, sides in zip(self._cached_polygons, self._cached_sides):
                if poly is None or len(poly) < 3:
                    continue
                glColor4f(1.0, 0.8745, 0.3490, 1.0) if sides == 6 else glColor4f(0.5647, 0.9333, 0.5647, 0.7059)
                glBegin(GL_POLYGON)
                for v in poly:
                    glVertex2f(v[0], v[1])
                glEnd()

            glColor4f(0.8039, 0.5216, 0.0, 1.0)
            glLineWidth(1.5)
            for poly in self._cached_polygons:
                if poly is None or len(poly) < 3:
                    continue
                glBegin(GL_LINE_LOOP)
                for v in poly:
                    glVertex2f(v[0], v[1])
                glEnd()

        if self.show_points:
            glColor4f(0.3922, 0.3922, 0.3922, 1.0)
            glPointSize(2.0)
            glBegin(GL_POINTS)
            for pt in self.points:
                glVertex2f(pt[0], pt[1])
            glEnd()


@ti.kernel
def compute_centroids_shoelace(
    polygons: ti.types.ndarray(),
    region_offsets: ti.types.ndarray(),
    region_sizes: ti.types.ndarray(),
    centroids: ti.types.ndarray(),
    n_regions: ti.i32
):
    
    ti.loop_config(parallelize=8)
    for i in range(n_regions):
        start = region_offsets[i]
        nv = region_sizes[i]
        if nv < 3:
            centroids[i, 0] = 0.0
            centroids[i, 1] = 0.0
            continue
        area = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(nv):
            k = (j + 1) % nv
            x1, y1 = polygons[start + j, 0], polygons[start + j, 1]
            x2, y2 = polygons[start + k, 0], polygons[start + k, 1]
            cross = x1 * y2 - x2 * y1
            area += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross

        if ti.abs(area) < 1e-12:
            avg_x = 0.0
            avg_y = 0.0
            for j in range(nv):
                avg_x += polygons[start + j, 0]
                avg_y += polygons[start + j, 1]
            centroids[i, 0] = avg_x / nv
            centroids[i, 1] = avg_y / nv
        else:
            area *= 0.5
            centroids[i, 0] = cx / (6.0 * area)
            centroids[i, 1] = cy / (6.0 * area)


@ti.kernel
def compute_monte_carlo_centroids(
    polygons: ti.types.ndarray(),
    offsets: ti.types.ndarray(),
    sizes: ti.types.ndarray(),
    centroids: ti.types.ndarray(),
    n_regions: ti.i32,
    samples: ti.i32,
    prob: ti.f32
):
    # ... 保持不变 ...
    ti.loop_config(parallelize=8)
    for i in range(n_regions):
        nv = sizes[i]
        if nv < 3:
            continue
        if prob < 1.0 and ti.random() > prob:
            continue
        start = offsets[i]
        min_x = polygons[start, 0]
        max_x = min_x
        min_y = polygons[start, 1]
        max_y = min_y
        for j in range(start, start + nv):
            x = polygons[j, 0]
            y = polygons[j, 1]
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y

        cx = 0.0
        cy = 0.0
        cnt = 0
        for _ in range(samples):
            px = min_x + ti.random() * (max_x - min_x)
            py = min_y + ti.random() * (max_y - min_y)
            inside = False
            j = start + nv - 1
            for k in range(start, start + nv):
                yi = polygons[j, 1]
                yj = polygons[k, 1]
                if (yi > py) != (yj > py):
                    xi = polygons[j, 0]
                    xj = polygons[k, 0]
                    if px < (xj - xi) * (py - yi) / (yj - yi + 1e-20) + xi:
                        inside = not inside
                j = k
            if inside:
                cx += px
                cy += py
                cnt += 1
        if cnt > 0:
            centroids[i, 0] = cx / cnt
            centroids[i, 1] = cy / cnt

@ti.kernel
def move_points(
    points: ti.types.ndarray(),
    centroids: ti.types.ndarray(),
    mobility: ti.f32,
    xmin: ti.f32, xmax: ti.f32,
    ymin: ti.f32, ymax: ti.f32,
    n_points: ti.i32
):
    # ... 保持不变 ...
    ti.loop_config(parallelize=8)
    for i in range(n_points):
        points[i, 0] += (centroids[i, 0] - points[i, 0]) * mobility
        points[i, 1] += (centroids[i, 1] - points[i, 1]) * mobility
        points[i, 0] = ti.max(xmin + 1e-6, ti.min(xmax - 1e-6, points[i, 0]))
        points[i, 1] = ti.max(ymin + 1e-6, ti.min(ymax - 1e-6, points[i, 1]))


class TaichiHoneycombCVT:
    
    def __init__(self, n_points=300, bbox=(0.0, 1.0, 0.0, 1.0),
                 mobility=2.0, seed=42,
                 monte_enabled=False, monte_samples=50, monte_probability=0.2):
        self.xmin, self.xmax, self.ymin, self.ymax = bbox
        self.mobility = mobility
        self.step_count = 0
        self.monte_enabled = monte_enabled
        self.monte_samples = int(monte_samples)
        self.monte_probability = monte_probability
        self.n_points = n_points
        self.rng = np.random.RandomState(seed)
        self.points = self._init_points(n_points)
        self._prev_points = self.points.copy()
        self._recompute_tol = 1e-6 * max(self.xmax - self.xmin, self.ymax - self.ymin)
        self._cached_regions = None
        self._cached_vertices = None
        self._last_regions = None
        self._last_vertices = None
        self._last_sides = None

        self._init_taichi_arrays()

    def _init_points(self, n_points):
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True, seed=self.rng.randint(0, 10000))
        pts = sobol.random(n_points).astype(np.float32)
        pts[:, 0] = self.xmin + pts[:, 0] * (self.xmax - self.xmin)
        pts[:, 1] = self.ymin + pts[:, 1] * (self.ymax - self.ymin)
        jitter = (self.rng.rand(n_points, 2) - 0.5) * 0.05 * (self.xmax - self.xmin)
        pts += jitter.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], self.xmin + 1e-6, self.xmax - 1e-6)
        pts[:, 1] = np.clip(pts[:, 1], self.ymin + 1e-6, self.ymax - 1e-6)
        return pts

    def _init_taichi_arrays(self):
        max_vertices_per_region = 64
        self.total_vertices = self.n_points * max_vertices_per_region
        self.points_ti = ti.ndarray(dtype=ti.f32, shape=(self.n_points, 2))
        self.centroids_ti = ti.ndarray(dtype=ti.f32, shape=(self.n_points, 2))
        self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(self.total_vertices, 2))
        self.region_offsets_ti = ti.ndarray(dtype=ti.i32, shape=(self.n_points,))
        self.region_sizes_ti = ti.ndarray(dtype=ti.i32, shape=(self.n_points,))
        self.points_ti.from_numpy(self.points.astype(np.float32))
        self._polygons_np = np.zeros((self.total_vertices, 2), dtype=np.float32)

    def _ensure_polygon_capacity(self, required):
        if required > self.total_vertices:
            new_total = max(int(required * 1.5), self.total_vertices * 2)
            self.total_vertices = new_total
            self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(new_total, 2))
            self._polygons_np = np.zeros((new_total, 2), dtype=np.float32)

    def _finite_voronoi(self, vor, radius=None):
        if radius is None:
            radius = np.ptp(vor.points, axis=0).max() * 2.0
        center = vor.points.mean(axis=0)
        ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))
        verts_np = vor.vertices
        extra_verts = []
        new_regions = []
        for p in range(len(vor.points)):
            region = vor.regions[vor.point_region[p]]
            if all(v >= 0 for v in region):
                new_regions.append(region)
                continue
            new_region = [v for v in region if v >= 0]
            if p in ridges:
                for q, v1, v2 in ridges[p]:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        continue
                    t = vor.points[q] - vor.points[p]
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])
                    midpoint = (vor.points[p] + vor.points[q]) * 0.5
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = verts_np[v2] + direction * radius
                    extra_verts.append(far_point)
                    new_region.append(len(verts_np) + len(extra_verts) - 1)
            if len(new_region) > 2:
                all_verts = np.vstack([verts_np] + extra_verts) if extra_verts else verts_np
                vs = all_verts[new_region]
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_region = [new_region[i] for i in np.argsort(angles)]
            new_regions.append(new_region)
        all_vertices = np.vstack([verts_np] + extra_verts) if extra_verts else verts_np
        return new_regions, np.asarray(all_vertices, dtype=np.float32)

    def _compute_sides_vectorized(self, regions, vertices):
        sides = np.zeros(len(regions), dtype=np.int32)
        for i, region in enumerate(regions):
            nv = len(region)
            if nv < 3:
                sides[i] = 0
                continue
            poly = vertices[region]
            if (np.any(poly[:, 0] < self.xmin - 1e-9) or
                np.any(poly[:, 0] > self.xmax + 1e-9) or
                np.any(poly[:, 1] < self.ymin - 1e-9) or
                np.any(poly[:, 1] > self.ymax + 1e-9)):
                sides[i] = nv + 1
            else:
                sides[i] = nv
        return sides

    def step(self):
        self.step_count += 1
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

        need_recompute = (self._cached_regions is None or self._cached_vertices is None)
        if not need_recompute:
            max_disp = np.max(np.abs(self.points - self._prev_points))
            if max_disp > self._recompute_tol:
                need_recompute = True

        if need_recompute:
            vor = Voronoi(self.points)
            regions, vertices = self._finite_voronoi(vor)
            self._cached_regions = regions
            self._cached_vertices = vertices
            self._prev_points = self.points.copy()
        else:
            regions = self._cached_regions
            vertices = self._cached_vertices

        sides_list = self._compute_sides_vectorized(regions, vertices)
        self._ensure_polygon_capacity(self.n_points * 64)

        region_sizes = np.zeros(self.n_points, dtype=np.int32)
        region_offsets = np.zeros(self.n_points, dtype=np.int32)
        idx = 0

        for i, region in enumerate(regions):
            nv = len(region)
            if nv < 3:
                region_sizes[i] = 0
                continue
            poly_ref = vertices[region]
            if np.max(np.abs(poly_ref[0] - poly_ref[-1])) > 1e-9:
                nv_closed = nv + 1
            else:
                nv_closed = nv

            while idx + nv_closed > self.total_vertices:
                self._ensure_polygon_capacity(idx + nv_closed)

            region_sizes[i] = nv_closed
            region_offsets[i] = idx
            self._polygons_np[idx:idx + nv] = poly_ref
            if nv_closed > nv:
                self._polygons_np[idx + nv] = poly_ref[0]
            idx += nv_closed

        self.polygons_ti.from_numpy(self._polygons_np)
        self.region_offsets_ti.from_numpy(region_offsets)
        self.region_sizes_ti.from_numpy(region_sizes)

        compute_centroids_shoelace(
            self.polygons_ti, self.region_offsets_ti, self.region_sizes_ti,
            self.centroids_ti, self.n_points
        )

        if self.monte_enabled and self.monte_samples > 0:
            compute_monte_carlo_centroids(
                self.polygons_ti, self.region_offsets_ti, self.region_sizes_ti,
                self.centroids_ti, self.n_points,
                self.monte_samples, self.monte_probability
            )

        move_points(
            self.points_ti, self.centroids_ti, float(self.mobility),
            float(xmin), float(xmax), float(ymin), float(ymax), self.n_points
        )
        self.points = self.points_ti.to_numpy()
        self._last_regions, self._last_vertices, self._last_sides = regions, vertices, sides_list
        return regions, vertices

    def get_stats(self, regions, vertices=None, sides_list=None):
        if sides_list is None:
            sides_list = self._compute_sides_vectorized(regions, vertices)
        counts = {}
        for s in sides_list:
            if s >= 3:
                counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0, 0, 0.0, 0.0
        hex_count = counts.get(6, 0)
        return hex_count, total, hex_count / total, sum(k * v for k, v in counts.items()) / total


class VoronoiSimulation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.single_step)
        self.is_running = False
        self.init_ui()
        QTimer.singleShot(1, self.reset_simulation)

    def init_ui(self):
        self.setWindowTitle("Voronoi + Lloyd 松弛 (Taichi + OpenGL)")
        self.setGeometry(100, 100, 840, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # 左侧控制面板
        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)

        # 依次添加各个分组
        panel_layout.addWidget(self._create_control_group())
        panel_layout.addWidget(self._create_performance_group())
        panel_layout.addWidget(self._create_viz_group())
        panel_layout.addWidget(self._create_parameter_group())
        panel_layout.addWidget(self._create_monte_group())
        panel_layout.addWidget(self._create_stats_group())
        panel_layout.addStretch()

        # 右侧绘图区
        self.voronoi_widget = VoronoiWidget()
        main_layout.addWidget(panel)
        main_layout.addWidget(self.voronoi_widget)

    def _create_control_group(self):
        group = QGroupBox("控制")
        layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.step_btn = QPushButton("单步")
        self.reset_btn = QPushButton("重置")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.step_btn.clicked.connect(self.single_step)
        self.reset_btn.clicked.connect(self.reset_simulation)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.step_btn)
        layout.addWidget(self.reset_btn)
        group.setLayout(layout)
        return group

    def _create_performance_group(self):
        group = QGroupBox("性能设置")
        layout = QFormLayout()
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 20)
        self.steps_spin.setValue(1)
        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(1, 500)
        self.freq_spin.setValue(1)
        layout.addRow("每帧迭代:", self.steps_spin)
        layout.addRow("更新时间(ms):", self.freq_spin)
        group.setLayout(layout)
        return group

    def _create_viz_group(self):
        group = QGroupBox("显示选项")
        layout = QVBoxLayout()
        self.show_pts_check = QCheckBox("显示点")
        self.show_vor_check = QCheckBox("显示 Voronoi 图")
        self.show_pts_check.setChecked(True)
        self.show_vor_check.setChecked(True)
        self.show_pts_check.toggled.connect(self._update_viz_options)
        self.show_vor_check.toggled.connect(self._update_viz_options)
        layout.addWidget(self.show_pts_check)
        layout.addWidget(self.show_vor_check)
        group.setLayout(layout)
        return group

    def _create_parameter_group(self):
        group = QGroupBox("算法参数")
        layout = QFormLayout()
        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 10000)
        self.points_spin.setValue(2000)
        self.mob_spin = QDoubleSpinBox()
        self.mob_spin.setRange(0.01, 2.0)
        self.mob_spin.setValue(1.5)
        self.mob_spin.setSingleStep(0.05)
        self.mob_spin.setDecimals(3)
        layout.addRow("点数(10-10000):", self.points_spin)
        layout.addRow("移动性(0.01-2.0):", self.mob_spin)
        group.setLayout(layout)
        return group

    def _create_monte_group(self):
        group = QGroupBox("蒙特卡洛质心估计")
        layout = QFormLayout()
        self.monte_enable_check = QCheckBox("启用蒙特卡洛")
        self.monte_enable_check.toggled.connect(self._on_monte_toggled)
        self.monte_samples_spin = QSpinBox()
        self.monte_samples_spin.setRange(1, 2000000)
        self.monte_samples_spin.setValue(50)
        self.monte_prob_spin = QDoubleSpinBox()
        self.monte_prob_spin.setRange(0.0, 1.0)
        self.monte_prob_spin.setSingleStep(0.01)
        self.monte_prob_spin.setDecimals(3)
        self.monte_prob_spin.setValue(0.2)
        self.monte_samples_spin.valueChanged.connect(self._on_monte_param_changed)
        self.monte_prob_spin.valueChanged.connect(self._on_monte_param_changed)
        layout.addRow(self.monte_enable_check)
        layout.addRow("蒙卡次数:", self.monte_samples_spin)
        layout.addRow("启用概率:", self.monte_prob_spin)
        group.setLayout(layout)
        return group

    def _create_stats_group(self):
        group = QGroupBox("统计信息")
        layout = QVBoxLayout()
        self.status_label1 = QLabel("就绪")
        self.status_label2 = QLabel("")
        layout.addWidget(self.status_label1)
        layout.addWidget(self.status_label2)
        group.setLayout(layout)
        return group

    # ---------- 以下方法与原代码完全一致 ----------
    def init_simulation(self):
        self.sim = TaichiHoneycombCVT(
            n_points=self.points_spin.value(),
            mobility=self.mob_spin.value(),
            monte_enabled=self.monte_enable_check.isChecked(),
            monte_samples=self.monte_samples_spin.value(),
            monte_probability=self.monte_prob_spin.value()
        )
        self.sim.step_count = 0
        self.sim.step()
        self._refresh_display()

    def _refresh_display(self):
        if self.sim is None:
            return
        show_vor = self.show_vor_check.isChecked()
        self.voronoi_widget.set_show_options(self.show_pts_check.isChecked(), show_vor)

        regions = self.sim._last_regions
        vertices = self.sim._last_vertices
        sides = self.sim._last_sides
        if regions is not None:
            hex_cnt, total, hex_frac, avg = self.sim.get_stats(regions, vertices, sides)
            self.status_label1.setText(f"迭代: {self.sim.step_count}, 点数: {self.sim.n_points}")
            self.status_label2.setText(f"六边形: {hex_frac:.1%}, 平均边: {avg:.2f}")
            self.voronoi_widget.set_data(self.sim.points, regions, vertices, sides)
        else:
            self.voronoi_widget.set_data(self.sim.points, [], np.empty((0, 2)), [])

    def _update_viz_options(self):
        self.voronoi_widget.set_show_options(
            self.show_pts_check.isChecked(),
            self.show_vor_check.isChecked()
        )

    def _on_monte_toggled(self, checked):
        if self.sim:
            self.sim.monte_enabled = checked

    def _on_monte_param_changed(self):
        if self.sim:
            self.sim.monte_samples = self.monte_samples_spin.value()
            self.sim.monte_probability = self.monte_prob_spin.value()

    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText("开始")
        else:
            self.timer.start(self.freq_spin.value())
            self.start_btn.setText("暂停")
        self.is_running = not self.is_running

    def single_step(self):
        if self.sim:
            for _ in range(self.steps_spin.value()):
                self.sim.step()
            self._refresh_display()

    def reset_simulation(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("开始")
        self.init_simulation()

    def closeEvent(self, event):
        self.timer.stop()
        ti.reset()
        event.accept()

if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = VoronoiSimulation()
    window.show()
    sys.exit(app.exec())