from custom_import import *

from scipy.spatial import Voronoi

ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)

_EPS = 1e-20

# ── Voronoi 显示组件 ──────────────────────
class VoronoiWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.array([])
        self.bbox = (0, 1, 0, 1)
        self.show_points = True
        self.show_voronoi = True
        self.setMinimumSize(600, 600)
        self._cached_polygons = []
        self._cached_brushes = []
        self._cached_pen = QPen(QColor(205, 133, 0), 1.5)
        self._cached_points = []
        self._cached_need_update = True
        self._region_sides = None

    def set_data(self, points, regions, vertices, region_sides=None):
        self.points = points
        self._regions = regions
        self._vertices = vertices
        self._region_sides = region_sides
        self._cached_need_update = True
        self.update()

    def set_show_options(self, show_points, show_voronoi):
        self.show_points = show_points
        self.show_voronoi = show_voronoi
        self.update()

    def resizeEvent(self, event):
        self._cached_need_update = True
        super().resizeEvent(event)

    def _update_cache(self):
        if len(self.points) == 0:
            self._cached_polygons.clear()
            self._cached_brushes.clear()
            self._cached_points.clear()
            return

        width, height = self.width(), self.height()
        xmin, xmax, ymin, ymax = self.bbox
        scale = min(width / (xmax - xmin), height / (ymax - ymin))
        ox = (width - (xmax - xmin) * scale) / 2
        oy = (height - (ymax - ymin) * scale) / 2

        def transform(x, y):
            return ox + (x - xmin) * scale, oy + (ymax - y) * scale

        self._cached_points = [QPointF(*transform(p[0], p[1])) for p in self.points]

        self._cached_polygons.clear()
        self._cached_brushes.clear()

        if self.show_voronoi and self._regions:
            sides_list = self._region_sides
            for i, region in enumerate(self._regions):
                polygon = QPolygonF()
                for vi in region:
                    if 0 <= vi < len(self._vertices):
                        v = self._vertices[vi]
                        polygon.append(QPointF(*transform(v[0], v[1])))
                if polygon.size() < 3:
                    continue
                sides = sides_list[i] if sides_list is not None else len(region)
                brush = QBrush(QColor(255, 223, 89)) if sides == 6 else QBrush(QColor(144, 238, 144, 180))
                self._cached_polygons.append(polygon)
                self._cached_brushes.append(brush)

        self._cached_need_update = False

    def paintEvent(self, event):
        if self._cached_need_update:
            self._update_cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.show_voronoi:
            for polygon, brush in zip(self._cached_polygons, self._cached_brushes):
                painter.setBrush(brush)
                painter.setPen(self._cached_pen)
                painter.drawPolygon(polygon)
        if self.show_points:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.setBrush(QBrush(QColor(100, 100, 100)))
            for pt in self._cached_points:
                painter.drawEllipse(int(pt.x() - 1), int(pt.y() - 1), 2, 2)


# ── Taichi 核心计算 ────────────────────────
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
        n_vertices = region_sizes[i]
        if n_vertices < 3:
            centroids[i, 0] = 0.0
            centroids[i, 1] = 0.0
            continue
        area = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(start, start + n_vertices - 1):
            x1 = polygons[j, 0]
            y1 = polygons[j, 1]
            x2 = polygons[j + 1, 0]
            y2 = polygons[j + 1, 1]
            cross = x1 * y2 - x2 * y1
            area += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        # 闭合边
        x1 = polygons[start + n_vertices - 1, 0]
        y1 = polygons[start + n_vertices - 1, 1]
        x2 = polygons[start, 0]
        y2 = polygons[start, 1]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
        if ti.abs(area) < 1e-12:
            avg_x = 0.0
            avg_y = 0.0
            for j in range(start, start + n_vertices):
                avg_x += polygons[j, 0]
                avg_y += polygons[j, 1]
            centroids[i, 0] = avg_x / n_vertices
            centroids[i, 1] = avg_y / n_vertices
        else:
            area *= 0.5
            centroids[i, 0] = cx / (6.0 * area)
            centroids[i, 1] = cy / (6.0 * area)


@ti.kernel
def move_points(
    points: ti.types.ndarray(),
    centroids: ti.types.ndarray(),
    mobility: ti.f32,
    xmin: ti.f32, xmax: ti.f32,
    ymin: ti.f32, ymax: ti.f32,
    n_points: ti.i32
):
    ti.loop_config(parallelize=8)
    for i in range(n_points):
        dx = (centroids[i, 0] - points[i, 0]) * mobility
        dy = (centroids[i, 1] - points[i, 1]) * mobility
        points[i, 0] += dx
        points[i, 1] += dy
        points[i, 0] = ti.max(xmin + 1e-6, ti.min(xmax - 1e-6, points[i, 0]))
        points[i, 1] = ti.max(ymin + 1e-6, ti.min(ymax - 1e-6, points[i, 1]))


# ── 模拟核心类 ────────────────────────────
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
        max_vertices_per_region = 32
        self.total_vertices = self.n_points * max_vertices_per_region
        self.points_ti = ti.ndarray(dtype=ti.f32, shape=(self.n_points, 2))
        self.centroids_ti = ti.ndarray(dtype=ti.f32, shape=(self.n_points, 2))
        self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(self.total_vertices, 2))
        self.region_offsets_ti = ti.ndarray(dtype=ti.i32, shape=(self.n_points,))
        self.region_sizes_ti = ti.ndarray(dtype=ti.i32, shape=(self.n_points,))
        self.points_ti.from_numpy(self.points.astype(np.float32))
        self.polygons_np = np.zeros((self.total_vertices, 2), dtype=np.float32)
        self.region_offsets_np = np.zeros(self.n_points, dtype=np.int32)
        self.region_sizes_np = np.zeros(self.n_points, dtype=np.int32)

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

    # ── 边界检查辅助 ──────────────────────
    def _is_outside_bbox(self, poly):
        """检测多边形顶点是否超出边界框"""
        return (np.any(poly[:, 0] < self.xmin - 1e-9) or
                np.any(poly[:, 0] > self.xmax + 1e-9) or
                np.any(poly[:, 1] < self.ymin - 1e-9) or
                np.any(poly[:, 1] > self.ymax + 1e-9))

    def _compute_sides(self, regions, vertices):
        """计算所有区域的边数（考虑边界）"""
        sides_list = []
        for region in regions:
            if len(region) < 3:
                sides_list.append(0)
                continue
            poly = vertices[region]
            sides = len(region) + (1 if self._is_outside_bbox(poly) else 0)
            sides_list.append(sides)
        return sides_list

    # ── 蒙特卡洛质心 ──────────────────────
    def _monte_carlo_centroid(self, poly_closed, samples):
        if samples <= 0:
            return np.array([poly_closed[:-1, 0].mean(), poly_closed[:-1, 1].mean()], dtype=np.float32)
        minx, maxx = poly_closed[:-1, 0].min(), poly_closed[:-1, 0].max()
        miny, maxy = poly_closed[:-1, 1].min(), poly_closed[:-1, 1].max()
        xs = self.rng.rand(samples) * (maxx - minx) + minx
        ys = self.rng.rand(samples) * (maxy - miny) + miny
        pts = np.column_stack([xs, ys])
        try:
            from matplotlib.path import Path
            mask = Path(poly_closed).contains_points(pts)
        except ImportError:
            mask = self._points_in_polygon_vectorized(pts, poly_closed)
        if np.any(mask):
            sel = pts[mask]
            return np.array([sel[:, 0].mean(), sel[:, 1].mean()], dtype=np.float32)
        return np.array([poly_closed[:-1, 0].mean(), poly_closed[:-1, 1].mean()], dtype=np.float32)

    @staticmethod
    def _points_in_polygon_vectorized(pts, poly):
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(poly) - 1
        inside = np.zeros(len(pts), dtype=bool)
        for i in range(n):
            j = i + 1
            xi, yi = poly[i, 0], poly[i, 1]
            xj, yj = poly[j, 0], poly[j, 1]
            intersect = ((yi > ys) != (yj > ys)) & (
                xs < (xj - xi) * (ys - yi) / (yj - yi + _EPS) + xi
            )
            inside ^= intersect
        return inside

    # ── 单步迭代 ──────────────────────────
    def step(self):
        self.step_count += 1
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        points = self.points
        mobility = self.mobility
        monte_enabled = self.monte_enabled
        monte_samples = self.monte_samples
        monte_probability = self.monte_probability
        n_points = self.n_points

        # 是否需要重新计算 Voronoi
        need_recompute = (self._cached_regions is None or
                          self._cached_vertices is None)
        if not need_recompute and self._prev_points is not None:
            disp2 = np.max(np.sum((points - self._prev_points)**2, axis=1))
            if disp2 > self._recompute_tol ** 2:
                need_recompute = True

        if need_recompute:
            vor = Voronoi(points)
            regions, vertices = self._finite_voronoi(vor)
            self._cached_regions = regions
            self._cached_vertices = vertices
            self._prev_points = points.copy()
        else:
            regions = self._cached_regions
            vertices = self._cached_vertices

        # 一次性填充多边形缓冲区，并计算 sides
        sides_list = self._compute_sides(regions, vertices)
        idx = 0
        region_sizes = self.region_sizes_np
        region_offsets = self.region_offsets_np
        polygons_np = self.polygons_np
        for i, region in enumerate(regions):
            nv = len(region)
            if nv < 3:
                region_sizes[i] = 0
                continue
            poly = vertices[region]
            poly_closed = np.vstack([poly, poly[0]]) if not np.allclose(poly[0], poly[-1]) else poly
            nv_closed = len(poly_closed)
            if idx + nv_closed > self.total_vertices:
                self._resize_buffers(idx + nv_closed)
                polygons_np = self.polygons_np
            region_sizes[i] = nv_closed
            region_offsets[i] = idx
            polygons_np[idx:idx + nv_closed] = poly_closed
            idx += nv_closed

        self.polygons_ti.from_numpy(polygons_np)
        self.region_offsets_ti.from_numpy(region_offsets)
        self.region_sizes_ti.from_numpy(region_sizes)

        compute_centroids_shoelace(
            self.polygons_ti, self.region_offsets_ti, self.region_sizes_ti,
            self.centroids_ti, n_points
        )

        # 蒙特卡洛修正
        if monte_enabled and monte_samples > 0 and monte_probability > 0:
            centroids_np = self.centroids_ti.to_numpy()
            rand_vals = self.rng.rand(n_points) if monte_probability < 1.0 else None
            for i in range(n_points):
                if region_sizes[i] < 3:
                    continue
                if rand_vals is not None and rand_vals[i] >= monte_probability:
                    continue
                start = region_offsets[i]
                end = start + region_sizes[i]
                poly_closed = polygons_np[start:end]
                try:
                    centroids_np[i] = self._monte_carlo_centroid(poly_closed, monte_samples)
                except Exception:
                    pass
            self.centroids_ti.from_numpy(centroids_np)

        move_points(
            self.points_ti, self.centroids_ti, float(mobility),
            float(xmin), float(xmax), float(ymin), float(ymax), n_points
        )
        self.points = self.points_ti.to_numpy()
        self._last_regions = regions
        self._last_vertices = vertices
        self._last_sides = sides_list
        return regions, vertices

    def _resize_buffers(self, required_vertices):
        new_total = int(required_vertices * 1.5) + 1000
        self.total_vertices = new_total
        self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(new_total, 2))
        self.polygons_np = np.zeros((new_total, 2), dtype=np.float32)
        self._cached_regions = None
        self._cached_vertices = None

    def get_stats(self, regions, vertices=None, sides_list=None):
        """统计六边形比例与平均边数"""
        if sides_list is None:
            sides_list = self._compute_sides(regions, vertices)
        counts = {}
        for s in sides_list:
            if s >= 3:
                counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0, 0, 0.0, 0.0
        hex_count = counts.get(6, 0)
        hex_frac = hex_count / total
        avg_sides = sum(k * v for k, v in counts.items()) / total
        return hex_count, total, hex_frac, avg_sides


# ── 主界面 ────────────────────────────────
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
        self.setWindowTitle("Voronoi + Lloyd 松弛 (Taichi加速)")
        self.setGeometry(100, 100, 840, 600)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # ── 控制面板 ──
        panel = QWidget()
        panel.setFixedWidth(280)
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)

        # 控制按钮
        btn_group = QGroupBox("控制")
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.step_btn = QPushButton("单步")
        self.step_btn.clicked.connect(self.single_step)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.step_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_group.setLayout(btn_layout)
        panel_layout.addWidget(btn_group)

        # 性能设置
        perf_group = QGroupBox("性能设置")
        perf_layout = QVBoxLayout()
        self.steps_spin = self._add_spin(perf_layout, "每帧迭代:", 1, 20, 1)
        self.freq_spin = self._add_spin(perf_layout, "更新时间(ms):", 1, 500, 1)
        perf_group.setLayout(perf_layout)
        panel_layout.addWidget(perf_group)

        # 显示选项
        viz_group = QGroupBox("显示选项")
        viz_layout = QVBoxLayout()
        self.show_pts_check = QCheckBox("显示点")
        self.show_pts_check.setChecked(True)
        self.show_pts_check.toggled.connect(self._update_viz_options)
        self.show_vor_check = QCheckBox("显示 Voronoi 图")
        self.show_vor_check.setChecked(True)
        self.show_vor_check.toggled.connect(self._update_viz_options)
        viz_layout.addWidget(self.show_pts_check)
        viz_layout.addWidget(self.show_vor_check)
        viz_group.setLayout(viz_layout)
        panel_layout.addWidget(viz_group)

        # 算法参数
        param_group = QGroupBox("算法参数")
        param_layout = QVBoxLayout()
        self.points_spin = self._add_spin(param_layout, "点数(10-10000):", 10, 10000, 2000)
        self.mob_spin = self._add_double_spin(param_layout, "移动性(0.01-2.0):", 0.01, 2.0, 1.5, 0.05)
        param_group.setLayout(param_layout)
        panel_layout.addWidget(param_group)

        # 蒙特卡洛
        monte_group = QGroupBox("蒙特卡洛质心估计")
        monte_layout = QVBoxLayout()
        self.monte_enable_check = QCheckBox("启用蒙特卡洛")
        self.monte_enable_check.toggled.connect(self._on_monte_toggled)
        monte_layout.addWidget(self.monte_enable_check)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("蒙卡次数:"))
        self.monte_samples_spin = QSpinBox()
        self.monte_samples_spin.setRange(1, 2000000)
        self.monte_samples_spin.setValue(50)
        self.monte_samples_spin.valueChanged.connect(self._on_monte_param_changed)
        row1.addWidget(self.monte_samples_spin)
        row1.addStretch()
        monte_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("启用概率:"))
        self.monte_prob_spin = QDoubleSpinBox()
        self.monte_prob_spin.setRange(0.0, 1.0)
        self.monte_prob_spin.setSingleStep(0.01)
        self.monte_prob_spin.setDecimals(3)
        self.monte_prob_spin.setValue(0.2)
        self.monte_prob_spin.valueChanged.connect(self._on_monte_param_changed)
        row2.addWidget(self.monte_prob_spin)
        row2.addStretch()
        monte_layout.addLayout(row2)
        monte_group.setLayout(monte_layout)
        panel_layout.addWidget(monte_group)

        # 统计信息
        stat_group = QGroupBox("统计信息")
        stat_layout = QVBoxLayout()
        self.status_label1 = QLabel("就绪")
        self.status_label2 = QLabel("")
        stat_layout.addWidget(self.status_label1)
        stat_layout.addWidget(self.status_label2)
        stat_group.setLayout(stat_layout)
        panel_layout.addWidget(stat_group)
        panel_layout.addStretch()

        # Voronoi 显示
        self.voronoi_widget = VoronoiWidget()
        main_layout.addWidget(panel)
        main_layout.addWidget(self.voronoi_widget)

    # ── 控件构建辅助 ──
    def _add_spin(self, parent_layout, label, min_val, max_val, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        row.addWidget(spin)
        row.addStretch()
        parent_layout.addLayout(row)
        return spin

    def _add_double_spin(self, parent_layout, label, min_val, max_val, default, step, decimals=3):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        row.addWidget(spin)
        row.addStretch()
        parent_layout.addLayout(row)
        return spin

    # ── 模拟管理 ──
    def init_simulation(self):
        params = {
            'n_points': self.points_spin.value(),
            'mobility': self.mob_spin.value(),
            'monte_enabled': self.monte_enable_check.isChecked(),
            'monte_samples': self.monte_samples_spin.value(),
            'monte_probability': self.monte_prob_spin.value()
        }
        self.sim = TaichiHoneycombCVT(**params)
        self.sim.step_count = 0
        # 初始化显示 (force compute)
        self.sim.step()
        self._refresh_display()

    def _refresh_display(self):
        """统一更新统计与显示"""
        sim = self.sim
        if sim is None:
            return
        regions = sim._last_regions
        vertices = sim._last_vertices
        sides = sim._last_sides
        if regions is not None:
            hex_cnt, total, hex_frac, avg = sim.get_stats(regions, vertices, sides)
            self.status_label1.setText(f"迭代: {sim.step_count}, 点数: {sim.n_points}")
            self.status_label2.setText(f"六边形: {hex_frac:.1%}, 平均边: {avg:.2f}")
            self.voronoi_widget.set_data(sim.points, regions, vertices, sides)

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

    # ── 控制操作 ──
    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.start_btn.setText("开始")
            self.is_running = False
        else:
            self.timer.start(self.freq_spin.value())
            self.start_btn.setText("暂停")
            self.is_running = True

    def single_step(self):
        if self.sim:
            for _ in range(self.steps_spin.value()):
                self.sim.step()
            self._refresh_display()

    def reset_simulation(self):
        if self.timer:
            self.timer.stop()
            self.is_running = False
            self.start_btn.setText("开始")
        self.init_simulation()

    def closeEvent(self, event):
        self.timer.stop()
        ti.reset()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoronoiSimulation()
    window.show()
    sys.exit(app.exec())