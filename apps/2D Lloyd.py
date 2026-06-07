from custom_import import *

from scipy.spatial import Voronoi

ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)
# 全局常量，避免重复创建
_EPS = 1e-20
# ========== VoronoiWidget（支持预计算颜色） ==========
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
        self._cached_points = [
            QPointF(*transform(p[0], p[1])) for p in self.points
        ]
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
                # 直接使用预计算的 sides，已由外部保证长度匹配
                sides = sides_list[i] if sides_list is not None else len(region)
                if sides == 6:
                    brush = QBrush(QColor(255, 223, 89))
                else:
                    brush = QBrush(QColor(144, 238, 144, 180))
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
# ========== Taichi 核心计算 ==========
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
# ========== TaichiHoneycombCVT（极致优化） ==========
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
        # 提高到 32，几乎永不触发动态扩容
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
        # 用 NumPy 数组管理所有顶点，避免频繁的 Python list 操作
        verts_np = vor.vertices  # (n_verts, 2) float64
        extra_verts = []  # 收集新顶点，最后一次性 vstack
        new_regions = []
        for p in range(len(vor.points)):
            region_idx = vor.point_region[p]
            region = vor.regions[region_idx]
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
            # 按角度排序
            if len(new_region) > 2:
                indices = new_region
                # 收集顶点坐标（可能来自原 verts_np 或 extra_verts）
                if extra_verts:
                    all_verts = np.vstack([verts_np] + [np.array(extra_verts)])
                else:
                    all_verts = verts_np
                vs = all_verts[indices]
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_region = [indices[i] for i in np.argsort(angles)]
            new_regions.append(new_region)
        if extra_verts:
            all_vertices = np.vstack([verts_np] + [np.array(extra_verts)])
        else:
            all_vertices = verts_np
        return new_regions, np.asarray(all_vertices, dtype=np.float32)
    def _monte_carlo_centroid(self, poly_closed, samples):
        # poly_closed 是闭合多边形 (n+1) 个点
        if samples <= 0:
            return np.array([poly_closed[:-1, 0].mean(),
                             poly_closed[:-1, 1].mean()], dtype=np.float32)
        minx, maxx = poly_closed[:-1, 0].min(), poly_closed[:-1, 0].max()
        miny, maxy = poly_closed[:-1, 1].min(), poly_closed[:-1, 1].max()
        xs = self.rng.rand(samples) * (maxx - minx) + minx
        ys = self.rng.rand(samples) * (maxy - miny) + miny
        pts = np.column_stack([xs, ys])
        try:
            from matplotlib.path import Path
            path = Path(poly_closed)
            mask = path.contains_points(pts)
        except ImportError:
            mask = self._points_in_polygon_vectorized(pts, poly_closed)
        if np.any(mask):
            sel = pts[mask]
            return np.array([sel[:, 0].mean(), sel[:, 1].mean()], dtype=np.float32)
        else:
            return np.array([poly_closed[:-1, 0].mean(),
                             poly_closed[:-1, 1].mean()], dtype=np.float32)
    @staticmethod
    def _points_in_polygon_vectorized(pts, poly):
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(poly) - 1  # 闭合多边形
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
    def step(self):
        self.step_count += 1
        # 绑定局部变量
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        points = self.points
        mobility = self.mobility
        monte_enabled = self.monte_enabled
        monte_samples = self.monte_samples
        monte_probability = self.monte_probability
        n_points = self.n_points
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
        # 一次性构建闭合多边形、填充缓冲区、计算 sides
        idx = 0
        sides_list = [0] * n_points  # 预分配
        region_sizes = self.region_sizes_np
        region_offsets = self.region_offsets_np
        polygons_np = self.polygons_np
        for i, region in enumerate(regions):
            nv = len(region)
            if nv < 3:
                region_sizes[i] = 0
                continue
            poly = vertices[region]  # 原始多边形 (nv, 2)
            # 构造闭合多边形
            if not np.allclose(poly[0], poly[-1]):
                poly_closed = np.vstack([poly, poly[0]])
            else:
                poly_closed = poly
            nv_closed = len(poly_closed)
            # 动态扩容（概率极低）
            if idx + nv_closed > self.total_vertices:
                self._resize_buffers(idx + nv_closed)
                polygons_np = self.polygons_np  # 更新引用
            region_sizes[i] = nv_closed
            region_offsets[i] = idx
            polygons_np[idx:idx + nv_closed] = poly_closed
            idx += nv_closed
            # 计算 sides（无 try/except）
            sides = nv
            if (np.any(poly[:, 0] < xmin - 1e-9) or
                np.any(poly[:, 0] > xmax + 1e-9) or
                np.any(poly[:, 1] < ymin - 1e-9) or
                np.any(poly[:, 1] > ymax + 1e-9)):
                sides += 1
            sides_list[i] = sides
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
            rng = self.rng
            # 预生成随机数，减少函数调用开销
            if monte_probability < 1.0:
                rand_vals = rng.rand(n_points)
            for i in range(n_points):
                if region_sizes[i] < 3:
                    continue
                if monte_probability < 1.0 and rand_vals[i] >= monte_probability:
                    continue
                start = region_offsets[i]
                end = start + region_sizes[i]
                poly_closed = polygons_np[start:end]
                try:
                    mc = self._monte_carlo_centroid(poly_closed, monte_samples)
                    centroids_np[i] = mc
                except Exception:
                    pass
            self.centroids_ti.from_numpy(centroids_np)
        move_points(
            self.points_ti, self.centroids_ti, float(mobility),
            float(xmin), float(xmax),
            float(ymin), float(ymax),
            n_points
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
        """若提供 sides_list 则直接使用，否则回退到旧逻辑（仅在无预计算时使用）"""
        if sides_list is not None:
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
        # 回退逻辑（保留兼容性）
        counts = {}
        verts = vertices if vertices is not None else self._cached_vertices
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        for i, region in enumerate(regions):
            if len(region) >= 3:
                sides = len(region)
                if verts is not None:
                    try:
                        coords = np.asarray(verts)[region]
                        if coords.size and (
                            np.any(coords[:, 0] < xmin - 1e-9) or
                            np.any(coords[:, 0] > xmax + 1e-9) or
                            np.any(coords[:, 1] < ymin - 1e-9) or
                            np.any(coords[:, 1] > ymax + 1e-9)
                        ):
                            sides += 1
                    except Exception:
                        pass
                counts[sides] = counts.get(sides, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0, 0, 0.0, 0.0
        hex_count = counts.get(6, 0)
        hex_frac = hex_count / total
        avg_sides = sum(k * v for k, v in counts.items()) / total
        return hex_count, total, hex_frac, avg_sides
# ========== 主界面（保持不变） ==========
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        control_panel = QWidget()
        control_panel.setFixedWidth(280)
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        # 控制按钮
        button_group = QGroupBox("控制")
        button_layout = QHBoxLayout()
        button_group.setLayout(button_layout)
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.toggle_simulation)
        button_layout.addWidget(self.start_button)
        self.step_button = QPushButton("单步")
        self.step_button.clicked.connect(self.single_step)
        button_layout.addWidget(self.step_button)
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)
        control_layout.addWidget(button_group)
        # 性能设置
        perf_group = QGroupBox("性能设置")
        perf_layout = QVBoxLayout()
        perf_group.setLayout(perf_layout)
        self.steps_spin = self._create_spinbox("每帧迭代:", 1, 20, 1)
        self.freq_spin = self._create_spinbox("更新时间(ms):", 1, 500, 1)
        perf_layout.addLayout(self.steps_spin)
        perf_layout.addLayout(self.freq_spin)
        control_layout.addWidget(perf_group)
        # 显示选项
        viz_group = QGroupBox("显示选项")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        self.show_points_check = QCheckBox("显示点")
        self.show_points_check.setChecked(True)
        self.show_points_check.toggled.connect(self.update_viz_options)
        self.show_voronoi_check = QCheckBox("显示 Voronoi 图")
        self.show_voronoi_check.setChecked(True)
        self.show_voronoi_check.toggled.connect(self.update_viz_options)
        viz_layout.addWidget(self.show_points_check)
        viz_layout.addWidget(self.show_voronoi_check)
        control_layout.addWidget(viz_group)
        # 算法参数
        params_group = QGroupBox("算法参数")
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)
        self.points_spin = self._create_spinbox("点数(10-10000):", 10, 10000, 2000)
        self.mob_spin = self._create_double_spinbox("移动性(0.01-2.0):", 0.01, 2.0, 1.5, 0.05)
        params_layout.addLayout(self.points_spin)
        params_layout.addLayout(self.mob_spin)
        control_layout.addWidget(params_group)
        # 蒙特卡洛设置
        monte_group = QGroupBox("蒙特卡洛质心估计")
        monte_layout = QVBoxLayout()
        monte_group.setLayout(monte_layout)
        self.enable_monte_check = QCheckBox("启用蒙特卡洛")
        self.enable_monte_check.setChecked(False)
        self.enable_monte_check.toggled.connect(self.update_monte_enabled)
        monte_layout.addWidget(self.enable_monte_check)
        layout_samples = QHBoxLayout()
        layout_samples.addWidget(QLabel("蒙卡次数:"))
        self.monte_samples_spin = QSpinBox()
        self.monte_samples_spin.setRange(1, 2000000)
        self.monte_samples_spin.setValue(50)
        self.monte_samples_spin.setFixedWidth(100)
        self.monte_samples_spin.valueChanged.connect(self.update_monte_samples)
        layout_samples.addWidget(self.monte_samples_spin)
        layout_samples.addStretch()
        monte_layout.addLayout(layout_samples)
        layout_mprob = QHBoxLayout()
        layout_mprob.addWidget(QLabel("启用概率:"))
        self.monte_prob_spin = QDoubleSpinBox()
        self.monte_prob_spin.setRange(0.0, 1.0)
        self.monte_prob_spin.setSingleStep(0.01)
        self.monte_prob_spin.setDecimals(3)
        self.monte_prob_spin.setValue(0.2)
        self.monte_prob_spin.setFixedWidth(100)
        self.monte_prob_spin.valueChanged.connect(self.update_monte_probability)
        layout_mprob.addWidget(self.monte_prob_spin)
        layout_mprob.addStretch()
        monte_layout.addLayout(layout_mprob)
        control_layout.addWidget(monte_group)
        # 统计信息
        status_group = QGroupBox("统计信息")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        self.status_label1 = QLabel("就绪")
        self.status_label2 = QLabel("")
        status_layout.addWidget(self.status_label1)
        status_layout.addWidget(self.status_label2)
        control_layout.addWidget(status_group)
        control_layout.addStretch()
        self.voronoi_widget = VoronoiWidget()
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.voronoi_widget)
    def _create_spinbox(self, label, min_val, max_val, default):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setFixedWidth(80)
        layout.addWidget(spin)
        layout.addStretch()
        return layout
    def _create_double_spinbox(self, label, min_val, max_val, default, step, decimals=3):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setFixedWidth(80)
        layout.addWidget(spin)
        layout.addStretch()
        return layout
    def init_simulation(self):
        params = {
            'n_points': self._get_spin_value(self.points_spin),
            'mobility': self._get_double_spin_value(self.mob_spin),
            'monte_enabled': self.enable_monte_check.isChecked(),
            'monte_samples': self.monte_samples_spin.value(),
            'monte_probability': self.monte_prob_spin.value()
        }
        self.sim = TaichiHoneycombCVT(**params)
        self.sim.step_count = 0
        self.update_display(initial=True)
    def update_monte_enabled(self, enabled):
        if self.sim:
            self.sim.monte_enabled = bool(enabled)
    def update_monte_samples(self, value):
        if self.sim:
            try:
                self.sim.monte_samples = int(value)
            except Exception:
                pass
    def update_monte_probability(self, value):
        if self.sim:
            try:
                self.sim.monte_probability = float(value)
            except Exception:
                pass
    def _get_spin_value(self, layout):
        return layout.itemAt(1).widget().value()
    def _get_double_spin_value(self, layout):
        return layout.itemAt(1).widget().value()
    def update_display(self, initial=False):
        if self.sim is None:
            return
        if initial and self.sim._last_regions is None:
            vor = Voronoi(self.sim.points)
            regions, vertices = self.sim._finite_voronoi(vor)
            self.sim._last_regions = regions
            self.sim._last_vertices = vertices
            # 计算初始 sides
            sides_list = []
            for region in regions:
                if len(region) < 3:
                    sides_list.append(0)
                    continue
                poly = vertices[region]
                sides = len(region)
                if (np.any(poly[:, 0] < self.sim.xmin - 1e-9) or
                    np.any(poly[:, 0] > self.sim.xmax + 1e-9) or
                    np.any(poly[:, 1] < self.sim.ymin - 1e-9) or
                    np.any(poly[:, 1] > self.sim.ymax + 1e-9)):
                    sides += 1
                sides_list.append(sides)
            self.sim._last_sides = sides_list
        regions = self.sim._last_regions
        vertices = self.sim._last_vertices
        sides_list = self.sim._last_sides
        if regions is not None:
            hex_count, total, hex_frac, avg_sides = self.sim.get_stats(regions, vertices, sides_list)
            status1 = f"迭代: {self.sim.step_count}, 点数: {self.sim.n_points}"
            status2 = f"六边形: {hex_frac:.1%}, 平均边: {avg_sides:.2f}"
            self.status_label1.setText(status1)
            self.status_label2.setText(status2)
            self.voronoi_widget.set_data(self.sim.points, regions, vertices, sides_list)
    def update_viz_options(self):
        self.voronoi_widget.set_show_options(
            self.show_points_check.isChecked(),
            self.show_voronoi_check.isChecked()
        )
    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.start_button.setText("开始")
            self.is_running = False
        else:
            freq = self._get_spin_value(self.freq_spin)
            self.timer.start(freq)
            self.start_button.setText("暂停")
            self.is_running = True
    def single_step(self):
        if self.sim:
            steps = self._get_spin_value(self.steps_spin)
            for _ in range(steps):
                self.sim.step()
            regions = self.sim._last_regions
            vertices = self.sim._last_vertices
            sides_list = self.sim._last_sides
            self.voronoi_widget.set_data(self.sim.points, regions, vertices, sides_list)
            hex_count, total, hex_frac, avg_sides = self.sim.get_stats(regions, vertices, sides_list)
            status1 = f"迭代: {self.sim.step_count}, 点数: {self.sim.n_points}"
            status2 = f"六边形: {hex_frac:.1%}, 平均边: {avg_sides:.2f}"
            self.status_label1.setText(status1)
            self.status_label2.setText(status2)
    def reset_simulation(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.is_running = False
            if hasattr(self, 'start_button'):
                self.start_button.setText("开始")
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

