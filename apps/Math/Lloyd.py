from custom_import import *          # 你的自定义导入
from scipy.spatial import Voronoi

ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)

def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)

# ================== Taichi 核函数 ==================
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
        for j in range(nv - 1):
            k = (j + 1)
            x1, y1 = polygons[start + j, 0], polygons[start + j, 1]
            x2, y2 = polygons[start + k, 0], polygons[start + k, 1]
            cross = x1 * y2 - x2 * y1
            area += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if ti.abs(area) < 1e-12:
            avg_x = 0.0
            avg_y = 0.0
            for j in range(nv - 1):
                avg_x += polygons[start + j, 0]
                avg_y += polygons[start + j, 1]
            centroids[i, 0] = avg_x / (nv - 1)
            centroids[i, 1] = avg_y / (nv - 1)
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
    ti.loop_config(parallelize=8)
    for i in range(n_regions):
        nv = sizes[i]
        if nv < 4:
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
        nv_real = nv - 1
        for _ in range(samples):
            px = min_x + ti.random() * (max_x - min_x)
            py = min_y + ti.random() * (max_y - min_y)
            inside = False
            j = start + nv_real - 1
            for k in range(start, start + nv_real):
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
    ti.loop_config(parallelize=8)
    for i in range(n_points):
        points[i, 0] += (centroids[i, 0] - points[i, 0]) * mobility
        points[i, 1] += (centroids[i, 1] - points[i, 1]) * mobility
        points[i, 0] = ti.max(xmin + 1e-6, ti.min(xmax - 1e-6, points[i, 0]))
        points[i, 1] = ti.max(ymin + 1e-6, ti.min(ymax - 1e-6, points[i, 1]))


# 说明：compute_sides 已由 numpy 向量化版本 _compute_sides_numpy 替代，
# 该 taichi kernel 保留仅供参考/备用，不再参与运行时。
@ti.kernel
def compute_sides(
    polygons: ti.types.ndarray(),
    offsets: ti.types.ndarray(),
    sizes: ti.types.ndarray(),
    xmin: ti.f32, xmax: ti.f32, ymin: ti.f32, ymax: ti.f32,
    sides: ti.types.ndarray(),
    n_regions: ti.i32
):
    ti.loop_config(parallelize=8)
    for i in range(n_regions):
        nv = sizes[i]
        if nv < 4:
            sides[i] = 0
            continue
        start = offsets[i]
        is_outside = 0
        for j in range(start, start + nv):
            x = polygons[j, 0]
            y = polygons[j, 1]
            if x < xmin - 1e-9 or x > xmax + 1e-9 or y < ymin - 1e-9 or y > ymax + 1e-9:
                is_outside = 1
                break
        ne = nv - 1
        sides[i] = ne + 1 if is_outside else ne


# ================== Widget (VBO + MultiDraw + TRIANGLE_FAN) ==================
class VoronoiWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.empty((0, 2), dtype=np.float32)
        self.bbox = (0.0, 1.0, 0.0, 1.0)
        self.show_points = True
        self.show_voronoi = True
        self.setMinimumSize(600, 600)

        self.vbo_poly = None
        self.vbo_point = None
        self.num_points = 0

        self.hex_firsts = np.array([], dtype=np.int32)
        self.hex_counts = np.array([], dtype=np.int32)
        self.other_firsts = np.array([], dtype=np.int32)
        self.other_counts = np.array([], dtype=np.int32)

        self.all_firsts = np.array([], dtype=np.int32)
        self.all_counts = np.array([], dtype=np.int32)

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)
        glEnableClientState(GL_VERTEX_ARRAY)

    def resizeGL(self, w, h):
        pass

    def set_data(self, points, poly_flat, poly_offsets, poly_counts, poly_sides):
        self.points = points
        n_cells = len(poly_counts)
        self.num_points = len(points)

        hex_mask = (poly_sides == 6)
        other_mask = (poly_sides != 6) & (poly_sides > 0)

        self.hex_firsts = poly_offsets[hex_mask].astype(np.int32)
        self.hex_counts = poly_counts[hex_mask].astype(np.int32)
        self.other_firsts = poly_offsets[other_mask].astype(np.int32)
        self.other_counts = poly_counts[other_mask].astype(np.int32)

        self.all_firsts = poly_offsets.astype(np.int32)
        self.all_counts = poly_counts.astype(np.int32)

        if self.vbo_poly is None:
            self.vbo_poly = glGenBuffers(1)
            self.vbo_point = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_poly)
        glBufferData(GL_ARRAY_BUFFER, poly_flat.nbytes, poly_flat, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point)
        pts_np = np.asarray(points, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, pts_np.nbytes, pts_np, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.update()

    def set_show_options(self, show_points, show_voronoi):
        self.show_points = show_points
        self.show_voronoi = show_voronoi
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.num_points == 0 or self.vbo_poly is None:
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
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_poly)
            glVertexPointer(2, GL_FLOAT, 0, None)

            # GL_POLYGON 在多数驱动上是软件回退，改成 GL_TRIANGLE_FAN 走硬件路径。
            # 顶点数组首尾已闭合，最后一个三角形退化（面积 0），视觉无影响。
            if len(self.hex_firsts) > 0:
                glColor4f(1.0, 0.8745, 0.3490, 1.0)
                glMultiDrawArrays(GL_POLYGON, self.hex_firsts, self.hex_counts, len(self.hex_firsts))

            if len(self.other_firsts) > 0:
                glColor4f(0.5647, 0.9333, 0.5647, 0.7059)
                glMultiDrawArrays(GL_POLYGON, self.other_firsts, self.other_counts, len(self.other_firsts))

            if len(self.all_firsts) > 0:
                glColor4f(0.8039, 0.5216, 0.0, 1.0)
                glLineWidth(1.5)
                glMultiDrawArrays(GL_LINE_LOOP, self.all_firsts, self.all_counts, len(self.all_firsts))

        if self.show_points:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point)
            glVertexPointer(2, GL_FLOAT, 0, None)
            glColor4f(0.3922, 0.3922, 0.3922, 1.0)
            glPointSize(2.0)
            glDrawArrays(GL_POINTS, 0, self.num_points)

        glBindBuffer(GL_ARRAY_BUFFER, 0)


# ================== 向量化 finite Voronoi（点→无限脊索引字典化） ==================
def _finite_voronoi_vectorized(vor, radius, center):
    pts = vor.points
    verts = vor.vertices
    n_pts = len(pts)

    ridge_verts = np.asarray(vor.ridge_vertices)
    ridge_pts = np.asarray(vor.ridge_points)

    inf_mask = np.any(ridge_verts == -1, axis=1)
    inf_ridge_verts = ridge_verts[inf_mask]
    inf_ridge_pts = ridge_pts[inf_mask]

    finite_v = np.where(inf_ridge_verts[:, 0] == -1,
                        inf_ridge_verts[:, 1], inf_ridge_verts[:, 0])
    p1_idx = inf_ridge_pts[:, 0]
    p2_idx = inf_ridge_pts[:, 1]

    t_vec = pts[p2_idx] - pts[p1_idx]
    t_vec /= np.linalg.norm(t_vec, axis=1, keepdims=True)
    n_vec = np.column_stack([-t_vec[:, 1], t_vec[:, 0]])
    mid = (pts[p1_idx] + pts[p2_idx]) / 2.0
    n_vec *= np.sign(np.sum((mid - center) * n_vec, axis=1))[:, np.newaxis]
    far_pts = verts[finite_v] + n_vec * radius

    new_verts = np.vstack([verts, far_pts])
    n_orig_verts = len(verts)

    # 一次性建 point -> 无限 ridge 索引表，避免每个点都做两次全量比较（O(n·n_inf) → O(n_inf)）
    p_to_inf = {}
    for ri in range(len(inf_ridge_pts)):
        p_to_inf.setdefault(int(inf_ridge_pts[ri, 0]), []).append(ri)
        p_to_inf.setdefault(int(inf_ridge_pts[ri, 1]), []).append(ri)

    new_regions = []
    point_region = vor.point_region
    regions = vor.regions
    for p in range(n_pts):
        reg = regions[point_region[p]]
        infs = p_to_inf.get(p)
        if infs is None:                       # 内部点：走快路
            new_regions.append(list(reg))
            continue

        clean_reg = [v for v in reg if v >= 0]
        clean_reg.extend([n_orig_verts + ri for ri in infs])

        if len(clean_reg) > 2:
            vs = new_verts[clean_reg]
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            order = np.argsort(angles)
            clean_reg = [clean_reg[i] for i in order]

        new_regions.append(clean_reg)

    return new_regions, np.asarray(new_verts, dtype=np.float32)


# ================== 主算法类 ==================
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

        # 从 64 降到 16，显著减少 polygons_ti.from_numpy 的 memcpy 量；
        # 若个别 cell 边数超过 16 会自动扩容，无需担心溢出。
        self.max_verts_per_region = 16
        self.total_vertices = n_points * self.max_verts_per_region
        self.points_ti = ti.ndarray(dtype=ti.f32, shape=(n_points, 2))
        self.centroids_ti = ti.ndarray(dtype=ti.f32, shape=(n_points, 2))
        self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(self.total_vertices, 2))
        self.region_offsets_ti = ti.ndarray(dtype=ti.i32, shape=(n_points,))
        self.region_sizes_ti = ti.ndarray(dtype=ti.i32, shape=(n_points,))
        self.sides_ti = ti.ndarray(dtype=ti.i32, shape=(n_points,))
        self.points_ti.from_numpy(self.points.astype(np.float32))
        self._polygons_np = np.zeros((self.total_vertices, 2), dtype=np.float32)
        self._sizes_np = np.zeros(n_points, dtype=np.int32)
        self._offsets_np = np.zeros(n_points, dtype=np.int32)

    def _init_points(self, n_points):
        from scipy.stats import qmc
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        sobol = qmc.Sobol(d=2, scramble=True, seed=self.rng.randint(0, 10000))
        m = int(np.ceil(np.log2(n_points)))
        pts = sobol.random_base2(m).astype(np.float32)[:n_points]
        pts[:, 0] = self.xmin + pts[:, 0] * (self.xmax - self.xmin)
        pts[:, 1] = self.ymin + pts[:, 1] * (self.ymax - self.ymin)
        jitter = (self.rng.rand(n_points, 2) - 0.5) * 0.05 * (self.xmax - self.xmin)
        pts += jitter.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], self.xmin + 1e-6, self.xmax - 1e-6)
        pts[:, 1] = np.clip(pts[:, 1], self.ymin + 1e-6, self.ymax - 1e-6)
        return pts

    def _finite_voronoi(self, vor):
        radius = np.ptp(vor.points, axis=0).max() * 2.0
        center = vor.points.mean(axis=0)
        return _finite_voronoi_vectorized(vor, radius, center)

    def _prepare_polygons(self, regions, vertices):
        # 原地填充，去掉每个 region 的 vstack / norm 开销；只在容量不足时扩容一次。
        n = len(regions)
        sizes = np.zeros(n, dtype=np.int32)
        offsets = np.zeros(n, dtype=np.int32)
        flat = self._polygons_np
        pos = 0
        for i, r in enumerate(regions):
            nv = len(r)
            if nv < 3:
                continue
            poly = vertices[r]
            need_close = 1 if (poly[0, 0] != poly[-1, 0] or poly[0, 1] != poly[-1, 1]) else 0
            nv_closed = nv + need_close

            if pos + nv_closed > len(flat):
                new_total = max(len(flat) * 2, pos + nv_closed)
                new_flat = np.zeros((new_total, 2), dtype=np.float32)
                new_flat[:pos] = flat[:pos]
                self._polygons_np = new_flat
                self.total_vertices = new_total
                self.polygons_ti = ti.ndarray(dtype=ti.f32, shape=(new_total, 2))
                flat = new_flat

            offsets[i] = pos
            sizes[i] = nv_closed
            flat[pos:pos + nv] = poly
            if need_close:
                flat[pos + nv] = poly[0]
            pos += nv_closed

        flat[pos:] = 0.0
        self._sizes_np[:n] = sizes
        self._sizes_np[n:] = 0
        self._offsets_np[:n] = offsets
        self._offsets_np[n:] = 0
        return sizes, offsets

    def _compute_sides_numpy(self):
        # 完全向量化，替代 compute_sides taichi kernel（省去一次 kernel launch + to_numpy）。
        n = self.n_points
        if n == 0:
            self._last_sides = np.zeros(0, dtype=np.int32)
            return
        sizes = self._sizes_np[:n]
        offsets = self._offsets_np[:n]
        used_total = int((offsets + sizes).max())
        if used_total == 0:
            self._last_sides = np.zeros(n, dtype=np.int32)
            return

        poly = self._polygons_np
        x = poly[:used_total, 0]
        y = poly[:used_total, 1]
        oob = (x < self.xmin - 1e-9) | (x > self.xmax + 1e-9) | \
              (y < self.ymin - 1e-9) | (y > self.ymax + 1e-9)

        cs = np.zeros(used_total + 1, dtype=np.int64)
        cs[1:] = np.cumsum(oob)

        valid = sizes > 0
        starts = offsets[valid]
        ends = offsets[valid] + sizes[valid]
        has_out = (cs[ends] - cs[starts]) > 0

        sides = np.zeros(n, dtype=np.int32)
        sides[valid] = (sizes[valid] - 1) + has_out.astype(np.int32)
        self._last_sides = sides

    def step(self, return_data=True):
        self.step_count += 1
        need_recompute = (self._cached_regions is None)
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

        sizes, offsets = self._prepare_polygons(regions, vertices)

        self.polygons_ti.from_numpy(self._polygons_np)
        self.region_sizes_ti.from_numpy(self._sizes_np)
        self.region_offsets_ti.from_numpy(self._offsets_np)

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
            float(self.xmin), float(self.xmax), float(self.ymin), float(self.ymax),
            self.n_points
        )
        self.points = self.points_ti.to_numpy()

        self._compute_sides_numpy()
        self._last_regions, self._last_vertices = regions, vertices
        if return_data:
            return regions, vertices
        return None

    def multi_step(self, n):
        if n <= 0:
            return
        regions = vertices = None
        for _ in range(n):
            self.step_count += 1
            vor = Voronoi(self.points)
            regions, vertices = self._finite_voronoi(vor)

            sizes, offsets = self._prepare_polygons(regions, vertices)
            self.polygons_ti.from_numpy(self._polygons_np)
            self.region_sizes_ti.from_numpy(self._sizes_np)
            self.region_offsets_ti.from_numpy(self._offsets_np)

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
                float(self.xmin), float(self.xmax), float(self.ymin), float(self.ymax),
                self.n_points
            )
            self.points = self.points_ti.to_numpy()

        # 中间步骤不维护缓存（本来也不会命中），只在最后更新一次，避免无谓的 copy。
        self._cached_regions = regions
        self._cached_vertices = vertices
        self._prev_points = self.points.copy()
        self._compute_sides_numpy()
        self._last_regions, self._last_vertices = regions, vertices

    def get_stats(self):
        sides = self._last_sides
        if sides is None:
            return 0, 0, 0.0, 0.0
        valid = sides[sides >= 3]
        if valid.size == 0:
            return 0, 0, 0.0, 0.0
        total = int(valid.size)
        hex_count = int(np.count_nonzero(valid == 6))
        return hex_count, total, hex_count / total, float(valid.mean())


# ================== GUI ==================
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
        self.setWindowTitle("Voronoi + Lloyd 松弛 (优化版)")
        self.setGeometry(100, 100, 840, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)

        panel_layout.addWidget(self._create_control_group())
        panel_layout.addWidget(self._create_performance_group())
        panel_layout.addWidget(self._create_viz_group())
        panel_layout.addWidget(self._create_parameter_group())
        panel_layout.addWidget(self._create_monte_group())
        panel_layout.addWidget(self._create_stats_group())
        panel_layout.addStretch()

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
        self.monte_samples_spin.setRange(1, 10000)
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
        self.voronoi_widget.set_show_options(
            self.show_pts_check.isChecked(),
            self.show_vor_check.isChecked()
        )

        sides = self.sim._last_sides
        if sides is not None:
            hex_cnt, total, hex_frac, avg = self.sim.get_stats()
            self.status_label1.setText(f"迭代: {self.sim.step_count}, 点数: {self.sim.n_points}")
            self.status_label2.setText(f"六边形: {hex_frac:.1%}, 平均边: {avg:.2f}")

        self.voronoi_widget.set_data(
            self.sim.points,
            self.sim._polygons_np,
            self.sim._offsets_np,
            self.sim._sizes_np,
            sides if sides is not None else np.zeros(0, dtype=np.int32)
        )

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
            steps = self.steps_spin.value()
            if steps == 1:
                self.sim.step()
            else:
                self.sim.multi_step(steps)
            self._refresh_display()

    def reset_simulation(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("开始")
        self.init_simulation()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    setup_gl_format()
    app = QApplication(sys.argv)
    window = VoronoiSimulation()
    window.show()
    sys.exit(app.exec())