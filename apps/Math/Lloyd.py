from common.qt_base import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QOpenGLWidget, QPushButton, QSpinBox, QSurfaceFormat, QTimer, QVBoxLayout, QWidget, Qt, math, sys, time)
from common.numpy_base import (np)
from common.numba_base import (njit)
from common.opengl_base import (GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST, GL_DYNAMIC_DRAW, GL_ELEMENT_ARRAY_BUFFER, GL_FLOAT, GL_LINES, GL_LINE_SMOOTH, GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_POINTS, GL_POINT_SMOOTH, GL_PROJECTION, GL_SRC_ALPHA, GL_TRIANGLES, GL_UNSIGNED_INT, GL_VERTEX_ARRAY, glBindBuffer, glBlendFunc, glBufferData, glBufferSubData, glClear, glClearColor, glColor4f, glDisable, glDrawArrays, glDrawElements, glEnable, glEnableClientState, glGenBuffers, glLineWidth, glLoadIdentity, glMatrixMode, glOrtho, glPointSize, glVertexPointer, glViewport)
from itertools import chain
from scipy.spatial import Voronoi

# Lloyd 松弛的热点不在 Taichi，而在每个 step 的 Python 后处理：
#   scipy Voronoi -> 无限脊扩展 -> 逐 cell 建多边形 -> 质心 -> 移动。
# 旧实现把 polygons 拷进 Taichi ndarray 再拷回来，并在 Python 里逐 cell
# 做 numpy fancy-index；这里改为：
#   * 一次向量化构建扁平多边形缓冲（只对少量边界 cell 走 Python 循环）；
#   * 质心 / 移动 / 蒙特卡洛直接用 Numba 操作 NumPy 数组，零拷贝。
def setup_gl_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)


# ================== Numba 核函数 ==================
@njit(nogil=True, fastmath=True, cache=True)
def compute_cell_centroids(polygons, offsets, sizes, centroids):
    """鞋带公式求多边形质心（多边形首尾已闭合）。"""
    n = sizes.shape[0]
    for i in range(n):
        nv = sizes[i]
        if nv < 3:
            centroids[i, 0] = 0.0
            centroids[i, 1] = 0.0
            continue
        start = offsets[i]
        area = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(nv - 1):
            x1 = polygons[start + j, 0]
            y1 = polygons[start + j, 1]
            x2 = polygons[start + j + 1, 0]
            y2 = polygons[start + j + 1, 1]
            cross = x1 * y2 - x2 * y1
            area += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if abs(area) < 1e-12:
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


@njit(nogil=True, fastmath=True, cache=True)
def move_points(points, centroids, mobility,
                xmin, xmax, ymin, ymax):
    n = points.shape[0]
    for i in range(n):
        points[i, 0] += (centroids[i, 0] - points[i, 0]) * mobility
        points[i, 1] += (centroids[i, 1] - points[i, 1]) * mobility
        points[i, 0] = max(xmin + 1e-6, min(xmax - 1e-6, points[i, 0]))
        points[i, 1] = max(ymin + 1e-6, min(ymax - 1e-6, points[i, 1]))


@njit(nogil=True, fastmath=True, cache=True)
def compute_monte_carlo_centroids(polygons, offsets, sizes, centroids,
                                  samples, probability):
    """按概率用蒙特卡洛估计部分 cell 的质心（可选功能）。"""
    n = sizes.shape[0]
    for i in range(n):
        nv = sizes[i]
        if nv < 4:
            continue
        if probability < 1.0 and np.random.random() > probability:
            continue
        start = offsets[i]
        min_x = polygons[start, 0]
        max_x = min_x
        min_y = polygons[start, 1]
        max_y = min_y
        for j in range(start, start + nv):
            x = polygons[j, 0]
            y = polygons[j, 1]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        cx = 0.0
        cy = 0.0
        cnt = 0
        dx = max_x - min_x
        dy = max_y - min_y
        nv_real = nv - 1
        for _ in range(samples):
            px = min_x + np.random.random() * dx
            py = min_y + np.random.random() * dy
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
        self.vbo_hex_idx = None
        self.vbo_other_idx = None
        self.vbo_line_idx = None
        self.num_points = 0
        self.poly_capacity = 0
        self.point_capacity = 0
        self.idx_capacity = {}
        self.hex_tri_count = 0
        self.other_tri_count = 0
        self.line_count = 0

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

    @staticmethod
    def _fan_indices(firsts, counts):
        """把每个闭合多边形展开成 TRIANGLES 的索引（0,i,i+1）。"""
        if len(firsts) == 0:
            return np.zeros(0, dtype=np.int32)
        triangles = counts - 2
        keep = triangles > 0
        firsts = firsts[keep]
        triangles = triangles[keep]
        if len(triangles) == 0:
            return np.zeros(0, dtype=np.int32)
        total = int(triangles.sum())
        tri_id = np.repeat(np.arange(len(triangles), dtype=np.int64), triangles)
        starts = np.repeat(np.cumsum(triangles) - triangles, triangles)
        local = np.arange(total, dtype=np.int64) - starts
        base = firsts[tri_id].astype(np.int64)
        out = np.empty(total * 3, dtype=np.int32)
        out[0::3] = base
        out[1::3] = base + local + 1
        out[2::3] = base + local + 2
        return out

    @staticmethod
    def _line_indices(firsts, counts):
        """把闭合多边形的边展开成 GL_LINES 索引。"""
        if len(firsts) == 0:
            return np.zeros(0, dtype=np.int32)
        edges = counts - 1
        keep = edges > 0
        firsts = firsts[keep]
        edges = edges[keep]
        if len(edges) == 0:
            return np.zeros(0, dtype=np.int32)
        total = int(edges.sum())
        edge_id = np.repeat(np.arange(len(edges), dtype=np.int64), edges)
        starts = np.repeat(np.cumsum(edges) - edges, edges)
        local = np.arange(total, dtype=np.int64) - starts
        base = firsts[edge_id].astype(np.int64)
        out = np.empty(total * 2, dtype=np.int32)
        out[0::2] = base + local
        out[1::2] = base + local + 1
        return out

    def _upload_indices(self, vbo, data):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo)
        nbytes = int(data.nbytes)
        if self.idx_capacity.get(vbo) != nbytes:
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, nbytes, data, GL_DYNAMIC_DRAW)
            self.idx_capacity[vbo] = nbytes
        else:
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, nbytes, data)

    def set_data(self, points, poly_flat, poly_offsets, poly_counts, poly_sides):
        self.points = points
        self.num_points = len(points)

        # 六边形 / 非六边形分组：只做一次向量化筛选
        hex_mask = (poly_sides == 6)
        other_mask = (poly_sides != 6) & (poly_sides > 0)
        hex_firsts = poly_offsets[hex_mask].astype(np.int32)
        hex_counts = poly_counts[hex_mask].astype(np.int32)
        other_firsts = poly_offsets[other_mask].astype(np.int32)
        other_counts = poly_counts[other_mask].astype(np.int32)

        hex_idx = self._fan_indices(hex_firsts, hex_counts)
        other_idx = self._fan_indices(other_firsts, other_counts)
        line_idx = self._line_indices(
            np.ascontiguousarray(poly_offsets, dtype=np.int32),
            np.ascontiguousarray(poly_counts, dtype=np.int32))
        self.hex_tri_count = len(hex_idx)
        self.other_tri_count = len(other_idx)
        self.line_count = len(line_idx)

        if self.vbo_poly is None:
            self.vbo_poly = glGenBuffers(1)
            self.vbo_point = glGenBuffers(1)
            self.vbo_hex_idx = glGenBuffers(1)
            self.vbo_other_idx = glGenBuffers(1)
            self.vbo_line_idx = glGenBuffers(1)

        poly_bytes = int(poly_flat.nbytes)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_poly)
        if poly_bytes != self.poly_capacity:
            glBufferData(GL_ARRAY_BUFFER, poly_bytes, poly_flat, GL_DYNAMIC_DRAW)
            self.poly_capacity = poly_bytes
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, poly_bytes, poly_flat)

        pts_np = np.asarray(points, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point)
        if pts_np.nbytes != self.point_capacity:
            glBufferData(GL_ARRAY_BUFFER, pts_np.nbytes, pts_np, GL_DYNAMIC_DRAW)
            self.point_capacity = pts_np.nbytes
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, pts_np.nbytes, pts_np)

        self._upload_indices(self.vbo_hex_idx, hex_idx)
        self._upload_indices(self.vbo_other_idx, other_idx)
        self._upload_indices(self.vbo_line_idx, line_idx)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
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

            # 旧实现用 GL_POLYGON + glMultiDrawArrays：几千次独立 draw call，
            # 且 GL_POLYGON 在多数驱动上是软件回退。这里预先展开三角形/线段
            # 索引，填充和描边各只需 1~2 次 draw call。
            if self.hex_tri_count:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_hex_idx)
                glColor4f(1.0, 0.8745, 0.3490, 1.0)
                glDrawElements(GL_TRIANGLES, self.hex_tri_count,
                               GL_UNSIGNED_INT, None)

            if self.other_tri_count:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_other_idx)
                glColor4f(0.5647, 0.9333, 0.5647, 0.7059)
                glDrawElements(GL_TRIANGLES, self.other_tri_count,
                               GL_UNSIGNED_INT, None)

            if self.line_count:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_line_idx)
                glColor4f(0.8039, 0.5216, 0.0, 1.0)
                glLineWidth(1.5)
                glDrawElements(GL_LINES, self.line_count, GL_UNSIGNED_INT, None)

        if self.show_points:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_point)
            glVertexPointer(2, GL_FLOAT, 0, None)
            glColor4f(0.3922, 0.3922, 0.3922, 1.0)
            glPointSize(2.0)
            glDrawArrays(GL_POINTS, 0, self.num_points)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


# ================== 主算法类 ==================
class HoneycombCVT:
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

        # 供 UI 读取的扁平多边形数据
        self._polygons_np = np.zeros((0, 2), dtype=np.float32)
        self._sizes_np = np.zeros(n_points, dtype=np.int32)
        self._offsets_np = np.zeros(n_points, dtype=np.int32)
        self.centroids = np.zeros((n_points, 2), dtype=np.float32)

        # 没有发生足够位移时复用上一帧的几何
        self._cached_flat = None
        self._cached_sizes = None
        self._cached_offsets = None
        self._prev_build_points = None
        self._sides_valid = False
        self._last_sides = None

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

    def _build_cells(self, vor):
        """把 Voronoi 图一次性展开成扁平多边形缓冲。
        内部点完全向量化（gather + scatter）；只有带无限脊的边界点
        （约 O(sqrt(N)) 个）需要扩展远点并按角度排序。
        """
        pts = vor.points
        verts = np.asarray(vor.vertices, dtype=np.float32)
        ridge_verts = vor.ridge_vertices
        ridge_points = np.asarray(vor.ridge_points)
        n_pts = len(pts)

        inf_flags = np.fromiter((-1 in rv for rv in ridge_verts),
                                dtype=bool, count=len(ridge_verts))
        if inf_flags.any():
            inf_rv = np.asarray([rv for rv in ridge_verts
                                 if rv[0] == -1 or rv[1] == -1], dtype=np.int32)
            inf_rp = ridge_points[inf_flags]
            finite_v = np.where(inf_rv[:, 0] == -1, inf_rv[:, 1], inf_rv[:, 0])
            p1 = inf_rp[:, 0]
            p2 = inf_rp[:, 1]
            t_vec = pts[p2] - pts[p1]
            t_vec /= np.linalg.norm(t_vec, axis=1, keepdims=True)
            n_vec = np.column_stack([-t_vec[:, 1], t_vec[:, 0]])
            mid = (pts[p1] + pts[p2]) / 2.0
            center = pts.mean(axis=0)
            n_vec *= np.sign(np.sum((mid - center) * n_vec, axis=1))[:, None]
            radius = np.ptp(pts, axis=0).max() * 2.0
            far_pts = verts[finite_v] + n_vec * radius
            new_verts = np.vstack([verts, far_pts]).astype(np.float32)
            n_orig = len(verts)
            p_to_inf = {}
            for ri in range(len(inf_rp)):
                p_to_inf.setdefault(int(inf_rp[ri, 0]), []).append(ri)
                p_to_inf.setdefault(int(inf_rp[ri, 1]), []).append(ri)
        else:
            new_verts = verts
            n_orig = len(verts)
            p_to_inf = {}

        regions = vor.regions
        point_region = vor.point_region
        boundary = np.fromiter(p_to_inf.keys(), dtype=np.int32, count=len(p_to_inf))
        is_boundary = np.zeros(n_pts, dtype=bool)
        is_boundary[boundary] = True
        interior = np.nonzero(~is_boundary)[0]

        sizes = np.zeros(n_pts, dtype=np.int32)

        # ---- 内部 cell：向量化 gather / scatter ----
        if len(interior):
            pr = point_region[interior]
            regs = [regions[int(j)] for j in pr]
            length = np.fromiter(map(len, regs), dtype=np.int32, count=len(regs))
            src_start = np.empty(len(regs) + 1, dtype=np.int64)
            src_start[0] = 0
            np.cumsum(length, out=src_start[1:])
            flat_ids = np.fromiter(chain.from_iterable(regs), dtype=np.int32,
                                   count=int(src_start[-1]))
            first = flat_ids[src_start[:-1]]
            last = flat_ids[src_start[1:] - 1]
            need_close = ~np.all(new_verts[first] == new_verts[last], axis=1)
            sizes[interior] = length + need_close
        else:
            first = last = np.zeros(0, dtype=np.int32)
            need_close = np.zeros(0, dtype=bool)
            length = np.zeros(0, dtype=np.int32)
            src_start = np.zeros(1, dtype=np.int64)

        # ---- 边界 cell：扩展远点 + 角度排序（数量少） ----
        boundary_cells = []
        for p in boundary:
            reg = regions[int(point_region[p])]
            clean = [v for v in reg if v >= 0]
            clean.extend([n_orig + ri for ri in p_to_inf[int(p)]])
            if len(clean) > 2:
                sx = 0.0
                sy = 0.0
                for v in clean:
                    sx += new_verts[v, 0]
                    sy += new_verts[v, 1]
                cx = sx / len(clean)
                cy = sy / len(clean)
                clean.sort(key=lambda v: math.atan2(
                    new_verts[v, 1] - cy, new_verts[v, 0] - cx))
            close = 1 if (new_verts[clean[0]] != new_verts[clean[-1]]).any() else 0
            sizes[p] = len(clean) + close
            boundary_cells.append((int(p), clean, close))

        offsets = np.empty(n_pts + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        flat = np.empty((int(offsets[-1]), 2), dtype=np.float32)

        if len(interior):
            dst_start = offsets[interior]
            dst_idx = np.repeat(dst_start, length) + (
                np.arange(int(src_start[-1])) - np.repeat(src_start[:-1], length))
            flat[dst_idx] = new_verts[flat_ids]
            close_dst = dst_start + length
            flat[close_dst[need_close]] = new_verts[first[need_close]]

        if boundary_cells:
            dst = []
            src = []
            for p, poly, close in boundary_cells:
                start = int(offsets[p])
                dst.extend(range(start, start + len(poly)))
                src.extend(poly)
                if close:
                    dst.append(start + len(poly))
                    src.append(poly[0])
            flat[np.asarray(dst, dtype=np.int64)] = \
                new_verts[np.asarray(src, dtype=np.int32)]

        return flat, sizes, offsets[:-1].astype(np.int32)

    def _rebuild(self):
        vor = Voronoi(self.points)
        flat, sizes, offsets = self._build_cells(vor)
        self._polygons_np = flat
        self._sizes_np = sizes
        self._offsets_np = offsets
        self._cached_flat = flat
        self._cached_sizes = sizes
        self._cached_offsets = offsets
        self._prev_build_points = self.points.copy()
        self._prev_points = self._prev_build_points
        self._sides_valid = False

    def _use_cached(self):
        self._polygons_np = self._cached_flat
        self._sizes_np = self._cached_sizes
        self._offsets_np = self._cached_offsets

    def _compute_sides_numpy(self):
        n = self.n_points
        if n == 0:
            self._last_sides = np.zeros(0, dtype=np.int32)
            return
        sizes = self._sizes_np
        offsets = self._offsets_np
        used_total = int((offsets + sizes).max()) if n else 0
        if used_total == 0:
            self._last_sides = np.zeros(n, dtype=np.int32)
            return

        poly = self._polygons_np
        x = poly[:used_total, 0]
        y = poly[:used_total, 1]
        oob = ((x < self.xmin - 1e-9) | (x > self.xmax + 1e-9) |
               (y < self.ymin - 1e-9) | (y > self.ymax + 1e-9))
        cs = np.zeros(used_total + 1, dtype=np.int64)
        np.cumsum(oob, out=cs[1:])

        valid = sizes > 0
        starts = offsets[valid]
        ends = offsets[valid] + sizes[valid]
        has_out = (cs[ends] - cs[starts]) > 0
        sides = np.zeros(n, dtype=np.int32)
        sides[valid] = (sizes[valid] - 1) + has_out.astype(np.int32)
        self._last_sides = sides

    def _centroids_and_move(self):
        compute_cell_centroids(self._polygons_np, self._offsets_np,
                               self._sizes_np, self.centroids)
        if self.monte_enabled and self.monte_samples > 0:
            compute_monte_carlo_centroids(
                self._polygons_np, self._offsets_np, self._sizes_np,
                self.centroids, self.monte_samples, float(self.monte_probability))
        move_points(self.points, self.centroids, float(self.mobility),
                    float(self.xmin), float(self.xmax),
                    float(self.ymin), float(self.ymax))

    def step(self, return_data=True):
        self.step_count += 1
        need_recompute = self._cached_flat is None
        if not need_recompute:
            max_disp = np.max(np.abs(self.points - self._prev_points))
            if max_disp > self._recompute_tol:
                need_recompute = True
        if need_recompute:
            self._rebuild()
        else:
            self._use_cached()

        self._centroids_and_move()
        if not self._sides_valid:
            self._compute_sides_numpy()
            self._sides_valid = True

        if return_data:
            return self._polygons_np, self._offsets_np, self._sizes_np
        return None

    def multi_step(self, n):
        if n <= 0:
            return
        for _ in range(n):
            self.step_count += 1
            self._rebuild()
            self._centroids_and_move()
        self._prev_points = self.points.copy()
        self._compute_sides_numpy()
        self._sides_valid = True

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


# 兼容旧的类名（外部若直接 import TaichiHoneycombCVT 仍然可用）
TaichiHoneycombCVT = HoneycombCVT


# ================== GUI ==================
class VoronoiSimulation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
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
        self.freq_spin.setValue(16)     # 约 60 fps；旧默认 1 ms 会让 GUI 一直满载
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
        self.sim = HoneycombCVT(
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
            self.is_running = False
            self.timer.stop()
            self.start_btn.setText("开始")
        else:
            self.is_running = True
            self.start_btn.setText("暂停")
            self._tick()

    def _tick(self):
        """单次定时器：先跑一步，再按剩余预算安排下一步。

        重复 QTimer 在 step 比 interval 慢时会被连续超时事件占满，绘制事件
        被饿死；改成“跑完再排下一次”后，每一步之间都能插入 paint。
        """
        if not self.is_running:
            return
        start = time.perf_counter()
        self.single_step()
        if not self.is_running:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        delay = int(self.freq_spin.value() - elapsed_ms)
        if delay < 1:
            delay = 1
        self.timer.start(delay)

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
