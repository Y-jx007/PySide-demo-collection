from custom_import import *
from PySide6.QtGui import QImage

# ==================== 地形与侵蚀 ====================
def _blur3(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


CRACK_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _shift4(mask):
    """四邻域计数（布尔场的扩散统计）。"""
    m = mask.astype(np.float32)
    out = np.zeros_like(m)
    out[1:, :] += m[:-1, :]
    out[:-1, :] += m[1:, :]
    out[:, 1:] += m[:, :-1]
    out[:, :-1] += m[:, 1:]
    return out


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box_blur(a, r):
    """可分离滑动平均（半径 r 像素）。"""
    if r < 1:
        return a
    k = 2 * r + 1
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    cs = np.cumsum(p, axis=1, dtype=np.float32)
    cs = np.concatenate([np.zeros((cs.shape[0], 1), np.float32), cs], axis=1)
    a = (cs[:, k:] - cs[:, :-k]) / k
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    cs = np.cumsum(p, axis=0, dtype=np.float32)
    cs = np.concatenate([np.zeros((1, cs.shape[1]), np.float32), cs], axis=0)
    return (cs[k:, :] - cs[:-k, :]) / k


def value_noise(n, rng, octaves=5, base=3):
    """多倍频平滑噪声：白噪声反复模糊，没有网格折痕。"""
    total = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    for i in range(octaves):
        f = base * (2 ** i)
        if f > n:
            break
        r = max(1, int(round(n / f / 3.0)))
        g = rng.random((n, n)).astype(np.float32)
        for _ in range(3):
            g = _box_blur(g, r)
        total += amp * g
        amp *= 0.42
    total -= total.min()
    total /= max(1e-6, total.max())
    return total


class ErosionSim:
    """雨滴侵蚀 + 干燥龟裂。

    侵蚀：每个雨滴沿坡向下流，按"携沙容量"侵蚀或沉积，容量与坡度、速度、水量成正比。
    龟裂：泥板收缩时沿 Voronoi 板块边界开裂，随干燥程度逐渐张开，只出现在平缓处。
    """

    MAX_DROPS = 6000

    def __init__(self, size=320, seed=1):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.h = self._make_terrain()
        self.h0 = self.h.copy()
        self._update_relief()
        self.moisture = np.full((size, size), 0.85, dtype=np.float32)   # 逐格含水量
        self.raining = False
        self.drying = False
        self.rain_rate = 320

        # 侵蚀参数
        self.inertia = 0.35
        self.erode = 0.12
        self.deposit = 0.20
        self.capacity = 4.0
        self.min_slope = 0.008
        self.evap = 0.02
        self.gravity = 4.0
        self.max_life = 34
        # 龟裂动力学参数
        self.nucleate_rate = 0.004
        self.grow_ratio = 0.8
        self.stress_gain = 1.6
        self.strength_base = 0.22
        self.relief_radius = 12
        self.relief_gain = 3.0

        # 雨滴状态
        self.px = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.py = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.dx = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.dy = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.spd = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.wat = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.sed = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self.life = np.zeros(self.MAX_DROPS, dtype=np.int32)
        self.count = 0
        self.total_drops = 0

        # 表层物质：细颗粒占比（黏土/粉砂），由雨水搬运决定
        self.fines = np.full((size, size), 0.78, dtype=np.float32)
        self.fines += 0.16 * (value_noise(size, self.rng, octaves=3, base=4) - 0.5)
        np.clip(self.fines, 0.15, 1.0, out=self.fines)
        # 干缩拉应力、裂缝张开度
        self.stress = np.zeros((size, size), dtype=np.float32)
        self.crack = np.zeros((size, size), dtype=np.float32)
        # 雨滴携带的细颗粒比例
        self.sed_fine = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self._build_texture()

    # ---------- 地形 ----------
    def _make_terrain(self):
        """平滑的多倍频山脉：低频做主体，ridged 项做山脊。"""
        n = self.size
        rng = self.rng
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

        base = value_noise(n, rng, octaves=4, base=3)
        h = 0.85 * base

        dome = np.exp(-(((xx - n * 0.5) ** 2 + (yy - n * 0.45) ** 2) / (2 * (n * 0.32) ** 2)))
        h = h * 0.55 + 0.95 * dome
        h += 0.25 * value_noise(n, rng, octaves=3, base=2)

        h -= h.min()
        h /= max(1e-6, h.max())
        for _ in range(3):
            h = _blur3(h)
        return (h * 1.6).astype(np.float32)

    def reset(self):
        self.h = self._make_terrain()
        self.h0 = self.h.copy()
        self._update_relief()
        self.moisture[:] = 0.85
        self.fines[:] = 0.78
        self.fines += 0.16 * (value_noise(self.size, self.rng, octaves=3, base=4) - 0.5)
        np.clip(self.fines, 0.15, 1.0, out=self.fines)
        self.stress[:] = 0.0
        self.crack[:] = 0.0
        self.count = 0
        self.total_drops = 0
        self._build_texture()

    def _update_relief(self):
        h = self.h
        lo, hi = float(h.min()), float(h.max())
        self._relief = ((h - lo) / max(1e-6, hi - lo)).astype(np.float32)

    @property
    def dryness(self):
        return float(1.0 - self.moisture.mean())

    # ---------- 龟裂 ----------
    def _build_texture(self):
        """板块几何（Voronoi 边界距离）+ 泥沙颗粒 + 起裂缺陷。

        裂缝的**位置**来自板块边界：泥在收缩时向各自的收缩中心坍缩，边界处受拉，
        这正是干缩龟裂成多边形的物理来源。开裂的**强弱**则由应力场决定。
        """
        n = self.size
        rng = self.rng
        cell = max(12.0, n / 9.0)
        gr = int(np.ceil(n / cell)) + 1
        gsz = gr + 2
        jitter = 0.85
        site_x = ((np.arange(gsz)[None, :] - 0.5) + (rng.random((gsz, gsz)) - 0.5) * jitter) * cell
        site_y = ((np.arange(gsz)[:, None] - 0.5) + (rng.random((gsz, gsz)) - 0.5) * jitter) * cell

        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
        ci = np.clip((xs / cell).astype(np.int32), 0, gr - 1) + 1
        cj = np.clip((ys / cell).astype(np.int32), 0, gr - 1) + 1

        d1 = np.full((n, n), 1e9, np.float32)
        d2 = np.full((n, n), 1e9, np.float32)
        x1 = np.zeros((n, n), np.float32); y1 = np.zeros((n, n), np.float32)
        x2 = np.zeros((n, n), np.float32); y2 = np.zeros((n, n), np.float32)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                j = cj + dj
                i = ci + di
                sx = site_x[j, i]
                sy = site_y[j, i]
                d = np.hypot(xs - sx, ys - sy)
                closer = d < d1
                d2 = np.where(closer, d1, np.minimum(d2, d))
                x2 = np.where(closer, x1, x2); y2 = np.where(closer, y1, y2)
                d1 = np.where(closer, d, d1)
                x1 = np.where(closer, sx, x1); y1 = np.where(closer, sy, y1)

        ab = np.maximum(np.hypot(x2 - x1, y2 - y1), 1e-3)
        self.crack_edge = ((d2 * d2 - d1 * d1) / (2.0 * ab)).astype(np.float32)

        # 泥沙颗粒
        grain = value_noise(n, rng, octaves=2, base=n // 8)
        self.grain = (grain - 0.5).astype(np.float32)
        # 起裂缺陷：真实泥面总有划痕、石子、干缩点，那里强度更低
        flaws = value_noise(n, rng, octaves=3, base=6)
        self.flaws = (flaws ** 2).astype(np.float32)

    # ---------- 雨滴 ----------
    def add_rain(self, count, cx=None, cy=None, radius=0):
        if count <= 0:
            return
        free = self.MAX_DROPS - self.count
        count = int(min(count, free))
        if count <= 0:
            return
        i0 = self.count
        i1 = i0 + count
        n = self.size - 2
        if cx is None:
            self.px[i0:i1] = self.rng.random(count) * n + 1.0
            self.py[i0:i1] = self.rng.random(count) * n + 1.0
        else:
            ang = self.rng.random(count) * (2 * np.pi)
            rad = np.sqrt(self.rng.random(count)) * max(1.0, radius)
            self.px[i0:i1] = np.clip(cx + np.cos(ang) * rad, 1, n)
            self.py[i0:i1] = np.clip(cy + np.sin(ang) * rad, 1, n)
        self.dx[i0:i1] = 0.0
        self.dy[i0:i1] = 0.0
        self.spd[i0:i1] = 1.0
        self.wat[i0:i1] = 1.0
        self.sed[i0:i1] = 0.0
        self.sed_fine[i0:i1] = 0.5
        self.life[i0:i1] = self.max_life
        self.count = i1
        self.total_drops += count

    def _sample(self, x, y):
        n = self.size
        xi = np.clip(x.astype(np.int32), 0, n - 2)
        yi = np.clip(y.astype(np.int32), 0, n - 2)
        u = np.clip(x - xi, 0.0, 1.0)
        v = np.clip(y - yi, 0.0, 1.0)
        h = self.h
        h00 = h[yi, xi]
        h10 = h[yi, xi + 1]
        h01 = h[yi + 1, xi]
        h11 = h[yi + 1, xi + 1]
        height = h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v + h11 * u * v
        gx = (h00 - h10) * (1 - v) + (h01 - h11) * v
        gy = (h00 - h01) * (1 - u) + (h10 - h11) * u
        return height, gx, gy, yi, xi

    def step_drops(self):
        if self.count == 0:
            return
        n = self.count
        x = self.px[:n]
        y = self.py[:n]
        h_old, gx, gy, yi, xi = self._sample(x, y)

        # 方向：惯性 + 下坡
        inv = 1.0 - self.inertia
        self.dx[:n] = self.dx[:n] * self.inertia - gx * inv
        self.dy[:n] = self.dy[:n] * self.inertia - gy * inv
        norm = np.hypot(self.dx[:n], self.dy[:n])
        norm[norm < 1e-6] = 1.0
        self.dx[:n] /= norm
        self.dy[:n] /= norm

        nx = x + self.dx[:n]
        ny = y + self.dy[:n]
        h_new, _, _, _, _ = self._sample(nx, ny)

        dh = h_new - h_old
        cap = np.maximum(-dh, self.min_slope) * self.spd[:n] * self.capacity * self.wat[:n]
        dep = np.maximum(self.sed[:n] - cap, 0.0) * self.deposit
        ero = np.minimum((cap - self.sed[:n]) * self.erode, np.maximum(-dh, 0.0))

        # 细颗粒优先被冲走：侵蚀带走的是当地的细泥，沉积下来的也按携带比例归还
        f_loc = self.fines[yi, xi]
        fine_ero = ero * (0.30 + 1.4 * f_loc)          # 细泥更容易被冲走
        fine_dep = dep * self.sed_fine[:n] * 0.8       # 沉积物按携带比例落回

        delta = dep - ero
        np.add.at(self.h, (yi, xi), delta)
        np.add.at(self.fines, (yi, xi), fine_dep - fine_ero)
        np.clip(self.fines, 0.42, 1.0, out=self.fines)   # 冲沟里仍是泥质，能干裂

        carried = self.sed[:n] * self.sed_fine[:n]
        self.sed[:n] = np.clip(self.sed[:n] + ero - dep, 0.0, 4.0)
        carried = np.clip(carried + fine_ero - fine_dep, 0.0, 4.0)
        self.sed_fine[:n] = np.where(self.sed[:n] > 1e-4, carried / np.maximum(self.sed[:n], 1e-4), 0.5)

        self.spd[:n] = np.sqrt(np.maximum(0.0, self.spd[:n] ** 2 + dh * self.gravity))
        self.wat[:n] = self.wat[:n] * (1.0 - self.evap)
        self.px[:n] = nx
        self.py[:n] = ny
        self.life[:n] -= 1

        # 记录湿润痕迹
        self.moisture[yi, xi] += 0.02
        np.clip(self.moisture, 0.0, 1.0, out=self.moisture)

        dead = (self.life[:n] <= 0) | (self.wat[:n] < 0.02) | (nx < 1) | (ny < 1) \
            | (nx > self.size - 2) | (ny > self.size - 2)
        if dead.any():
            keep = ~dead
            m = int(keep.sum())
            for arr in (self.px, self.py, self.dx, self.dy, self.spd, self.wat,
                        self.sed, self.sed_fine, self.life):
                arr[:m] = arr[:n][keep]
            self.count = m

    # ---------- 每帧 ----------
    def step(self):
        loss = np.float32(0.0)
        if self.raining:
            self.add_rain(self.rain_rate)
            # 下雨时地面趋近饱和，雨水在低洼处积得更多
            self.moisture += 0.006 * (1.0 - self.moisture)
        elif self.drying:
            # 逐格蒸发：高处、坡地干得快，低洼处保留水分
            rate = 0.0035 * (0.65 + 0.75 * np.clip(self._relief, 0.0, 1.0))
            loss = rate
            self.moisture -= rate
        else:
            self.moisture -= 0.0004
        np.clip(self.moisture, 0.0, 1.0, out=self.moisture)
        self.step_drops()
        np.clip(self.h, 0.0, 3.0, out=self.h)
        self._update_cracks(loss)

    # ---------- 龟裂（几何来自板块边界，开合由应力决定） ----------
    def _update_cracks(self, loss):
        """准静态干缩应力：干得越透、细泥越多，拉应力越大。

        裂缝的几何来自板块边界（`crack_edge`——泥向各自收缩中心坍缩，边界处受拉，
        这正是干缩龟裂成多边形的物理来源）；但**开不开裂、开多宽**由这里的应力决定。
        所以被雨水冲刷过的沟床只要还是泥质、只要干了，一样会裂。
        """
        dry = np.clip(1.0 - self.moisture, 0.0, 1.0)
        self.stress = self.stress_gain * self.fines * dry

        # 几何：板块边界处受拉；宽度随干燥程度增加
        mean_dry = float(dry.mean())
        w = 0.55 + 1.15 * mean_dry
        edge_mask = 1.0 - smoothstep(w * 0.5, w * 1.9, self.crack_edge)
        # 只有应力足够大（干透、细泥多）的地方才真正裂开
        opening = np.clip((self.stress - 0.20) * 2.2, 0.0, 1.0)
        self.crack = np.clip(edge_mask * opening, 0.0, 1.0)

        # 裂缝里的水分蒸发更快
        wet_crack = self.crack > 0.35
        self.moisture[wet_crack] -= 0.010
        np.clip(self.moisture, 0.0, 1.0, out=self.moisture)


# ==================== 画布 ====================
class TerrainCanvas(QWidget):
    def __init__(self, sim, app=None):
        super().__init__()
        self.sim = sim
        self.app = app
        self.show_water = True
        self.show_cracks = True
        self.show_shade = True
        self._soil_lut = None
        self._image = None
        self._dirty = True
        self.setMinimumSize(420, 420)
        self._drawing = False

    def set_sim(self, sim):
        self.sim = sim
        self._dirty = True

    def invalidate(self):
        self._dirty = True
        self.update()

    # ---------- 着色 ----------
    def _build_soil_lut(self):
        """256×256 查表：高度 × 湿度 → 土壤颜色（含高处露岩）。"""
        hi = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        wi = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
        dry = np.array([188, 162, 118], np.float32)
        wetc = np.array([118, 92, 66], np.float32)
        lut = dry[None, None, :] * (1.0 - wi[..., None]) + wetc[None, None, :] * wi[..., None]
        lut = np.broadcast_to(lut, (256, 256, 3)).copy()
        rock = np.clip((hi - 0.66) / 0.34, 0.0, 1.0)[..., None] * 0.55
        lut = lut * (1.0 - rock) + np.array([146, 142, 138], np.float32) * rock
        self._soil_lut = np.clip(lut, 0, 255).astype(np.uint8).reshape(-1, 3)

    def _render_image(self):
        sim = self.sim
        n = sim.size
        h = _blur3(_blur3(sim.h))     # 显示用平滑：抹掉单格开凿痕迹

        # 裂缝两侧的泥板会翘起：用裂缝场的晕边做抬升（只影响显示）
        rim = np.clip(_blur3(sim.crack) - sim.crack, 0.0, 1.0)
        h = h + rim * 0.02

        hmin, hmax = float(h.min()), float(h.max())
        hn = (h - hmin) / max(1e-6, hmax - hmin)

        # 光影
        gy, gx = np.gradient(h)
        exag = n * 0.30
        nx, ny, nz = -gx * exag, -gy * exag, np.ones_like(hn)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        light = (nx * -0.48 + ny * -0.48 + nz * 0.73) * inv
        shade = np.clip(0.74 + 0.52 * light, 0.40, 1.3)

        wetfield = _blur3(sim.moisture)

        if self._soil_lut is None:
            self._build_soil_lut()
        idx = (hn * 255.0).astype(np.int32) * 256 + (np.clip(wetfield * 1.15, 0, 1) * 255.0).astype(np.int32)
        rgb = self._soil_lut[idx].astype(np.float32)

        if self.show_shade:
            rgb *= shade[..., None]

        # 粗砂（细颗粒被冲走的地方）偏灰、颗粒感强；细泥偏暖
        coarse = np.clip((0.62 - sim.fines) * 1.8, 0.0, 1.0)[..., None]
        rgb = rgb * (1 - coarse * 0.45) + np.array([172, 168, 160], np.float32) * (coarse * 0.45)
        rgb += sim.grain[..., None] * (7.0 + 10.0 * coarse)

        # 积水 / 湿润
        if self.show_water:
            pond = np.clip((0.16 - hn) / 0.16, 0.0, 1.0) * np.clip((wetfield - 0.90) * 8.0, 0, 1)
            water = np.clip(pond, 0.0, 1.0)[..., None]
            rgb = rgb * (1 - water * 0.75) + np.array([70, 118, 165], np.float32) * (water * 0.75)

        # 龟裂：直接读应力模型产出的裂缝场
        if self.show_cracks:
            crack = np.clip(_blur3(sim.crack) * 1.2, 0.0, 1.0)[..., None]
            rgb = rgb * (1 - crack * 0.85) + np.array([40, 32, 25], np.float32) * (crack * 0.85)

        out = np.clip(rgb, 0, 255).astype(np.uint8)
        self._image = QImage(out.data, n, n, n * 3, QImage.Format_RGB888).copy()
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._image is None:
            self._render_image()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        side = min(self.width(), self.height())
        x0 = (self.width() - side) // 2
        y0 = (self.height() - side) // 2
        painter.drawImage(QRect(x0, y0, side, side), self._image)
        painter.end()

    # ---------- 鼠标：局部降雨 ----------
    def _to_sim(self, pos):
        side = min(self.width(), self.height())
        x0 = (self.width() - side) // 2
        y0 = (self.height() - side) // 2
        if side <= 0:
            return None
        sx = (pos.x() - x0) / side * self.sim.size
        sy = (pos.y() - y0) / side * self.sim.size
        if 0 <= sx < self.sim.size and 0 <= sy < self.sim.size:
            return sx, sy
        return None

    def mousePressEvent(self, event):
        p = self._to_sim(event.position().toPoint())
        if p is None:
            return
        self._drawing = True
        self.sim.add_rain(160, p[0], p[1], radius=self.sim.size * 0.05)

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        p = self._to_sim(event.position().toPoint())
        if p is not None:
            self.sim.add_rain(60, p[0], p[1], radius=self.sim.size * 0.04)

    def mouseReleaseEvent(self, event):
        self._drawing = False


# ==================== 主窗口 ====================
class ErosionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水力侵蚀 · 干燥龟裂")
        self.setGeometry(120, 70, 1080, 800)

        self.size = 320
        self.sim = ErosionSim(self.size, seed=1)
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(45)
        self.update_stats()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(272)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = TerrainCanvas(self.sim, self)

        self._build_weather_group(panel_layout)
        self._build_erosion_group(panel_layout)
        self._build_display_group(panel_layout)
        panel_layout.addStretch()

        root.addWidget(panel)
        root.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("点「降雨」开始冲刷，点「干燥」看泥面开裂")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_dry = QLabel("干燥 0%")
        self.lbl_drops = QLabel("雨滴 0")
        self.lbl_fps = QLabel("帧率 –")
        for w in (self.lbl_dry, self.lbl_drops, self.lbl_fps):
            self.statusBar().addPermanentWidget(w)
        self._frames = 0
        self._fps_t = time.perf_counter()

    def _build_weather_group(self, layout):
        group = QGroupBox("天气")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.rain_btn = QPushButton("降雨")
        self.rain_btn.clicked.connect(self.toggle_rain)
        row.addWidget(self.rain_btn, 1)
        self.dry_btn = QPushButton("干燥")
        self.dry_btn.clicked.connect(self.toggle_dry)
        row.addWidget(self.dry_btn, 1)
        vbox.addLayout(row)

        form = QFormLayout()
        form.setSpacing(4)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.rain_slider = QSlider(Qt.Horizontal)
        self.rain_slider.setRange(40, 1200)
        self.rain_slider.setValue(self.sim.rain_rate)
        self.rain_slider.valueChanged.connect(self.on_rain_rate)
        self.rain_label = QLabel(str(self.sim.rain_rate))
        self.rain_label.setFixedWidth(44)
        self.rain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        r = QHBoxLayout()
        r.setSpacing(4)
        r.addWidget(self.rain_slider, 1)
        r.addWidget(self.rain_label)
        form.addRow("雨量", r)

        reset_btn = QPushButton("重置地形")
        reset_btn.clicked.connect(self.reset)
        form.addRow("", reset_btn)
        vbox.addLayout(form)
        layout.addWidget(group)

    def _build_erosion_group(self, layout):
        group = QGroupBox("侵蚀参数")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.sl_erode, _ = self._slider(form, "侵蚀", 5, 100, int(self.sim.erode * 100),
                                        lambda v: setattr(self.sim, "erode", v / 100.0))
        self.sl_deposit, _ = self._slider(form, "沉积", 5, 100, int(self.sim.deposit * 100),
                                          lambda v: setattr(self.sim, "deposit", v / 100.0))
        self.sl_inertia, _ = self._slider(form, "惯性", 0, 95, int(self.sim.inertia * 100),
                                          lambda v: setattr(self.sim, "inertia", v / 100.0))
        self.sl_capacity, _ = self._slider(form, "携沙", 10, 100, int(self.sim.capacity * 10),
                                           lambda v: setattr(self.sim, "capacity", v / 10.0))
        layout.addWidget(group)

    def _slider(self, form, text, lo, hi, value, setter):
        row = QHBoxLayout()
        row.setSpacing(4)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(str(value))
        label.setFixedWidth(38)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(v):
            label.setText(str(v))
            setter(v)

        slider.valueChanged.connect(on_change)
        row.addWidget(slider, 1)
        row.addWidget(label)
        form.addRow(text, row)
        return slider, label

    def _build_display_group(self, layout):
        group = QGroupBox("显示")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(2)
        for text, attr, default in (("光影", "show_shade", True),
                                    ("积水 / 湿润", "show_water", True),
                                    ("干燥龟裂", "show_cracks", True)):
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.toggled.connect(lambda checked, a=attr: self._toggle(a, checked))
            vbox.addWidget(cb)
        layout.addWidget(group)

    def _toggle(self, attr, checked):
        setattr(self.canvas, attr, bool(checked))
        self.canvas.invalidate()

    # ---------- 交互 ----------
    def toggle_rain(self):
        self.sim.raining = not self.sim.raining
        self.rain_btn.setText("停雨" if self.sim.raining else "降雨")
        if self.sim.raining:
            self.sim.drying = False
            self.dry_btn.setText("干燥")
            self.lbl_status.setText("正在降雨冲刷…")

    def toggle_dry(self):
        self.sim.drying = not self.sim.drying
        self.dry_btn.setText("停止干燥" if self.sim.drying else "干燥")
        if self.sim.drying:
            self.sim.raining = False
            self.rain_btn.setText("降雨")
            self.lbl_status.setText("泥土干燥中，裂纹逐渐张开…")

    def reset(self):
        self.sim.reset()
        self.canvas.invalidate()
        self.lbl_status.setText("地形已重置")

    def on_rain_rate(self, v):
        self.sim.rain_rate = v
        self.rain_label.setText(str(v))

    def tick(self):
        sim = self.sim
        busy = sim.raining or sim.drying or sim.count > 0
        if not busy:
            return
        sim.step()
        if sim.raining:
            sim.step()          # 下雨时每帧推进两步，视觉速度不变但重绘次数少一半
        self.canvas.invalidate()
        self.update_stats()
        self._frames += 1

    def update_stats(self):
        self.lbl_dry.setText(f"干燥 {self.sim.dryness * 100:.0f}%")
        self.lbl_drops.setText(f"雨滴 {self.sim.count:,}")
        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            self.lbl_fps.setText(f"帧率 {self._frames / (now - self._fps_t):.0f}")
            self._frames = 0
            self._fps_t = now

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ErosionApp()
    window.show()
    sys.exit(app.exec())
