from common.field_canvas import ArrayCanvas
from common.numpy_base import (np)
from common.paths import DATA_DIR
from common.qt_base import (FPSCounter, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                            QPushButton, QSlider, QTimer, QVBoxLayout, QWidget, Qt, LabeledSlider, os, run_app)

from support.physics.soil_water import SoilWater

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
        # 小尺度泥地是平面，不再生成地形/坡度。
        self.h = np.zeros((self.size, self.size), dtype=np.float32)
        self.h0 = self.h.copy()
        self._relief = np.zeros((size, size), dtype=np.float32)
        self.raining = False
        self.drying = False
        self.rain_mm_h = 15.0
        self.dt_min = 20.0
        self.temp_c = 25.0
        self.rh = 55.0
        self.wind = 2.0
        self.soil_clay = 0.35
        self.fracture_kic = 0.030e6      # MPa*m^0.5

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
        self.crack_target = np.zeros((size, size), dtype=np.float32)
        # 裂缝开合不是开关：黏土吸水膨胀/失水收缩需要时间。
        self.crack_tau_open = 720.0       # min
        self.crack_tau_close = 480.0      # min
        # 两层土壤水：0-10 cm 表层 + 10-50 cm 根区
        self.soil = SoilWater(self.fines, self.soil_clay)
        self.moisture = np.zeros((size, size), dtype=np.float32)
        self.moisture[:] = self.soil.wetness
        # 雨滴携带的细颗粒比例
        self.sed_fine = np.zeros(self.MAX_DROPS, dtype=np.float32)
        self._build_texture()
        # 细胞型断裂近似：能量最小化的 Voronoi 板块边界作为裂缝路径，
        # 开合由线弹性断裂力学 K_I/K_IC + COD 控制。
        self.crack[:] = 0.0
        # 离线 CZM 裂缝场：多个独立 Voronoi 图案组成的 bank。
        self.show_czm = False
        self.czm_bank = None
        self.czm_frames = None
        self.czm_levels = None
        self.czm_variant = 0
        try:
            z = np.load(DATA_DIR / "czm_fields.npz")
            self.czm_bank = z["frames"].astype(np.float32) / 255.0
            self.czm_levels = z["levels"].astype(np.float32)
            self.czm_variant = int(self.rng.integers(len(self.czm_bank)))
            self.czm_frames = self.czm_bank[self.czm_variant]
        except Exception:
            try:
                z = np.load(DATA_DIR / "czm_field.npz")
                self.czm_frames = z["frames"].astype(np.float32) / 255.0
                self.czm_levels = z["levels"].astype(np.float32)
            except Exception:
                self.czm_frames = None
        # 裂缝开合不是随土壤水瞬时切换：给 CZM 帧选择加一个吸水/干燥滞后。
        self.czm_shrink_state = None
        self.czm_tau_open = 240.0       # min
        self.czm_tau_close = 900.0      # min


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
        # 小尺度泥地是平面，不再生成地形/坡度。
        self.h = np.zeros((self.size, self.size), dtype=np.float32)
        self.h0 = self.h.copy()
        self._relief = np.zeros((self.size, self.size), dtype=np.float32)
        self.fines[:] = 0.78
        self.fines += 0.16 * (value_noise(self.size, self.rng, octaves=3, base=4) - 0.5)
        np.clip(self.fines, 0.15, 1.0, out=self.fines)
        self.soil.reset(self.fines, self.soil_clay)
        self.moisture[:] = self.soil.wetness
        self.stress[:] = 0.0
        self.crack[:] = 0.0
        self.crack_target[:] = 0.0
        self.czm_shrink_state = None
        if self.czm_bank is not None:
            self.choose_czm_variant()
        self.count = 0
        self.total_drops = 0
        self._build_texture()

    def _update_relief(self):
        h = self.h
        lo, hi = float(h.min()), float(h.max())
        self._relief = ((h - lo) / max(1e-6, hi - lo)).astype(np.float32)

    @property
    def dryness(self):
        return self.soil.dryness

    @property
    def crack_relaxing(self):
        return bool(np.mean(np.abs(self.crack_target - self.crack)) > 0.001)

    @property
    def czm_relaxing(self):
        if not (self.show_czm and self.czm_frames is not None):
            return False
        target = float(np.clip(self.soil.shrinkage.mean(), 0.0, 1.0))
        return self.czm_shrink_state is not None and abs(target - self.czm_shrink_state) > 0.002

    def choose_czm_variant(self):
        """切换到另一组预计算的 Voronoi 裂缝图案。"""
        if self.czm_bank is None:
            return
        self.czm_variant = int(self.rng.integers(len(self.czm_bank)))
        self.czm_frames = self.czm_bank[self.czm_variant]
        self.czm_shrink_state = 0.0
        self.crack[:] = 0.0

    def new_crack_pattern(self):
        """重新生成一组随机板块边界；在“已闭合”状态下开启新一轮干燥时调用。"""
        if self.show_czm and self.czm_bank is not None:
            self.choose_czm_variant()
            return
        self._build_texture()
        self.crack[:] = 0.0
        self.crack_target[:] = 0.0
        self.czm_shrink_state = None

    # ---------- 龟裂 ----------
    def _build_texture(self):
        """板块几何（Voronoi 边界距离）+ 泥沙颗粒 + 起裂缺陷。

        裂缝的**位置**来自板块边界：泥在收缩时向各自的收缩中心坍缩，边界处受拉，
        这正是干缩龟裂成多边形的物理来源。开裂的**强弱**则由应力场决定。
        """
        n = self.size
        rng = self.rng
        # 每次重建裂缝网络都换一组板块尺度/抖动，干裂形态不会每次一样。
        cell = max(10.0, n / (8.0 + 4.0 * float(self.rng.random())))
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
    def _et0(self):
        """FAO-56 Penman-Monteith 潜在蒸散（mm/day，简化净辐射）。"""
        t = float(self.temp_c)
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
        ea = es * float(self.rh) / 100.0
        delta = 4098.0 * es / (t + 237.3) ** 2
        gamma = 0.066
        rn = 12.0                       # MJ/m2/day，固定晴空近似
        u2 = max(0.1, float(self.wind))
        num = 0.408 * delta * rn + gamma * 900.0 / (t + 273.0) * u2 * max(0.0, es - ea)
        den = delta + gamma * (1.0 + 0.34 * u2)
        return float(np.clip(num / max(den, 1e-6), 0.0, 12.0))

    def set_clay(self, value):
        self.soil_clay = float(np.clip(value, 0.05, 0.75))
        self.soil.clay_bias = self.soil_clay
        self.soil.update_texture(self.fines)
        self.moisture[:] = self.soil.wetness

    def step(self):
        if self.raining:
            self.soil.step(self.rain_mm_h, self._et0(), self.dt_min)
        elif self.drying:
            self.soil.step(0.0, self._et0(), self.dt_min)
        self.moisture[:] = self.soil.wetness
        self._update_cracks()

    # ---------- 龟裂（几何来自板块边界，开合由应力决定） ----------
    def _update_cracks(self):
        """细胞型线弹性断裂模型。

        多边形拓扑来自能量最小化的 Voronoi 板块边界；但裂缝是否张开由
        mode-I 应力强度因子控制：

            K_I = sigma * sqrt(pi * a) >= K_IC

        sigma 是受限干缩应力（E * 收缩应变），a 是单元半长；裂缝宽度取
        COD ~ 2*sigma*a/E。再湿润时 K_I 下降，裂缝闭合。
        """
        if self.show_czm and self.czm_frames is not None:
            target = float(np.clip(self.soil.shrinkage.mean(), 0.0, 1.0))
            if self.czm_shrink_state is None:
                self.czm_shrink_state = target
            tau = self.czm_tau_close if target < self.czm_shrink_state else self.czm_tau_open
            alpha = 1.0 - np.exp(-self.dt_min / max(tau, 1.0))
            self.czm_shrink_state += alpha * (target - self.czm_shrink_state)
            s = float(np.clip(self.czm_shrink_state, 0.0, 1.0))
            pos = float(np.interp(s, self.czm_levels, np.arange(len(self.czm_levels))))
            i0 = int(np.floor(pos))
            i1 = min(i0 + 1, len(self.czm_levels) - 1)
            f = pos - i0
            self.crack[:] = ((1.0 - f) * self.czm_frames[i0] +
                             f * self.czm_frames[i1])
            return
        shrink = self.soil.shrinkage
        clay = self.soil.clay
        E = 25.0e6
        KIC = self.fracture_kic
        sigma = E * 0.012 * shrink * (0.45 + 0.90 * clay)
        cell = max(12.0, self.size / 9.0)
        a_m = cell * 0.5 * 0.003
        k1 = sigma * np.sqrt(np.pi * a_m)
        ratio = np.clip((k1 - KIC) / KIC, 0.0, 1.0)

        cod_px = 2.0 * sigma * a_m / E / 0.003 * 4.0
        mean_ratio = float(ratio.mean())
        w = 0.45 + float(np.clip(cod_px.mean(), 0.0, 2.2)) + 1.10 * mean_ratio
        edge_mask = 1.0 - smoothstep(w * 0.5, w * 1.9, self.crack_edge)
        opening = np.clip(ratio * (0.45 + 0.55 * clay), 0.0, 1.0)
        self.stress = sigma
        target = np.clip(edge_mask * opening, 0.0, 1.0)
        # 开启/闭合都按指数松弛：τ_open ~ 12 h，τ_close ~ 8 h。
        tau = np.where(target < self.crack, self.crack_tau_close, self.crack_tau_open)
        alpha = 1.0 - np.exp(-self.dt_min / np.maximum(tau, 1.0))
        self.crack += alpha * (target - self.crack)
        self.crack_target = target

# ==================== 画布 ====================
class TerrainCanvas(ArrayCanvas):
    def __init__(self, sim, app=None):
        super().__init__(minimum=(420, 420), renderer=self._render_image)
        self.sim = sim
        self.app = app
        self.show_water = True
        self.show_cracks = True
        self.show_shade = True
        self._soil_lut = None
        self._drawing = False

    def set_sim(self, sim):
        self.sim = sim
        self.invalidate()

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
        wetfield = _blur3(sim.soil.wetness)
        dry = np.array([188, 162, 118], np.float32)
        wetc = np.array([118, 92, 66], np.float32)
        rgb = dry[None, None, :] * (1.0 - wetfield[..., None]) + \
            wetc[None, None, :] * wetfield[..., None]

        # 粗砂（细颗粒被冲走的地方）偏灰、颗粒感强；细泥偏暖
        coarse = np.clip((0.62 - sim.fines) * 1.8, 0.0, 1.0)[..., None]
        rgb = rgb * (1 - coarse * 0.45) + np.array([172, 168, 160], np.float32) * (coarse * 0.45)
        rgb += sim.grain[..., None] * (7.0 + 10.0 * coarse)

        # 积水 / 湿润
        if self.show_water:
            water = np.clip(sim.soil.pond / 3.0, 0.0, 1.0)[..., None]
            rgb = rgb * (1 - water * 0.75) + np.array([70, 118, 165], np.float32) * (water * 0.75)

        # 龟裂
        if self.show_cracks:
            crack = np.clip(_blur3(sim.crack) * 1.2, 0.0, 1.0)[..., None]
            rgb = rgb * (1 - crack * 0.85) + np.array([40, 32, 25], np.float32) * (crack * 0.85)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ---------- 鼠标：局部降雨 ----------
    def _to_sim(self, pos):
        uv = self.map_pos(pos)
        if uv is None:
            return None
        return uv[0] * self.sim.size, uv[1] * self.sim.size

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
        self.setWindowTitle("泥地干裂 · 土壤水")
        self.setGeometry(120, 70, 1080, 800)

        self.size = 320
        self.sim = ErosionSim(self.size, seed=None)
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
        self.lbl_dry = QLabel("含水 –")
        self.lbl_water = QLabel("地表水 0.0 mm")
        self.lbl_drops = QLabel("雨滴 0")
        self.lbl_fps = QLabel("帧率 –")
        for w in (self.lbl_dry, self.lbl_water, self.lbl_drops, self.lbl_fps):
            self.statusBar().addPermanentWidget(w)
        self.fps_counter = FPSCounter()

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
        self.rain_slider.setRange(0, 60)
        self.rain_slider.setValue(int(self.sim.rain_mm_h))
        self.rain_slider.valueChanged.connect(self.on_rain_rate)
        self.rain_label = QLabel(f"{self.sim.rain_mm_h:.0f} mm/h")
        self.rain_label.setFixedWidth(56)
        self.rain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        r = QHBoxLayout()
        r.setSpacing(4)
        r.addWidget(self.rain_slider, 1)
        r.addWidget(self.rain_label)
        form.addRow("雨强", r)

        self._slider(form, "气温", 5, 40, int(self.sim.temp_c),
                     lambda v: setattr(self.sim, "temp_c", float(v)))
        self._slider(form, "湿度", 20, 100, int(self.sim.rh),
                     lambda v: setattr(self.sim, "rh", float(v)))
        self._slider(form, "风速", 0, 8, int(self.sim.wind),
                     lambda v: setattr(self.sim, "wind", float(v)))

        reset_btn = QPushButton("重置泥面")
        reset_btn.clicked.connect(self.reset)
        form.addRow("", reset_btn)
        vbox.addLayout(form)
        layout.addWidget(group)

    def _build_erosion_group(self, layout):
        group = QGroupBox("土壤")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._slider(form, "黏粒", 5, 75, int(self.sim.soil_clay * 100),
                     lambda v: self.sim.set_clay(v / 100.0))
        self._slider(form, "韧性", 10, 60, int(self.sim.fracture_kic / 1000.0),
                     lambda v: setattr(self.sim, "fracture_kic", v * 1000.0))
        layout.addWidget(group)

    def _slider(self, form, text, lo, hi, value, setter):
        """一行滑块 + 数值标签，具体外观交给公共 LabeledSlider。"""
        widget = LabeledSlider("", lo, hi, value, value_width=38)
        widget.bind(setter, emit_now=False)
        form.addRow(text, widget)
        return widget

    def _build_display_group(self, layout):
        group = QGroupBox("显示")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(2)
        for text, attr, default in (("积水 / 湿润", "show_water", True),
                                    ("干燥龟裂", "show_cracks", True),
                                    ("CZM 裂缝（离线）", "show_czm", False)):
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.toggled.connect(lambda checked, a=attr: self._toggle(a, checked))
            vbox.addWidget(cb)
        layout.addWidget(group)

    def _toggle(self, attr, checked):
        if attr == "show_czm":
            self.sim.show_czm = bool(checked)
            self.sim._update_cracks()
            self.canvas.invalidate()
            return
        setattr(self.canvas, attr, bool(checked))
        self.canvas.invalidate()

    # ---------- 交互 ----------
    def toggle_rain(self):
        self.sim.raining = not self.sim.raining
        self.rain_btn.setText("停雨" if self.sim.raining else "降雨")
        if self.sim.raining:
            self.sim.drying = False
            self.dry_btn.setText("干燥")
            self.lbl_status.setText("降雨入渗/径流中；裂缝随湿润闭合…")

    def toggle_dry(self):
        self.sim.drying = not self.sim.drying
        self.dry_btn.setText("停止干燥" if self.sim.drying else "干燥")
        if self.sim.drying:
            self.sim.raining = False
            self.rain_btn.setText("降雨")
            if float(self.sim.crack.mean()) < 0.005:
                # 裂缝已闭合：换一组图案，避免每次干裂同一张图。
                self.sim.new_crack_pattern()
            self.lbl_status.setText("干燥中：K_I ≥ K_IC 时裂缝缓慢张开…")

    def reset(self):
        self.sim.reset()
        self.canvas.invalidate()
        self.update_stats()
        self.lbl_status.setText("地形与土壤水已重置")

    def on_rain_rate(self, v):
        self.sim.rain_mm_h = float(v)
        self.rain_label.setText(f"{v} mm/h")

    def tick(self):
        sim = self.sim
        busy = (sim.raining or sim.drying or sim.count > 0 or
                sim.crack_relaxing or sim.czm_relaxing)
        if not busy:
            return
        sim.step()
        self.canvas.invalidate()
        self.update_stats()
        fps = self.fps_counter.tick()
        if fps is not None:
            self.lbl_fps.setText(f"帧率 {fps:.0f}")

    def update_stats(self):
        wet = float(self.sim.soil.wetness.mean()) * 100.0
        self.lbl_dry.setText(f"表土含水 {wet:.0f}%")
        self.lbl_water.setText(f"地表水 {float(self.sim.soil.pond.mean()):.1f} mm")
        self.lbl_drops.setText(f"雨滴 {self.sim.count:,}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    raise SystemExit(run_app(ErosionApp))
