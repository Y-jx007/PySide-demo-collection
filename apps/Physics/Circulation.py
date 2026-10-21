from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainterPath

import base64
import zlib

from scipy.ndimage import zoom

# ==================== 海陆图 ====================
# 1-bit 世界地图（160×72），与参考 shader 内嵌的是同一份数据，解开即可
_LAND_B64 = (
    "eNqtlLFOHDEQhsfnixYkhJGgoEDsvUFSIhRxV/IW3BvQhSsQa4hEhVC6NLwLju4B0qZKNg0UILEXhDjBYjMzXtsLOpSGKdbrb8ee8b8zdu61AbgZDPIZrI9D1fqgkRU4nn3uR2aQSRxrfiY/APHkvDtbCd7WAbo4vSFWNazLDzfhbRrr8HMR4PojRNPwrrY6g32awTb+w5aaca3F5tJ5g/XCS57YoBNYOlsV/VBcQcLJ8TQynIuKRU8M52L6kil8kzTLWgzVlfWQWIyh8CdldY/WvmKkRGEC6xNzxLYGgRXM6PQLkJhG9lIrqpI3WQ+GifnYMPga41LhKKqo05O0tiJmQJym/Qr2K6Fz3GIa1Dk666OIZMGlXoFIYbtLXOrfJMSTUeFR7ZZSVPHHjHxLjDJRBiZ/8Vp7kI1jxlJrWmudugxFD5nmtU82v7FhrTr0+018s3mVf7D83B7Bb8cSs5eu5UcqQ1ZxyzXaC9+Lni03qTBT8OcWO6unAyOpji9uUwdKfhNXJcWtB4HVtOvdGJmNfvgVI7WSyfgSkEW79RWeXsN8c3VIbt3cTQ+wrVcYWUmOZeHMF6oof52QbpS9GaL6W3vecconc//WcmFUEwSz+UvDem44fHNf4drfm6O8BmHh0cUM93fLPuXfxXTufS9Ic66poPGWKR6CptvflV9hPsSOEflPZIoppPbwUr+HPQNe/acq"
)


def load_land():
    raw = zlib.decompress(base64.b64decode(_LAND_B64))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))
    return bits[: 160 * 72].reshape(72, 160).astype(bool)


# ==================== 环流模型 ====================
W, H = 160, 72


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur_x(a, sigma):
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([a[:, -r:], a, a[:, :r]], axis=1)
    out = np.zeros_like(a)
    for i, wgt in enumerate(k):
        out += wgt * pad[:, i:i + a.shape[1]]
    return out


def _blur_y(a, sigma):
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i, wgt in enumerate(k):
        out += wgt * pad[i:i + a.shape[0], :]
    return out


def _bilinear(field, x, y):
    x = np.mod(x, W - 1e-6)
    y = np.clip(y, 0.0, H - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    x1 = (x0 + 1) % W
    y1 = np.minimum(y0 + 1, H - 1)
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
            + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def _lic(u, v, iters=26, step=0.85, seed=5):
    """线积分卷积：把噪声沿流场来回拖，得到连续的流线纹理。"""
    rng = np.random.default_rng(seed)
    noise = rng.random((H, W)).astype(np.float32)
    noise = _blur_x(_blur_y(noise, 0.9), 0.9)
    acc = np.zeros((H, W), np.float32)
    for sgn in (1.0, -1.0):
        x = np.tile(np.arange(W, dtype=np.float32), (H, 1))
        y = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)
        for _ in range(iters):
            x = np.mod(x + _bilinear(u, x, y) * sgn * step, W)
            y = np.clip(y + _bilinear(v, x, y) * sgn * step, 0, H - 1)
            acc += _bilinear(noise, x, y)
    acc /= (2.0 * iters)
    lo, hi = np.percentile(acc, 3), np.percentile(acc, 97)
    return np.clip((acc - lo) / max(1e-6, hi - lo), 0, 1)


def _display_speed(u, v, v_ref):
    "保留方向，把速度压到可显示的区间：慢风也看得见，快风不会糊成一片。"
    mag = np.hypot(u, v)
    k = np.tanh(mag / v_ref) / np.maximum(mag, 1e-9)
    return u * k, v * k


class Circulation:
    """理想化全球大气环流 + 风生洋流。

    气压：按纬度给一条解析式剖面，海陆取不同参数，再各向异性高斯模糊；
    风：地转风（气压梯度旋转 90°）叠加科里奥利参数 f = 15·sin(lat)；
    洋流：风应力旋度做强迫，解泊松方程 ∇²ψ = -curl(τ) 得到流函数，
          u = -∂ψ/∂y, v = ∂ψ/∂x，陆地为狄利克雷边界。
    """

    def __init__(self):
        self.land = load_land()
        self.ocean = ~self.land
        ys = np.arange(H)[:, None] * np.ones((1, W))
        self.lat = 180.0 * ys / H - 90.0            # 第 0 行是南极
        self.lon = 360.0 * np.ones((H, 1)) * np.arange(W)[None, :] / W - 180.0
        self.month = 1
        self._solve_psi_setup()
        self.update(1)

    # ---------- 气压 ----------
    def _mslp(self):
        lat = self.lat
        land = self.land
        p = np.where(
            land,
            1012.5 - 6.0 * np.cos(lat * np.pi / 45.0),
            1014.5 - 20.0 * np.cos(lat * np.pi / 30.0),
        )
        d = np.where(
            land,
            15.0 * np.sin(lat * np.pi / 90.0),
            20.0 * np.sin(lat * np.pi / 35.0) * np.abs(lat) / 90.0,
        )
        return p, d

    def update(self, month):
        self.month = month          # 允许浮点：季节是连续量
        p0, d = self._mslp()
        # 参考里 mslp 返回 vec4（年均值 + 季节差），pass1/pass2 把两个分量
        # 一起模糊；只糊年均值会让季节项保留海陆硬边界，看起来像横条被截断
        p = _blur_y(_blur_x(p0, 6.0), 4.0)
        d = _blur_y(_blur_x(d, 6.0), 4.0)
        # 季节项：1/7 月反相
        t = float(month)
        delta = d * (1.0 - 2.0 * _smoothstep(1.5, 4.5, t) + 2.0 * _smoothstep(7.5, 10.5, t))
        self.p = p + delta
        self._wind()
        self._currents()

    def _wind(self):
        dp_dy, dp_dx = np.gradient(self.p)
        f = 15.0 * np.sin(np.deg2rad(self.lat))
        # 地转分量（旋转 90°）+ 直接的梯度项（摩擦/越赤道流）
        wx = -f * dp_dy - dp_dx
        wy = f * dp_dx - dp_dy
        self.wind = np.stack([wx, wy])

    # ---------- 洋流：风应力旋度 → 流函数 ----------
    def _solve_psi_setup(self):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        # 海洋格点编号
        self.idx = -np.ones((H, W), np.int32)
        oy, ox = np.nonzero(self.ocean)
        self.idx[oy, ox] = np.arange(len(oy))
        n = len(oy)
        k = np.arange(n)
        rows = [k]
        cols = [k]
        vals = [np.full(n, -4.0)]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = oy + dy
            xx = (ox + dx) % W
            ok = (yy >= 0) & (yy < H) & self.ocean[np.clip(yy, 0, H - 1), xx]
            rows.append(k[ok])
            cols.append(self.idx[yy[ok], xx[ok]])
            vals.append(np.ones(int(ok.sum())))
        self._A = sp.csc_matrix((np.concatenate(vals),
                                 (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n, n))
        self._lu = spla.splu(self._A)
        self._oy, self._ox = oy, ox
        self._n_ocean = n

    def _currents(self):
        t_x, t_y = self.wind
        dty_dx = np.roll(t_y, -1, axis=1) - np.roll(t_y, 1, axis=1)
        dtx_dy = np.zeros_like(t_x)
        dtx_dy[1:-1] = 0.5 * (t_x[2:] - t_x[:-2])
        curl = 0.5 * dty_dx - dtx_dy
        rhs = np.zeros(self._n_ocean)
        rhs = curl[self._oy, self._ox]
        psi = self._lu.solve(rhs)
        full = np.zeros((H, W))
        full[self._oy, self._ox] = psi
        dpsi_dy, dpsi_dx = np.gradient(full)
        self.psi = full
        self.ocean_u = -dpsi_dy
        self.ocean_v = dpsi_dx
        scale = np.abs(self.ocean_u).max()
        if scale > 1e-9:
            # 归一到与风速可比的量级，方便同一套粒子参数显示
            wmax = max(1e-9, np.abs(self.wind).max())
            k = 0.55 * wmax / scale
            self.ocean_u *= k
            self.ocean_v *= k


# ==================== 画布 ====================
_STREAK_W = (26.0, 77.0, 128.0)
_STREAK_O = (24.0, 110.0, 128.0)


class MapCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(640, 380)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(232, 231, 225))
        app = self.app
        if app.frame is None:
            p.end()
            return
        w, h = self.width(), self.height()
        scale = min((w - 16) / 2.0, (h - 16) / 1.0)
        mw, mh = scale * 2.0, scale
        x0 = (w - mw) / 2.0
        y0 = (h - mh) / 2.0
        p.drawImage(QRectF(x0, y0, mw, mh), app.frame)
        p.end()


# ==================== 主窗口 ====================
class CirculationApp(QMainWindow):
    """照 shadertoy 的 flow map 实现：每个像素最多存一个粒子，粒子每帧移动
    |v|≤1 像素，拖尾值每帧乘 0.9，粒子落到的像素置 1，最后按参考的 PAPER 配色
    反相成米白纸面 + 蓝灰细流线。"""

    BW, BH = 720, 324          # 缓冲分辨率（2:1，和地图一致）

    def __init__(self):
        super().__init__()
        self.setWindowTitle("全球环流 · 气压 / 风 / 洋流")
        self.setGeometry(110, 60, 1180, 780)

        self.model = Circulation()
        self.show_wind = True
        self.show_ocean = True
        self.show_pressure = False
        self.show_grid = True
        self.auto_season = True
        self._season_t = 1.0
        self._season_tick = 0
        self._pressure_rgb = None
        self._rng = np.random.default_rng(7)

        self.frame = None
        self._base = None
        self._dens_w = np.zeros((self.BH, self.BW), np.float32)
        self._dens_o = np.zeros((self.BH, self.BW), np.float32)
        self._buf_i = np.full((self.BH, self.BW), -1.0, np.float32)
        self._buf_j = np.full((self.BH, self.BW), -1.0, np.float32)
        self._px_w = np.full((self.BH, self.BW), -1.0, np.float32)
        self._py_w = np.full((self.BH, self.BW), -1.0, np.float32)
        self._px_o = np.full((self.BH, self.BW), -1.0, np.float32)
        self._py_o = np.full((self.BH, self.BW), -1.0, np.float32)

        self._build_velocity()
        self._build_base()

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self._t_prev = time.perf_counter()
        self._fps = 0.0

    # ---------- 速度场放大到缓冲分辨率 ----------
    def _build_velocity(self):
        wx, wy = self.model.wind
        ou = np.where(self.model.land, 0.0, self.model.ocean_u)
        ov = np.where(self.model.land, 0.0, self.model.ocean_v)

        def up(f):
            # 模型第 0 行是南极；缓冲第 0 行是北，所以先翻转
            g = np.flipud(f)
            ys = (np.arange(self.BH) * g.shape[0] / self.BH).astype(np.int32)
            xs = (np.arange(self.BW) * g.shape[1] / self.BW).astype(np.int32)
            return g[np.clip(ys, 0, g.shape[0] - 1)][:, np.clip(xs, 0, g.shape[1] - 1)]

        # 参考里 getVelocity 把速度归一到 ≤1 像素/帧
        def norm(u, v, ref):
            mag = np.hypot(u, v)
            k = np.clip(mag / max(1e-6, ref), 0.0, 1.0) / np.maximum(mag, 1e-6)
            return (u * k).astype(np.float32), (v * k).astype(np.float32)

        vxu, vyu = norm(up(wx), up(wy), np.percentile(np.hypot(wx, wy), 90))
        vox, voy = norm(up(ou), up(ov), max(1e-6, np.percentile(np.hypot(ou, ov), 90)))
        # 缓冲 y 向下，模型 v 向北 → 取负
        self._vx_w, self._vy_w = vxu * 1.0, -vyu * 1.0
        self._vx_o, self._vy_o = vox * 0.9, -voy * 0.9
        self._ocean_mask = up(self.model.ocean.astype(np.float32))

    # ---------- 静态底图 ----------
    def _build_base(self):
        BH, BW = self.BH, self.BW
        paper = np.array([232, 231, 225], np.float32)
        rgb = np.tile(paper, (BH, BW, 1))

        land = np.flipud(self.model.land)
        coast = np.zeros_like(land)
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        ys = (np.arange(BH) * land.shape[0] / BH).astype(np.int32)
        xs = (np.arange(BW) * land.shape[1] / BW).astype(np.int32)
        cb = coast[np.clip(ys, 0, coast.shape[0] - 1)][:, np.clip(xs, 0, coast.shape[1] - 1)]
        cb = cb[..., None].astype(np.float32)
        rgb = rgb * (1 - 0.9 * cb) + np.array([120, 118, 114], np.float32) * (0.9 * cb)

        if self.show_grid:
            g = np.zeros((BH, BW), np.float32)
            for k in range(1, 12):
                g[:, int(BW * k / 12)] = 1.0
            for k in range(1, 6):
                g[int(BH * k / 6), :] = 1.0
            g = g[..., None]
            rgb = np.where(g > 0, rgb * 0.8 + np.array([0, 128, 255], np.float32) * 0.2, rgb)

        self._base = np.clip(rgb, 0, 255).astype(np.uint8)
        self._base_f = self._base.astype(np.float32)
        if not hasattr(self, "_compose_buf"):
            self._compose_buf = np.empty((BH, BW, 3), np.float32)
            self._compose_u8 = np.empty((BH, BW, 3), np.uint8)

    # ---------- 气压层（照参考 shader 的配色） ----------
    def _build_pressure(self):
        """LOW = 蓝，HIGH = 橙；经过 0.9-0.8*rgb 反相后，
        低压呈暖橙色、高压呈淡蓝色，和参考的观测图一致。"""
        p = np.flipud(self.model.p)
        BH, BW = self.BH, self.BW
        ys = (np.arange(BH) * p.shape[0] / BH).astype(np.int32)
        xs = (np.arange(BW) * p.shape[1] / BW).astype(np.int32)
        pb = p[np.clip(ys, 0, p.shape[0] - 1)][:, np.clip(xs, 0, p.shape[1] - 1)]

        floor_p = np.floor(pb)
        t1 = _smoothstep(1000.0, 1012.0, floor_p)
        t2 = _smoothstep(1012.0, 1024.0, floor_p)
        low = np.array([0.0, 0.5, 1.0], np.float32)
        high = np.array([1.0, 0.5, 0.0], np.float32)
        r = low * (1.0 - t1)[..., None]
        r = r * (1.0 - t2[..., None]) + high * t2[..., None]
        rgb = 0.9 - 0.8 * (0.5 * r)
        small = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        from scipy.ndimage import zoom
        big = zoom(small.astype(np.float32), (BH / small.shape[0], BW / small.shape[1], 1), order=1)
        self._pressure_rgb = np.clip(big, 0, 255).astype(np.uint8)

    def _apply_pressure(self):
        base = self._base_f
        if self._pressure_rgb is None:
            return base
        pr = self._pressure_rgb.astype(np.float32)
        # 气压层只做 62% 叠加，海岸线和纸面纹理仍然保留
        return base * 0.38 + pr * 0.62

    # ---------- 粒子推进（照 getPosition 的规则） ----------
    def _step_particles(self, vx, vy, px, py, dens, spawn, die):
        """每个像素最多一个粒子；粒子一步移动 |v|≤1 像素，拖尾 0.9 衰减。"""
        BH, BW = self.BH, self.BW
        alive = px >= 0
        ys, xs = np.nonzero(alive)
        new_px = np.full((BH, BW), -1.0, np.float32)
        new_py = np.full((BH, BW), -1.0, np.float32)

        if len(ys):
            keep = self._rng.random(len(ys)) > die
            ys, xs = ys[keep], xs[keep]
            nx = np.mod(px[ys, xs] + vx[ys, xs], BW)
            ny = np.clip(py[ys, xs] + vy[ys, xs], 0, BH - 1)
            tx = np.clip(nx.astype(np.int32), 0, BW - 1)
            ty = np.clip(ny.astype(np.int32), 0, BH - 1)
            new_px[ty, tx] = nx
            new_py[ty, tx] = ny

        # 空白像素里稀疏地诞生新粒子（用泊松抽样，避免每帧生成整屏随机数）
        empty = (new_px < 0).sum()
        k = self._rng.poisson(spawn * empty)
        if k > 0:
            cy = self._rng.integers(0, BH, k)
            cx = self._rng.integers(0, BW, k)
            new_px[cy, cx] = cx + self._rng.random(k)
            new_py[cy, cx] = cy + self._rng.random(k)

        dens *= 0.90
        dens[new_px >= 0] = 1.0
        return new_px, new_py, dens

    # ---------- 合成 ----------
    def _compose(self):
        out = self._compose_buf
        out[:] = self._apply_pressure() if self.show_pressure else self._base_f
        # 逐通道就地混合，(H,W,1) 广播到 (H,W,3) 太慢
        if self.show_wind:
            a = self._dens_w
            inv = 1.0 - a
            for c in range(3):
                ch = out[:, :, c]
                ch *= inv
                ch += _STREAK_W[c] * a
        if self.show_ocean:
            a = self._dens_o * 0.9 * self._ocean_mask
            inv = 1.0 - a
            for c in range(3):
                ch = out[:, :, c]
                ch *= inv
                ch += _STREAK_O[c] * a
        self._compose_u8[:] = np.clip(out, 0, 255)
        self.frame = QImage(self._compose_u8.data, self.BW, self.BH,
                            self.BW * 3, QImage.Format_RGB888).copy()

    # ---------- 循环 ----------
    def tick(self):
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(1e-3, now - self._t_prev))
        self._t_prev = now

        # 季节连续演化：参考里 t = mod(iTime,12.) 是连续的，不能按月跳变
        if self.auto_season:
            self._season_t += 0.018
            if self._season_t >= 13.0:
                self._season_t -= 12.0
        self._season_tick += 1
        if self._season_tick % 4 == 0:
            self.model.update(self._season_t)
            self._build_velocity()
            if self.show_pressure:
                self._build_pressure()
            if hasattr(self, "lbl_month"):
                self.lbl_month.setText(f"{self._season_t % 12.0 + 0.5:.1f} 月")

        if self.show_wind:
            self._px_w, self._py_w, self._dens_w = self._step_particles(
                self._vx_w, self._vy_w, self._px_w, self._py_w, self._dens_w, 1.6e-4, 5e-3)
        else:
            self._dens_w *= 0.9
        if self.show_ocean:
            self._px_o, self._py_o, self._dens_o = self._step_particles(
                self._vx_o, self._vy_o, self._px_o, self._py_o, self._dens_o, 8e-5, 6e-3)
        else:
            self._dens_o *= 0.9

        self._compose()
        self.canvas.update()
        if hasattr(self, "lbl_fps"):
            self.lbl_fps.setText(f"{self._fps:.0f} fps")

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(272)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("全球环流")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("气压梯度 + 科里奥利力 → 风\n风应力旋度 → 海洋环流")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = MapCanvas(self)

        g1 = QGroupBox("季节")
        f1 = QFormLayout(g1)
        f1.setSpacing(6)
        f1.setContentsMargins(8, 8, 8, 8)
        self.month_s = QSlider(Qt.Horizontal)
        self.month_s.setRange(1, 12)
        self.month_s.setValue(1)
        self.lbl_month = QLabel("1 月")
        self.lbl_month.setFixedWidth(44)
        self.lbl_month.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.month_s.valueChanged.connect(self._on_month)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.month_s, 1)
        row.addWidget(self.lbl_month)
        f1.addRow("月份", row)
        self.chk_auto = QCheckBox("自动推演")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._on_auto)
        f1.addRow(self.chk_auto)
        pv.addWidget(g1)

        g2 = QGroupBox("显示")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(4)
        v2.setContentsMargins(8, 8, 8, 8)
        for text, attr, checked in (("风场流线", "show_wind", True),
                                    ("洋流流线", "show_ocean", True),
                                    ("气压分布", "show_pressure", False),
                                    ("经纬网", "show_grid", True)):
            cb = QCheckBox(text)
            cb.setChecked(checked)
            cb.toggled.connect(lambda c, a=attr: self._toggle(a, c))
            v2.addWidget(cb)
        pv.addWidget(g2)

        btn = QPushButton("重播粒子")
        btn.clicked.connect(self._reseed)
        pv.addWidget(btn)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(8)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("细线是气流（每像素一个粒子，一步移动不到一格，拖尾 0.9 衰减）")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_fps = QLabel("–")
        self.lbl_fps.setMinimumWidth(70)
        self.lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addPermanentWidget(self.lbl_fps)

    def _on_month(self, v):
        self.chk_auto.setChecked(False)
        self._season_t = float(v)
        self.model.update(float(v))
        self._build_velocity()
        if self.show_pressure:
            self._build_pressure()
        self.lbl_month.setText(f"{v} 月")
        self.canvas.update()

    def _on_auto(self, on):
        self.auto_season = bool(on)

    def _toggle(self, attr, checked):
        setattr(self, attr, bool(checked))
        if attr == "show_pressure" and checked:
            self._build_pressure()
        self._build_base()
        self._compose()
        self.canvas.update()

    def _reseed(self):
        for a in ("_px_w", "_py_w", "_px_o", "_py_o"):
            getattr(self, a)[:] = -1
        self._dens_w[:] = 0
        self._dens_o[:] = 0
        self.canvas.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CirculationApp()
    window.show()
    sys.exit(app.exec())
