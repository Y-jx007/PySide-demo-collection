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
        self.month = month
        p0, d = self._mslp()
        p = _blur_y(_blur_x(p0, 6.0), 4.0)
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
# ==================== 画布 ====================
class MapCanvas(QWidget):
    """纸面风格：米白底 + 灰色海岸线 + 逐像素拖尾的细流线（照参考 shader 的做法）。"""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(640, 380)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(247, 245, 240))
        app = self.app
        if app.frame is None:
            p.end()
            return
        w, h = self.width(), self.height()
        scale = min((w - 20) / 2.0, (h - 20) / 1.0)
        mw, mh = scale * 2.0, scale
        x0 = (w - mw) / 2.0
        y0 = (h - mh) / 2.0
        p.drawImage(QRectF(x0, y0, mw, mh), app.frame)
        p.setPen(QPen(QColor(120, 120, 126, 120), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x0, y0, mw, mh))
        p.end()


# ==================== 主窗口 ====================
class CirculationApp(QMainWindow):
    TW, TH = 480, 216          # 拖尾缓冲分辨率（网格的 3 倍）

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
        self._month_f = 1.0
        self._rng = np.random.default_rng(11)

        self.frame = None
        self._base = None                      # 静态底图（纸面 + 海岸线 + 经纬网）
        self._trail_w = np.zeros((self.TH, self.TW), np.float32)
        self._trail_o = np.zeros((self.TH, self.TW), np.float32)
        self._init_particles()
        self._build_base()

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self._t_prev = time.perf_counter()
        self._fps = 0.0

    # ---------- 粒子 ----------
    def _init_particles(self):
        oy, ox = np.nonzero(self.model.ocean)
        n_w, n_o = 7000, 2500
        wi = self._rng.integers(0, len(ox), n_w)
        self.wx = ox[wi].astype(np.float64) + self._rng.random(n_w)
        self.wy = oy[wi].astype(np.float64) + self._rng.random(n_w)
        oi = self._rng.integers(0, len(ox), n_o)
        self.oxp = ox[oi].astype(np.float64) + self._rng.random(n_o)
        self.oyp = oy[oi].astype(np.float64) + self._rng.random(n_o)
        self._oy, self._ox = oy, ox

    def _respawn(self, arr_x, arr_y, n, ocean_only=False):
        idx = self._rng.integers(0, len(self._ox), n)
        arr_x[:] = self._ox[idx] + self._rng.random(n)
        arr_y[:] = self._oy[idx] + self._rng.random(n)

    def _advect(self, dt):
        self.wx_prev = self.wx.copy()
        self.wy_prev = self.wy.copy()
        self.ox_prev = self.oxp.copy()
        self.oy_prev = self.oyp.copy()
        sp = 1.35 * dt
        # 风
        u, v = self.model.wind
        wu = _bilinear(u, self.wx, self.wy)
        wv = _bilinear(v, self.wx, self.wy)
        wu, wv = _display_speed(wu, wv, 2.2)
        self.wx = np.mod(self.wx + wu * sp, W)
        self.wy = np.clip(self.wy + wv * sp, 0.4, H - 1.4)
        bad = np.nonzero((self._rng.random(len(self.wx)) < 0.004)
                         | (self.wy <= 0.41) | (self.wy >= H - 1.41))[0]
        if len(bad):
            self._respawn(self.wx[bad], self.wy[bad], len(bad))
        # 洋流
        cu = _bilinear(self.model.ocean_u, self.oxp, self.oyp)
        cv = _bilinear(self.model.ocean_v, self.oxp, self.oyp)
        cu, cv = _display_speed(cu, cv, 1.6)
        self.oxp = np.mod(self.oxp + cu * sp, W)
        self.oyp = np.clip(self.oyp + cv * sp, 0.6, H - 1.6)
        ix = self.oxp.astype(np.int32) % W
        iy = np.clip(self.oyp.astype(np.int32), 0, H - 1)
        bad = np.nonzero((self._rng.random(len(self.oxp)) < 0.003)
                         | self.model.land[iy, ix])[0]
        if len(bad):
            self._respawn(self.oxp[bad], self.oyp[bad], len(bad))

    # ---------- 拖尾 ----------
    def _splat(self, buf, xs, ys, xp, yp, k=2):
        """把粒子打进拖尾缓冲；沿上一帧到这一帧的线段插值，流线才连续。

        模型第 0 行是南极，写进缓冲时要翻过来。
        """
        buf *= 0.88
        for i in range(k):
            f = i / float(k)
            xa = xp + (xs - xp) * f
            ya = yp + (ys - yp) * f
            tx = np.clip((xa / W * self.TW).astype(np.int32), 0, self.TW - 1)
            ty = np.clip(((H - 1 - ya) / H * self.TH).astype(np.int32), 0, self.TH - 1)
            np.maximum.at(buf, (ty, tx), 1.0)

    # ---------- 静态底图 ----------
    def _build_base(self):
        land = np.flipud(self.model.land)
        p = np.flipud(self.model.p)
        TH, TW = self.TH, self.TW

        paper = np.array([247, 244, 237], np.float32)
        land_tint = np.array([240, 235, 224], np.float32)
        rgb = np.where(_upsample_mask(land, TH, TW)[..., None], land_tint, paper)

        if self.show_pressure:
            # 参考 shader 的画法：低压蓝、高压橙，叠在纸面上
            lo = np.clip((1012.0 - p) / 12.0, 0.0, 1.0)[..., None]
            hi = np.clip((p - 1012.0) / 12.0, 0.0, 1.0)[..., None]
            low_c = np.array([0.0, 0.5, 1.0], np.float32) * 255.0
            high_c = np.array([1.0, 0.5, 0.0], np.float32) * 255.0
            tint = low_c * lo + high_c * hi
            amt = (0.5 * (lo + hi))
            mask = _upsample_mask(np.ones_like(p, bool), TH, TW)[..., None]
            rgb = np.where(mask, rgb * (1 - amt) + tint * amt, rgb)
            # 等压线
            iso = np.zeros_like(p)
            for lv in np.arange(984.0, 1046.0, 4.0):
                iso = np.maximum(iso, np.clip(1.0 - np.abs(p - lv) / 0.5, 0.0, 1.0))
            iso_big = _upsample_f(iso, TH, TW)[..., None]
            rgb = rgb * (1 - 0.5 * iso_big) + np.array([40, 40, 44], np.float32) * (0.5 * iso_big)

        # 海岸线：land 与周围不同处
        coast = np.zeros_like(land)
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        coast_big = _upsample_mask(coast, TH, TW)[..., None]
        rgb = rgb * (1 - 0.8 * coast_big) + np.array([104, 100, 94], np.float32) * (0.8 * coast_big)

        if self.show_grid:
            g = np.zeros((TH, TW), np.float32)
            for k in range(1, 12):
                g[:, int(TW * k / 12)] = 1.0
            for k in range(1, 6):
                g[int(TH * k / 6), :] = 1.0
            g = _upsample_f(g, TH, TW)[..., None]
            rgb = rgb * (1 - 0.22 * g) + np.array([60, 90, 150], np.float32) * (0.22 * g)

        self._base = np.clip(rgb, 0, 255).astype(np.uint8)
        self._ocean_mask = _upsample_mask(np.flipud(self.model.ocean), TH, TW)[..., None].astype(np.float32)

    # ---------- 合成 ----------
    def _compose(self):
        out = self._base.astype(np.float32)
        if self.show_wind:
            a = np.clip(self._trail_w, 0, 1)[..., None] * 0.62
            out = out * (1 - a) + np.array([66, 58, 50], np.float32) * a
        if self.show_ocean:
            a = np.clip(self._trail_o, 0, 1)[..., None] * 0.55 * self._ocean_mask
            out = out * (1 - a) + np.array([18, 108, 132], np.float32) * a
        img = np.clip(out, 0, 255).astype(np.uint8)
        self.frame = QImage(np.ascontiguousarray(img).data, self.TW, self.TH,
                            self.TW * 3, QImage.Format_RGB888).copy()

    # ---------- 循环 ----------
    def tick(self):
        now = time.perf_counter()
        dt = min(3.0, max(0.2, (now - self._t_prev) * 60.0))
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(1e-3, now - self._t_prev))
        self._t_prev = now

        if self.auto_season:
            self._month_f += 0.010
            if self._month_f >= 13.0:
                self._month_f = 1.0
            m = int(round(self._month_f))
            if m != self.model.month:
                self.model.update(m)
                self._build_base()
                if hasattr(self, "lbl_month"):
                    self.lbl_month.setText(f"{m} 月")

        self._advect(dt)
        if self.show_wind:
            self._splat(self._trail_w, self.wx, self.wy, self.wx_prev, self.wy_prev)
        else:
            self._trail_w *= 0.88
        if self.show_ocean:
            self._splat(self._trail_o, self.oxp, self.oyp, self.ox_prev, self.oy_prev)
        else:
            self._trail_o *= 0.88
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

        self.lbl_status = QLabel("白色为纸面，深色细线为风，青色为洋流；勾选「气压分布」可看高压/低压区")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_fps = QLabel("–")
        self.lbl_fps.setMinimumWidth(70)
        self.lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addPermanentWidget(self.lbl_fps)

    def _on_month(self, v):
        self.chk_auto.setChecked(False)
        self._month_f = float(v)
        self.model.update(int(v))
        self._build_base()
        self.lbl_month.setText(f"{v} 月")
        self._compose()
        self.canvas.update()

    def _on_auto(self, on):
        self.auto_season = bool(on)

    def _toggle(self, attr, checked):
        setattr(self, attr, bool(checked))
        self._build_base()
        self._compose()
        self.canvas.update()

    def _reseed(self):
        self._init_particles()
        self._trail_w[:] = 0
        self._trail_o[:] = 0
        self.canvas.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def _upsample_f(a, th, tw):
    """把小网格放大到拖尾缓冲尺寸（最近邻即可，底图本身会被再缩放着色）。"""
    ys = (np.arange(th) * a.shape[0] / th).astype(np.int32)
    xs = (np.arange(tw) * a.shape[1] / tw).astype(np.int32)
    return a[np.clip(ys, 0, a.shape[0] - 1)][:, np.clip(xs, 0, a.shape[1] - 1)]


def _upsample_mask(a, th, tw):
    ys = (np.arange(th) * a.shape[0] / th).astype(np.int32)
    xs = (np.arange(tw) * a.shape[1] / tw).astype(np.int32)
    return a[np.clip(ys, 0, a.shape[0] - 1)][:, np.clip(xs, 0, a.shape[1] - 1)]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CirculationApp()
    window.show()
    sys.exit(app.exec())
