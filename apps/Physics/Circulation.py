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
        self.month = 4
        self._solve_psi_setup()
        self.update(4)

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
class MapCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(640, 380)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.fillRect(self.rect(), QColor(250, 249, 246))
        app = self.app
        if app.image is None:
            p.end()
            return

        w, h = self.width(), self.height()
        # 地图按 2:1 摆，等比缩放
        scale = min((w - 24) / 2.0, (h - 24) / 1.0)
        mw, mh = scale * 2.0, scale
        x0 = (w - mw) / 2.0
        y0 = (h - mh) / 2.0
        rect = QRectF(x0, y0, mw, mh)
        p.drawImage(rect, app.image_big)

        sx, sy = mw / W, mh / H
        if app.show_wind and app.wind_hist:
            self._draw_trails(p, app.wind_hist, x0, y0, sx, sy, QColor(48, 60, 84, 165), 1.0)
        if app.show_ocean and app.ocean_hist:
            self._draw_trails(p, app.ocean_hist, x0, y0, sx, sy, QColor(20, 126, 148, 185), 1.2)

        p.setPen(QPen(QColor(120, 120, 126, 160), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        p.end()

    def _draw_trails(self, p, hist, x0, y0, sx, sy, color, width):
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        arr = np.stack(hist, 0)             # (K, n, 2)
        K, n, _ = arr.shape
        path = QPainterPath()
        for k in range(n):
            xs = arr[:, k, 0]
            ys = arr[:, k, 1]
            if np.abs(np.diff(xs)).max() > W * 0.5:
                continue                     # 跨日界线断开
            path.moveTo(x0 + xs[0] * sx, y0 + ys[0] * sy)
            for i in range(1, K):
                path.lineTo(x0 + xs[i] * sx, y0 + ys[i] * sy)
        p.drawPath(path)


# ==================== 主窗口 ====================
class CirculationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全球环流 · 气压 / 风 / 洋流")
        self.setGeometry(110, 60, 1180, 760)

        self.model = Circulation()
        self.image = None
        self.image_big = None
        self.show_wind = True
        self.show_ocean = True
        self.show_pressure = True
        self.show_grid = True
        self.wind_hist = []
        self.ocean_hist = []
        self.trail = 9
        self.auto_season = True
        self._month_f = 4.0
        self._rng = np.random.default_rng(11)
        self._init_particles()
        self._rebuild_image()

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self._t_last = time.perf_counter()
        self._fps = 0.0

    # ---------- 粒子 ----------
    def _init_particles(self):
        oy, ox = np.nonzero(self.model.ocean)
        n_w, n_o = 500, 280
        wi = self._rng.integers(0, len(ox), n_w)
        self.wx = ox[wi].astype(np.float64) + self._rng.random(n_w)
        self.wy = oy[wi].astype(np.float64) + self._rng.random(n_w)
        self.wpx = self.wx.copy()
        self.wpy = self.wy.copy()
        self.walive = np.ones(n_w, bool)

        oi = self._rng.integers(0, len(ox), n_o)
        self.ox_ = ox[oi].astype(np.float64) + self._rng.random(n_o)
        self.oy_ = oy[oi].astype(np.float64) + self._rng.random(n_o)
        self.opx = self.ox_.copy()
        self.opy = self.oy_.copy()
        self.oalive = np.ones(n_o, bool)

    def _sample(self, field, x, y):
        x = np.mod(x, W - 1e-6)
        y = np.clip(y, 0, H - 1.001)
        x0 = x.astype(np.int32)
        y0 = y.astype(np.int32)
        fx = x - x0
        fy = y - y0
        x1 = (x0 + 1) % W
        y1 = np.minimum(y0 + 1, H - 1)
        return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
                + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)

    def _advect_particles(self, dt):
        # 风
        u, v = self.model.wind
        sp = 1.1 * dt
        self.wpx, self.wpy = self.wx.copy(), self.wy.copy()
        wu = self._sample(u, self.wx, self.wy)
        wv = self._sample(v, self.wx, self.wy)
        wu, wv = _display_speed(wu, wv, 6.0)
        self.wx = np.mod(self.wx + wu * sp, W)
        self.wy = np.clip(self.wy - wv * sp, 0.5, H - 1.5)
        bad = (self._rng.random(len(self.wx)) < 0.006) | \
              (self.wy <= 0.51) | (self.wy >= H - 1.51)
        oy, ox = np.nonzero(self.model.ocean)
        idx = self._rng.integers(0, len(ox), int(bad.sum()))
        self.wx[bad] = ox[idx] + self._rng.random(int(bad.sum()))
        self.wy[bad] = oy[idx] + self._rng.random(int(bad.sum()))
        self.wpx[bad] = self.wx[bad]
        self.wpy[bad] = self.wy[bad]
        for h in self.wind_hist:
            h[bad, 0] = self.wx[bad]
            h[bad, 1] = self.wy[bad]

        # 洋流（只在海上走）
        ou, ov = self.model.ocean_u, self.model.ocean_v
        self.opx, self.opy = self.ox_.copy(), self.oy_.copy()
        cu = self._sample(ou, self.ox_, self.oy_)
        cv = self._sample(ov, self.ox_, self.oy_)
        cu, cv = _display_speed(cu, cv, 3.5)
        self.ox_ = np.mod(self.ox_ + cu * sp, W)
        self.oy_ = np.clip(self.oy_ - cv * sp, 1.0, H - 2.0)
        ox0 = self.ox_.astype(np.int32) % W
        oy0 = self.oy_.astype(np.int32).clip(0, H - 1)
        landed = self.model.land[oy0, ox0] | (self._rng.random(len(self.ox_)) < 0.004)
        idx = self._rng.integers(0, len(ox), int(landed.sum()))
        self.ox_[landed] = ox[idx] + self._rng.random(int(landed.sum()))
        self.oy_[landed] = oy[idx] + self._rng.random(int(landed.sum()))
        self.opx[landed] = self.ox_[landed]
        self.opy[landed] = self.oy_[landed]
        for h in self.ocean_hist:
            h[landed, 0] = self.ox_[landed]
            h[landed, 1] = self.oy_[landed]

        self.wind_hist.append(np.stack([self.wx, self.wy], 1))
        self.ocean_hist.append(np.stack([self.ox_, self.oy_], 1))
        if len(self.wind_hist) > self.trail:
            self.wind_hist.pop(0)
        if len(self.ocean_hist) > self.trail:
            self.ocean_hist.pop(0)

    # ---------- 底图 ----------
    def _rebuild_image(self):
        land = np.flipud(self.model.land)          # 北在上
        p = np.flipud(self.model.p)

        ocean_rgb = np.array([228, 236, 243], np.float32)
        land_rgb = np.array([201, 193, 178], np.float32)
        rgb = np.where(land[..., None], land_rgb, ocean_rgb).astype(np.float32)

        if self.show_pressure:
            lo, hi = 985.0, 1030.0
            t = np.clip((p - lo) / (hi - lo), 0.0, 1.0)[..., None]
            cold = np.array([70, 130, 200], np.float32)
            warm = np.array([214, 132, 74], np.float32)
            tint = cold * (1 - t) + warm * t
            rgb = rgb * 0.62 + tint * 0.38

        # 海岸线
        coast = np.zeros_like(land)
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        rgb[coast] = np.array([96, 92, 86], np.float32)

        img = np.clip(rgb, 0, 255).astype(np.uint8)
        self.image = QImage(np.ascontiguousarray(img).data, W, H, W * 3,
                            QImage.Format_RGB888).copy()
        big = np.clip(zoom(img.astype(np.float32), (6, 6, 1), order=1), 0, 255).astype(np.uint8)
        self.image_big = QImage(np.ascontiguousarray(big).data, W * 6, H * 6,
                                W * 18, QImage.Format_RGB888).copy()

    # ---------- 循环 ----------
    def tick(self):
        now = time.perf_counter()
        dt = min(3.0, (now - self._t_last) * 60.0)
        self._t_last = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(1e-3, now - getattr(self, "_t_prev", now)))
        self._t_prev = now

        if self.auto_season:
            self._month_f += 0.012
            if self._month_f >= 13.0:
                self._month_f = 1.0
            m = int(round(self._month_f))
            if m != self.model.month:
                self.model.update(m)
                self._rebuild_image()
                if hasattr(self, "lbl_month"):
                    self.lbl_month.setText(f"{m} 月")

        self._advect_particles(dt)
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
        self.month_s.setValue(4)
        self.lbl_month = QLabel("4 月")
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
        for text, attr in (("气压染色", "show_pressure"), ("风场流线", "show_wind"),
                           ("洋流流线", "show_ocean"), ("经纬网", "show_grid")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, a=attr: self._toggle(a, checked))
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

        self.lbl_status = QLabel("风速、气压均由解析气压场与地转关系推出；洋流由风应力旋度解泊松方程得到")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_fps = QLabel("–")
        self.lbl_fps.setMinimumWidth(70)
        self.lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addPermanentWidget(self.lbl_fps)

    def _on_month(self, v):
        self.chk_auto.setChecked(False)
        self._month_f = float(v)
        self.model.update(int(v))
        self._rebuild_image()
        self.lbl_month.setText(f"{v} 月")
        self.canvas.update()

    def _on_auto(self, on):
        self.auto_season = bool(on)

    def _toggle(self, attr, checked):
        setattr(self, attr, bool(checked))
        self._rebuild_image()
        self.canvas.update()

    def _reseed(self):
        self._init_particles()
        self.canvas.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CirculationApp()
    window.show()
    sys.exit(app.exec())
