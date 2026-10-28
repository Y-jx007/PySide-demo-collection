from common.field_canvas import ArrayCanvas
from common.numpy_base import (ndarray_to_qimage, np)
from common.paths import DATA_DIR
from common.widgets import LabeledSlider
from common.qt_base import (QCheckBox, QColor, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                            QMainWindow, QMessageBox, QPen, QPointF, QPushButton, QScrollArea, QTimer, QVBoxLayout,
                            QWidget, Qt, logging, run_app, time)
from common.taichi_base import (ti)

import base64
import zlib

from scipy.ndimage import zoom, gaussian_filter

from support.physics.circulation_data import (
    RealWind, elevation_at, load_elevation, terrain_adjust,
)

# ==================== 海陆图 ====================
# 1-bit 世界地图（160×72），与参考 shader 内嵌的是同一份数据，解开即可
_LAND_B64 = (
    "eNqtlLFOHDEQhsfnixYkhJGgoEDsvUFSIhRxV/IW3BvQhSsQa4hEhVC6NLwLju4B0qZKNg0UILEXhDjBYjMzXtsLOpSGKdbrb8ee8b8zdu61AbgZDPIZrI9D1fqgkRU4nn3uR2aQSRxrfiY/APHkvDtbCd7WAbo4vSFWNazLDzfhbRrr8HMR4PojRNPwrrY6g32awTb+w5aaca3F5tJ5g/XCS57YoBNYOlsV/VBcQcLJ8TQynIuKRU8M52L6kil8kzTLWgzVlfWQWIyh8CdldY/WvmKkRGEC6xNzxLYGgRXM6PQLkJhG9lIrqpI3WQ+GifnYMPga41LhKKqo05O0tiJmQJym/Qr2K6Fz3GIa1Dk666OIZMGlXoFIYbtLXOrfJMSTUeFR7ZZSVPHHjHxLjDJRBiZ/8Vp7kI1jxlJrWmudugxFD5nmtU82v7FhrTr0+018s3mVf7D83B7Bb8cSs5eu5UcqQ1ZxyzXaC9+Lni03qTBT8OcWO6unAyOpji9uUwdKfhNXJcWtB4HVtOvdGJmNfvgVI7WSyfgSkEW79RWeXsN8c3VIbt3cTQ+wrVcYWUmOZeHMF6oof52QbpS9GaL6W3vecconc//WcmFUEwSz+UvDem44fHNf4drfm6O8BmHh0cUM93fLPuXfxXTufS9Ic66poPGWKR6CptvflV9hPsSOEflPZIoppPbwUr+HPQNe/acq"
)


def load_land_fine():
    """720×360 的 Natural Earth 海岸线（第 0 行是北）。"""
    path = DATA_DIR / "worldmask.npz"
    try:
        d = np.load(path)["land"]
        if d.shape == (360, 720):
            return d
    except Exception:
        pass
    # 没有文件时退回内嵌的 160×72 位图（放大）
    raw = zlib.decompress(base64.b64decode(_LAND_B64))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(72, 160)
    ys = (np.arange(360) * 72 / 360).astype(int)
    xs = (np.arange(720) * 160 / 720).astype(int)
    return bits[np.ix_(ys, xs)].astype(bool)


def load_land():
    """物理网格用 160×72（第 0 行是南），从高精度掩膜抽样得到。"""
    fine = load_land_fine()[::-1]                 # 翻成南在下
    ys = (np.arange(72) * fine.shape[0] / 72).astype(int)
    xs = (np.arange(160) * fine.shape[1] / 160).astype(int)
    return fine[np.ix_(ys, xs)].astype(bool)


# ==================== 环流模型 ====================
W, H = 160, 72


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
    Hf, Wf = field.shape          # 必须用数组自己的尺寸（风场是 360×180）
    x = np.mod(x, Wf - 1e-6)
    y = np.clip(y, 0.0, Hf - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    x1 = (x0 + 1) % Wf
    y1 = np.minimum(y0 + 1, Hf - 1)
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


@ti.func
def _ssf(a: ti.f32, b: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _land_at(nw, nh):
    """把 720×360 的海岸线掩膜抽到 nw×nh（第 0 行仍是南）。"""
    fine = load_land_fine()[::-1]
    ys = (np.arange(nh) * fine.shape[0] / nh).astype(int)
    xs = (np.arange(nw) * fine.shape[1] / nw).astype(int)
    return fine[np.ix_(ys, xs)]


class WindSim:
    """气压场与风场，用 Taichi 并行算在 360×180 网格上。

    旧版的解析气压场（不含温度）：海洋连续场 + 海陆差距项，南极陆地
    因此是低压。这里保留 1° 地形，但只用于底图与风速修正。
    """

    NW, NH = 360, 180

    def __init__(self):
        self.land = _land_at(self.NW, self.NH)          # 第 0 行是南
        self.lats = (180.0 * np.arange(self.NH) / self.NH - 90.0).astype(np.float32)
        self.elev = elevation_at(self.NW, self.NH)
        self.elev_s = gaussian_filter(np.maximum(self.elev, 0.0), 1.2).astype(np.float32)

        ti.reset()
        ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=True)
        self.f_land = ti.field(ti.i32, (self.NH, self.NW))
        self.f_p0 = ti.field(ti.f32, (self.NH, self.NW))
        self.f_pd = ti.field(ti.f32, (self.NH, self.NW))
        self.f_p = ti.field(ti.f32, (self.NH, self.NW))
        self.f_b1 = ti.field(ti.f32, (self.NH, self.NW))
        self.f_b2 = ti.field(ti.f32, (self.NH, self.NW))
        self.f_wind = ti.Vector.field(2, ti.f32, (self.NH, self.NW))
        self.f_land.from_numpy(self.land.astype(np.int32))
        self._compile()
        self.t = 1.0
        self.update(1.0)

    def _compile(self):
        NH, NW = self.NH, self.NW
        land, p0, pd, p, b1, b2 = (self.f_land, self.f_p0, self.f_pd,
                                   self.f_p, self.f_b1, self.f_b2)
        wind = self.f_wind

        # 半径 45 的高斯核 = 3σ，σ=15 格（对应参考在 144 宽下的 σ=6）
        R = 45
        SIG = 15.0

        @ti.kernel
        def make_mslp(t: ti.f32, contrast: ti.f32):
            for i, j in p0:
                lat = 180.0 * i / NH - 90.0
                # 先算海洋（连续场），再按 contrast 把陆地拉离海洋
                po = 1014.5 - 20.0 * ti.cos(lat * 3.14159265 / 30.0)
                do = 20.0 * ti.sin(lat * 3.14159265 / 35.0) * ti.abs(lat) / 90.0
                pl = 1012.5 - 6.0 * ti.cos(lat * 3.14159265 / 45.0)
                dl = 15.0 * ti.sin(lat * 3.14159265 / 90.0)
                if land[i, j] != 0:
                    p0[i, j] = po + contrast * (pl - po)
                    pd[i, j] = do + contrast * (dl - do)
                else:
                    p0[i, j] = po
                    pd[i, j] = do
                # 季节项：1/7 月反相（必须是 ti.func，普通 Python 函数会被当成常数）
                amp = 1.0 - 2.0 * _ssf(1.5, 4.5, t) + 2.0 * _ssf(7.5, 10.5, t)
                p[i, j] = p0[i, j] + pd[i, j] * amp

        @ti.kernel
        def blur_x(r: ti.i32, sig: ti.f32):
            for i, j in p:
                acc = 0.0
                wsum = 0.0
                for k in range(-r, r + 1):
                    w = ti.exp(-0.5 * (k / sig) ** 2)
                    acc += p[i, (j + k + NW) % NW] * w
                    wsum += w
                b1[i, j] = acc / wsum

        @ti.kernel
        def blur_y(r: ti.i32, sig: ti.f32):
            for i, j in p:
                acc = 0.0
                wsum = 0.0
                for k in range(-r, r + 1):
                    w = ti.exp(-0.5 * (k / sig) ** 2)
                    ii = ti.min(ti.max(i + k, 0), NH - 1)
                    acc += b1[ii, j] * w
                    wsum += w
                b2[i, j] = acc / wsum

        @ti.kernel
        def wind_field():
            for i, j in wind:
                im = ti.max(i - 1, 0)
                ip = ti.min(i + 1, NH - 1)
                jm = (j - 1 + NW) % NW
                jp = (j + 1) % NW
                dpdy = (b2[ip, j] - b2[im, j]) * 0.5
                dpdx = (b2[i, jp] - b2[i, jm]) * 0.5
                lat = 180.0 * i / NH - 90.0
                f = 15.0 * ti.sin(lat * 3.14159265 / 180.0)
                wind[i, j] = ti.Vector([-f * dpdy - dpdx, f * dpdx - dpdy])

        self._make_mslp = make_mslp
        self._blur_x = blur_x
        self._blur_y = blur_y
        self._wind = wind_field

    def update(self, t, contrast=1.0):
        self.t = float(t)
        self.contrast = float(contrast)
        self._make_mslp(float(t), float(contrast))
        self._blur_x(45, 15.0)
        self._blur_y(45, 15.0)
        self._wind()

    def pressure(self):
        return self.f_b2.to_numpy()

    def wind_numpy(self, terrain_strength=1.0):
        """返回 (NH, NW, 2) 风场；terrain_strength=0 时不做地形修正。"""
        wind = self.f_wind.to_numpy()
        if terrain_strength > 0.0:
            u, v = terrain_adjust(wind[..., 0], wind[..., 1], self.elev_s,
                                  float(terrain_strength))
            wind = np.stack([u, v], axis=-1)
        return wind


# ==================== 画布 ====================
_STREAK_W = (104.0, 134.0, 166.0)
_LINE_RGB = (228.0, 156.0, 132.0)
_STREAK_O = (24.0, 110.0, 128.0)


class MapCanvas(ArrayCanvas):
    def __init__(self, app):
        # 等距柱状投影是 2:1；左右各留 8px
        super().__init__(aspect=2.0, minimum=(640, 380), padding=8,
                         background=QColor(232, 231, 225))
        self.app = app

    def paintEvent(self, event):
        # _compose() 每次都会生成新的 QImage，set_array 直接比对对象即可
        if self.app.frame is not None:
            self.set_array(self.app.frame)
        super().paintEvent(event)

    def paint_overlay(self, painter, rect):
        """特殊纬线标注。"""
        f = painter.font()
        f.setPointSize(8)
        painter.setFont(f)
        for lat, text in ((66.5, "北极圈"), (23.5, "北回归线"), (0.0, "赤道"),
                          (-23.5, "南回归线"), (-66.5, "南极圈")):
            yy = rect.y() + (90.0 - lat) / 180.0 * rect.height()
            painter.setPen(QPen(QColor(196, 132, 112, 170), 1))
            painter.drawText(QPointF(rect.x() + 4, yy - 2), text)


# ==================== 主窗口 ====================
class CirculationApp(QMainWindow):
    """照 shadertoy 的 flow map 实现：每个像素最多存一个粒子，粒子每帧移动
    |v|≤1 像素，拖尾值每帧乘 0.9，粒子落到的像素置 1，最后按参考的 PAPER 配色
    反相成米白纸面 + 蓝灰细流线。"""

    BW, BH = 800, 400          # 缓冲分辨率（2:1，和地图一致）

    def __init__(self):
        super().__init__()
        self.setWindowTitle("全球环流 · 地形 / 真实风场")
        self.setGeometry(150, 60, 1240, 820)

        self.wind = WindSim()
        try:
            self.real = RealWind()
        except Exception:
            self.real = None
        self.real_mode = False
        self.real_layer = "monthly"         # monthly / anomaly / trend
        self.real_year_idx = len(self.real.year_values) - 1 if self.real is not None else 0
        self.real_month_idx = len(self.real.month_values) - 1 if self.real is not None else 0
        self.real_autoplay = False
        self._year_tick = 0
        self.show_wind = True
        self.show_pressure = True
        self.show_grid = True
        self.show_terrain = True
        self.terrain_strength = 1.0
        self.auto_season = True
        self._day = 0.0            # 一年中的第几天
        self._season_t = 1.0
        self._season_tick = 0
        self._pressure_rgb = None
        self._pressure_cur = None
        self._compose_p = np.empty((self.BH, self.BW, 3), np.float32)
        self._rng = np.random.default_rng(7)
        self._contrast = 1.5
        self.speed_scale = 5.0
        self._land_timer = QTimer(self)
        self._land_timer.setSingleShot(True)
        self._land_timer.timeout.connect(self._apply_contrast)
        self._terrain_timer = QTimer(self)
        self._terrain_timer.setSingleShot(True)
        self._terrain_timer.timeout.connect(self._apply_terrain)
        self._change_rgb = None
        self._change_a = None
        self.substeps = 4
        # 初始风场使用界面滑杆的 150% 海陆差距。
        self.wind.update(1.0, self._contrast)
        # speed_scale 由滑杆设置      # 每帧推进的子步数（每步 dt 很小）
        self._buf_x = np.full((self.BH, self.BW), -1.0, np.float32)
        self._buf_y = np.full((self.BH, self.BW), -1.0, np.float32)

        self.frame = None
        self._base = None
        self._dens_w = np.zeros((self.BH, self.BW), np.float32)
        self._buf_i = np.full((self.BH, self.BW), -1.0, np.float32)
        self._buf_j = np.full((self.BH, self.BW), -1.0, np.float32)
        self._px_w = np.full((self.BH, self.BW), -1.0, np.float32)
        self._py_w = np.full((self.BH, self.BW), -1.0, np.float32)

        self._build_velocity()
        self._build_base()

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self._t_prev = time.perf_counter()
        self._fps = 0.0

    # ---------- 风场（粒子直接采样，不再预放大到缓冲） ----------
    def _current_wind(self):
        """合成风场（含地形）或真实月均风场，统一到 WindSim 的网格。"""
        if self.real_mode and self.real is not None:
            w = self.real.field(self.real_year_idx, self.real_month_idx, self.real_layer)
            nh, nw = self.wind.NH, self.wind.NW
            if w.shape[:2] != (nh, nw):
                w = np.stack([
                    zoom(w[..., 0], (nh / w.shape[0], nw / w.shape[1]), order=1),
                    zoom(w[..., 1], (nh / w.shape[0], nw / w.shape[1]), order=1),
                ], axis=-1)
            return w.astype(np.float32)
        return self.wind.wind_numpy(self.terrain_strength)

    def _build_velocity(self):
        w = self._current_wind()                     # (nlat,nlon,2)，第 0 行是南
        nu = w[:, :, 0].astype(np.float32)
        nv = w[:, :, 1].astype(np.float32)
        if getattr(self, "_wu", None) is None:
            self._wu, self._wv = nu, nv
            self._wu_a, self._wv_a = nu.copy(), nv.copy()
        else:
            self._wu_a, self._wv_a = self._wu.copy(), self._wv.copy()
            self._wu_t, self._wv_t = nu, nv
            self._wphase = 0.0
        mag = np.hypot(nu, nv)
        self._wref = max(1e-6, float(np.percentile(mag, 80)))
        self._wgain = self.BW / 144.0 * 0.65          # 参考的 1 像素/帧是相对 144 宽

    # ---------- 静态底图 ----------
    def _build_base(self):
        BH, BW = self.BH, self.BW
        paper = np.array([232, 231, 225], np.float32)
        rgb = np.tile(paper, (BH, BW, 1))

        fine = load_land_fine()                      # 720×360，北在上
        ys = (np.arange(BH) * fine.shape[0] / BH).astype(np.int32)
        xs = (np.arange(BW) * fine.shape[1] / BW).astype(np.int32)
        land = fine[np.ix_(ys, xs)]

        # 地形底图：ETOPO1 海拔配色 + 坡向阴影。数据缺失时退回纸面底色。
        if self.show_terrain:
            try:
                elev, _, _ = load_elevation()
                e = elev[::-1].astype(np.float32)          # 北在上
                e = zoom(e, (BH / e.shape[0], BW / e.shape[1]), order=1)[:BH, :BW]
                h = np.maximum(e, 0.0)
                stops = np.array([0.0, 300.0, 1000.0, 2500.0, 5000.0], np.float32)
                cols = np.array([[204, 197, 176], [188, 171, 139], [160, 139, 112],
                                 [138, 126, 116], [244, 246, 249]], np.float32)
                terr = np.empty((BH, BW, 3), np.float32)
                for c in range(3):
                    terr[:, :, c] = np.interp(h, stops, cols[:, c])
                gy, gx = np.gradient(h)
                slope = np.hypot(gx, gy)
                ref = max(1.0, float(np.percentile(slope, 90)))
                light = -(gx + gy) / ref
                shade = np.clip(1.0 + 0.22 * light, 0.78, 1.18)
                terr *= shade[:, :, None]
                # 高海拔（冰盖/雪山）提亮，避免整块 Antarctica 变成深灰。
                ice = (h > 2200.0)[..., None]
                terr = np.where(ice, terr * 0.35 + np.array([246, 248, 251], np.float32) * 0.65, terr)
                a = land[..., None].astype(np.float32) * 0.62
                rgb = rgb * (1.0 - a) + terr * a
            except Exception as exc:
                logging.warning("地形数据加载失败: %s", exc)

        coast = np.zeros_like(land)
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        cb = coast[..., None].astype(np.float32)
        rgb = rgb * (1 - 0.95 * cb) + np.array([72, 68, 62], np.float32) * (0.95 * cb)

        if self.show_grid:
            g = np.zeros((BH, BW), np.float32)
            for k in range(1, 12):                    # 经线每 30°
                g[:, int(BW * k / 12)] = 1.0
            for k in range(1, 6):                     # 纬线每 30°
                g[int(BH * k / 6), :] = 1.0
            g = g[..., None]
            rgb = np.where(g > 0, rgb * 0.82 + np.array([0, 128, 255], np.float32) * 0.18, rgb)

        # 特殊纬线做成独立遮罩，最后盖在气压层之上
        self._line_a = np.zeros((BH, BW, 1), np.float32)
        for lat, strength in ((0.0, 0.30), (23.5, 0.16), (-23.5, 0.16),
                              (66.5, 0.11), (-66.5, 0.11)):
            y = int((90.0 - lat) / 180.0 * BH)
            y = min(max(y, 0), BH - 1)
            thick = 2 if lat == 0.0 else 1
            self._line_a[max(0, y - thick + 1):y + 1, :, 0] = strength

        self._base = np.clip(rgb, 0, 255).astype(np.uint8)
        self._base_f = self._base.astype(np.float32)
        if not hasattr(self, "_compose_buf"):
            self._compose_buf = np.empty((BH, BW, 3), np.float32)
            self._compose_u8 = np.empty((BH, BW, 3), np.uint8)

    # ---------- 气压层（照参考 shader 的配色） ----------
    def _build_pressure(self):
        """LOW = 蓝，HIGH = 橙；经 0.9-0.8*rgb 反相后低压暖橙、高压淡蓝。
        配色在 360×180 上算完再放大，避免在大图上做浮点运算。"""
        p = np.flipud(self.wind.pressure())
        # 低压暖橙、准静止面米白、高压蓝；1045+ 的极地高压用深蓝，
        # 这样冬季南极高压不会和普通副热带高压混成一片。
        stops = np.array([995.0, 1005.0, 1013.0, 1025.0, 1045.0, 1065.0], np.float32)
        cols = np.array([[210, 120, 60], [236, 190, 140], [236, 238, 240],
                         [150, 190, 225], [60, 100, 170], [30, 50, 100]], np.float32)
        small = np.empty(p.shape + (3,), np.float32)
        for c in range(3):
            small[:, :, c] = np.interp(p, stops, cols[:, c])
        small = np.clip(small, 0, 255).astype(np.uint8)
        BH, BW = self.BH, self.BW
        ys = (np.arange(BH) * small.shape[0] / BH).astype(np.int32)
        xs = (np.arange(BW) * small.shape[1] / BW).astype(np.int32)
        big = small[ys][:, xs]
        if getattr(self, "_prv_b", None) is None:
            self._prv_a = big.astype(np.float32)
            self._prv_b = self._prv_a
        else:
            self._prv_a = self._pressure_rgb.astype(np.float32) if self._pressure_rgb is not None else big.astype(np.float32)
            self._prv_b = big.astype(np.float32)
        self._prv_phase = 0.0
        self._pressure_rgb = big.astype(np.float32)

    def _blend_pressure(self):
        """气压层每帧向目标淡入，季节变化不会一跳一跳。"""
        if getattr(self, "_prv_b", None) is None or self._prv_phase >= 1.0:
            return
        self._prv_phase = min(1.0, self._prv_phase + 1.0 / 10.0)
        f = self._prv_phase
        cur = self._compose_p
        a = self._prv_a
        b = self._prv_b
        np.subtract(b, a, out=cur)
        cur *= f
        cur += a
        self._pressure_cur = cur
        self._pressure_rgb = cur

    def _apply_pressure(self):
        base = self._base_f
        if self._pressure_rgb is None:
            return base
        pr = self._pressure_rgb.astype(np.float32)
        # 气压层只做 62% 叠加，海岸线和纸面纹理仍然保留
        return base * 0.48 + pr * 0.52

    # ---------- 真实年际变化着色 ----------
    def _build_change_layer(self):
        """年际变化模式下，用蓝/橙表示风向相对十年平均的偏转。"""
        if not (self.real_mode and self.real is not None
                and self.real_layer == "anomaly"):
            self._change_rgb = None
            self._change_a = None
            return
        d = self.real.direction_delta(self.real_year_idx, self.real_month_idx)
        # 在角度空间做圆周平滑，避免 2.5° 格点直接放大后出现方块，
        # 也避免 ±180° 附近的角度跳变。
        rad = np.radians(d)
        d = np.degrees(np.arctan2(gaussian_filter(np.sin(rad), 0.8),
                                  gaussian_filter(np.cos(rad), 0.8))).astype(np.float32)
        big = zoom(d, (self.BH / d.shape[0], self.BW / d.shape[1]), order=1)[:self.BH, :self.BW]
        step = np.clip(big / 30.0, -1.0, 1.0)
        paper = np.array([232, 231, 225], np.float32)
        warm = np.array([214, 126, 62], np.float32)
        cool = np.array([54, 112, 182], np.float32)
        rgb = np.empty((self.BH, self.BW, 3), np.float32)
        rgb[:] = paper * (1.0 - np.abs(step)[..., None])
        rgb += warm * np.clip(step, 0.0, 1.0)[..., None]
        rgb += cool * np.clip(-step, 0.0, 1.0)[..., None]
        self._change_rgb = rgb
        self._change_a = (np.clip(np.abs(big) / 30.0, 0.0, 1.0) * 0.55)[..., None].astype(np.float32)

    # ---------- 粒子推进（照 getPosition 的规则） ----------
    def _step_particles(self, px, py, dens, spawn, steps):
        """每个像素最多一个粒子。

        关键：**每步 dt 很小**（远小于一格），一帧走多步，每步都往拖尾上打点。
        一步跨太大时中间格子留不下痕迹，粒子就成了一条条跳跃的点，方向看着就乱。
        """
        BH, BW = self.BH, self.BW
        dt = self._wgain / steps                      # 每步位移（像素）
        dens *= 0.82                                  # 每帧衰减一次即可

        for _ in range(steps):
            alive = px >= 0
            ys, xs = np.nonzero(alive)
            nx_buf = self._buf_x
            ny_buf = self._buf_y
            nx_buf.fill(-1.0)
            ny_buf.fill(-1.0)

            if len(ys):
                keep = self._rng.random(len(ys)) > 0.004 / steps
                ys, xs = ys[keep], xs[keep]
                Hf, Wf = self._wu.shape
                mx = px[ys, xs] / BW * Wf
                my = (1.0 - py[ys, xs] / BH) * Hf
                uu = _bilinear(self._wu, mx, my)
                vv = _bilinear(self._wv, mx, my)
                m = np.hypot(uu, vv)
                k = np.clip(m / self._wref, 0.0, 1.0) / np.maximum(m, 1e-9)
                dt_s = dt * self.speed_scale
                nx = np.mod(px[ys, xs] + uu * k * dt_s, BW)
                ny = np.clip(py[ys, xs] - vv * k * dt_s, 0, BH - 1)
                tx = np.clip(nx.astype(np.int32), 0, BW - 1)
                ty = np.clip(ny.astype(np.int32), 0, BH - 1)
                nx_buf[ty, tx] = nx
                ny_buf[ty, tx] = ny

            # 稀疏诞生
            empty = (nx_buf < 0).sum()
            k = self._rng.poisson(spawn / steps * empty)
            if k > 0:
                cy = self._rng.integers(0, BH, k)
                cx = self._rng.integers(0, BW, k)
                nx_buf[cy, cx] = cx + self._rng.random(k)
                ny_buf[cy, cx] = cy + self._rng.random(k)

            dens[nx_buf >= 0] = 1.0
            px, nx_buf = nx_buf, px                    # 交换缓冲，避免每次分配
            py, ny_buf = ny_buf, py
            self._buf_x, self._buf_y = nx_buf, ny_buf

        return px, py, dens

    # ---------- 合成 ----------
    def _blend_wind(self):
        """每帧把风场从上一版平滑推向新版，季节才是连续变化而不是跳变。"""
        if getattr(self, "_wu_t", None) is None:
            return
        self._wphase = min(1.0, self._wphase + 1.0 / 10.0)
        f = self._wphase
        np.multiply(self._wu_t - self._wu_a, f, out=self._wu)
        self._wu += self._wu_a
        np.multiply(self._wv_t - self._wv_a, f, out=self._wv)
        self._wv += self._wv_a

    def _compose(self):
        out = self._compose_buf
        if self.show_pressure and self._pressure_cur is not None:
            base = self._base_f
            pr = self._pressure_cur
            for c in range(3):
                out[:, :, c] = base[:, :, c] * 0.48 + pr[:, :, c] * 0.52
        else:
            out[:] = self._base_f
        if self._change_a is not None and self._change_rgb is not None:
            a = self._change_a[:, :, 0]
            inv = 1.0 - a
            for c in range(3):
                out[:, :, c] = out[:, :, c] * inv + self._change_rgb[:, :, c] * a
        # 逐通道就地混合，(H,W,1) 广播到 (H,W,3) 太慢
        if self.show_wind:
            a = self._dens_w
            inv = 1.0 - a
            for c in range(3):
                ch = out[:, :, c]
                ch *= inv
                ch += _STREAK_W[c] * a
        if getattr(self, "_line_a", None) is not None:
            a = self._line_a
            inv = 1.0 - a
            for c, v in enumerate(_LINE_RGB):
                ch = out[:, :, c]
                ch *= inv[:, :, 0]
                ch += v * a[:, :, 0]
        self._compose_u8[:] = np.clip(out, 0, 255)
        self.frame = ndarray_to_qimage(self._compose_u8)

    # ---------- 循环 ----------
    def tick(self):
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(1e-3, now - self._t_prev))
        self._t_prev = now

        # 季节按天连续推进；真实模式下改为可选地逐月播放
        if not self.real_mode:
            if self.auto_season:
                self._day += 1.2                    # 每帧 1.2 天，一年约 300 帧
                if self._day >= 365.0:
                    self._day -= 365.0
                self._season_t = 1.0 + self._day / 365.0 * 12.0
            if hasattr(self, "lbl_month"):
                m = int(self._day // 30.44) + 1
                d = int(self._day - (m - 1) * 30.44) + 1
                self.lbl_month.setText(f"{m} 月 {d} 日")
                if hasattr(self, "season_slider"):
                    self.season_slider.blockSignals(True)
                    self.season_slider.setValue(m)
                    self.season_slider.blockSignals(False)
                    self.season_lbl.setText(f"{m} 月")
        elif (self.real_autoplay and self.real is not None
              and self.real_layer != "trend"):
            self._year_tick += 1
            if self._year_tick >= 30:
                self._year_tick = 0
                month = self.real_month_idx + 1
                year = self.real_year_idx
                if month >= len(self.real.month_values):
                    month = 0
                    year = (year + 1) % len(self.real.year_values)
                self.real_year_idx = year
                self.year_slider.blockSignals(True)
                self.year_slider.setValue(year)
                self.year_slider.blockSignals(False)
                self.month_slider.setValue(month)

        self._season_tick += 1
        self._blend_wind()
        if not self.real_mode:
            if self.show_pressure:
                self._blend_pressure()
            if self._season_tick % 10 == 0:
                self.wind.update(self._season_t, self._contrast)
                self._build_velocity()
                if self.show_pressure:
                    self._build_pressure()
        if self.show_wind:
            # 每帧走多步：步数上去，粒子看起来才够快。
            # 真实风场更易汇聚成窄带，适当提高出生率，避免整张图只剩零星虚线。
            spawn = 2.2e-4 * (2.8 if self.real_mode else 1.0)
            self._px_w, self._py_w, self._dens_w = self._step_particles(
                self._px_w, self._py_w, self._dens_w, spawn, self.substeps)
        else:
            self._dens_w *= 0.9
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
        panel.setFixedWidth(286)
        pv = QVBoxLayout(panel)
        pv.setSpacing(7)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("全球环流")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("气压梯度 + 科里奥利力 → 风\n地形减速/绕流 · 近十年真实风场")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = MapCanvas(self)

        # ---- 合成模式的季节推演 ----
        g1 = QGroupBox("合成模式 · 季节")
        f1 = QFormLayout(g1)
        f1.setSpacing(6)
        f1.setContentsMargins(8, 8, 8, 8)
        self.lbl_month = QLabel("1 月 1 日")
        self.lbl_month.setStyleSheet("color: #45454a; font-size: 12px;")
        f1.addRow("日期", self.lbl_month)
        self.season_slider = LabeledSlider("", 1, 12, 1, formatter=lambda v: f"{v} 月", value_width=40)
        self.season_lbl = self.season_slider.value_label   # 兼容其它路径直接改文字
        self.season_slider.valueChanged.connect(self._on_season)
        f1.addRow("月份", self.season_slider)
        self.chk_auto = QCheckBox("自动推演")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._on_auto)
        f1.addRow(self.chk_auto)
        pv.addWidget(g1)
        self.time_group = g1

        # ---- 真实月均风场 ----
        g_real = QGroupBox("真实模式 · 2015–2024 月均")
        fr = QVBoxLayout(g_real)
        fr.setSpacing(4)
        fr.setContentsMargins(8, 8, 8, 8)
        self.chk_real = QCheckBox("启用真实月均风场")
        self.chk_real.toggled.connect(self._on_real_mode)
        if self.real is None:
            self.chk_real.setEnabled(False)
            self.chk_real.setToolTip("缺少 wind_real.npz，真实模式不可用")
        fr.addWidget(self.chk_real)

        row_layer = QHBoxLayout()
        row_layer.setSpacing(6)
        row_layer.addWidget(QLabel("图层"))
        self.real_layer_combo = QComboBox()
        self.real_layer_combo.addItems(["月均风场", "月际变化 Δ", "十年趋势"])
        self.real_layer_combo.setEnabled(False)
        self.real_layer_combo.currentIndexChanged.connect(self._on_real_layer)
        row_layer.addWidget(self.real_layer_combo, 1)
        fr.addLayout(row_layer)

        self.year_slider = LabeledSlider(
            "年份", 0, max(0, len(self.real.year_values) - 1) if self.real else 0,
            self.real_year_idx, value_width=64,
            formatter=lambda i: f"{int(self.real.year_values[i])} 年" if self.real else "—")
        self.year_lbl = self.year_slider.value_label
        self.year_slider.setEnabled(False)
        self.year_slider.valueChanged.connect(self._on_real_year)
        fr.addWidget(self.year_slider)

        self.month_slider = LabeledSlider(
            "月份", 0, max(0, len(self.real.month_values) - 1) if self.real else 0,
            self.real_month_idx, value_width=52,
            formatter=lambda i: f"{int(self.real.month_values[i])} 月" if self.real else "—")
        self.month_lbl = self.month_slider.value_label
        self.month_slider.setEnabled(False)
        self.month_slider.valueChanged.connect(self._on_real_month)
        fr.addWidget(self.month_slider)

        self.chk_real_auto = QCheckBox("自动播放月份")
        self.chk_real_auto.setEnabled(False)
        self.chk_real_auto.toggled.connect(self._on_real_auto)
        fr.addWidget(self.chk_real_auto)
        src = QLabel("NCEP/NCAR Reanalysis 1 · 月平均 10 m 风")
        src.setStyleSheet("color: #8a8a90;")
        sf2 = src.font()
        sf2.setPointSize(7)
        src.setFont(sf2)
        fr.addWidget(src)
        pv.addWidget(g_real)
        self.real_group = g_real

        # ---- 地形 ----
        g_terrain = QGroupBox("地形 / 海拔")
        ft = QFormLayout(g_terrain)
        ft.setSpacing(6)
        ft.setContentsMargins(8, 8, 8, 8)
        self.chk_terrain = QCheckBox("显示地形阴影与海拔配色")
        self.chk_terrain.setChecked(True)
        self.chk_terrain.toggled.connect(lambda c: self._toggle("show_terrain", c))
        ft.addRow(self.chk_terrain)
        self.terrain_slider = LabeledSlider("", 0, 200, 100, formatter=lambda v: f"{v}%", value_width=46)
        self.terrain_lbl = self.terrain_slider.value_label
        self.terrain_slider.valueChanged.connect(self._on_terrain)
        ft.addRow("影响（合成）", self.terrain_slider)
        note = QLabel("高地减速、绕流；真实风场已含地形")
        note.setStyleSheet("color: #8a8a90;")
        nf = note.font()
        nf.setPointSize(7)
        note.setFont(nf)
        note.setWordWrap(True)
        ft.addRow(note)
        pv.addWidget(g_terrain)
        self.terrain_group = g_terrain

        # ---- 海陆差距 ----
        g_land = QGroupBox("海陆差距")
        f_land = QFormLayout(g_land)
        f_land.setSpacing(6)
        f_land.setContentsMargins(8, 8, 8, 8)
        self.land_slider = LabeledSlider("", 0, 300, 150, formatter=lambda v: f"{v}%", value_width=46)
        self.land_lbl = self.land_slider.value_label
        self.land_slider.valueChanged.connect(self._on_land)
        f_land.addRow("强度", self.land_slider)
        pv.addWidget(g_land)
        self.land_group = g_land

        # ---- 风速 ----
        g_speed = QGroupBox("风速")
        f_speed = QFormLayout(g_speed)
        f_speed.setSpacing(6)
        f_speed.setContentsMargins(8, 8, 8, 8)
        self.speed_slider = LabeledSlider("", 0, 1000, 500, formatter=lambda v: f"{v}%", value_width=46)
        self.speed_lbl = self.speed_slider.value_label
        self.speed_slider.valueChanged.connect(self._on_speed)
        f_speed.addRow("倍率", self.speed_slider)
        pv.addWidget(g_speed)

        # ---- 显示 ----
        g2 = QGroupBox("显示")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(4)
        v2.setContentsMargins(8, 8, 8, 8)
        self.chk_wind = QCheckBox("风场流线")
        self.chk_pressure = QCheckBox("气压分布")
        self.chk_grid = QCheckBox("经纬网")
        for cb, attr, checked in ((self.chk_wind, "show_wind", True),
                                  (self.chk_pressure, "show_pressure", True),
                                  (self.chk_grid, "show_grid", True)):
            cb.setChecked(checked)
            cb.toggled.connect(lambda c, a=attr: self._toggle(a, c))
            v2.addWidget(cb)
        pv.addWidget(g2)
        self.display_group = g2

        btn = QPushButton("重播粒子")
        btn.clicked.connect(self._reseed)
        pv.addWidget(btn)
        pv.addStretch()

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(304)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_scroll.setFrameShape(QFrame.NoFrame)
        # 不给 QScrollArea 设样式表，滚动条才会保持系统默认外观
        panel_scroll.viewport().setAutoFillBackground(False)
        panel_scroll.setWidget(panel)
        root.addWidget(panel_scroll)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(8)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_fps = QLabel("–")
        self.lbl_fps.setMinimumWidth(70)
        self.lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addPermanentWidget(self.lbl_fps)
        self._update_status()

    # ---------- 真实模式 ----------
    def _set_year_label(self):
        if self.real is None:
            return
        if self.real_layer == "trend":
            self.year_lbl.setText("2015–2024")
            self.month_lbl.setText("全年")
        else:
            self.year_lbl.setText(f"{int(self.real.year_values[self.real_year_idx])} 年")
            self.month_lbl.setText(f"{int(self.real.month_values[self.real_month_idx])} 月")

    def _on_real_mode(self, on):
        if on and self.real is None:
            self.chk_real.setChecked(False)
            QMessageBox.warning(self, "真实数据", "缺少 apps/Physics/wind_real.npz，无法启用真实模式。")
            return
        self.real_mode = bool(on)
        self.time_group.setEnabled(not self.real_mode)
        self.land_group.setEnabled(not self.real_mode)
        self.chk_auto.setEnabled(not self.real_mode)
        self.land_slider.setEnabled(not self.real_mode)
        self.terrain_slider.setEnabled(not self.real_mode)
        self.real_layer_combo.setEnabled(self.real_mode)
        self.year_slider.setEnabled(self.real_mode and self.real_layer != "trend")
        self.month_slider.setEnabled(self.real_mode and self.real_layer != "trend")
        self.chk_real_auto.setEnabled(self.real_mode and self.real_layer != "trend")
        self.chk_pressure.setEnabled(not self.real_mode)
        self._year_tick = 0

        if self.real_mode:
            self._pressure_before_real = self.show_pressure
            self.show_pressure = False
            self.chk_pressure.setChecked(False)
            self.lbl_month.setText("真实数据")
            self._pressure_cur = None
            self._pressure_rgb = None
            self._prv_a = None
            self._prv_b = None
        else:
            self.show_pressure = bool(getattr(self, "_pressure_before_real", True))
            self.chk_pressure.setChecked(self.show_pressure)
            m = int(self._day // 30.44) + 1
            d = int(self._day - (m - 1) * 30.44) + 1
            self.lbl_month.setText(f"{m} 月 {d} 日")
            self.wind.update(self._season_t, self._contrast)
            if self.show_pressure:
                self._build_pressure()

        self._build_velocity()
        self._build_change_layer()
        self._compose()
        self.canvas.update()
        self._update_status()

    def _on_real_layer(self, index):
        layers = ("monthly", "anomaly", "trend")
        self.real_layer = layers[int(index)] if 0 <= int(index) < len(layers) else "monthly"
        is_trend = self.real_layer == "trend"
        self.year_slider.setEnabled(self.real_mode and not is_trend)
        self.month_slider.setEnabled(self.real_mode and not is_trend)
        self.chk_real_auto.setEnabled(self.real_mode and not is_trend)
        self._set_year_label()
        if self.real_mode:
            self._build_velocity()
            self._build_change_layer()
            self._compose()
            self.canvas.update()
            self._update_status()

    def _on_real_year(self, value):
        if self.real is None:
            return
        self.real_year_idx = int(value)
        self._set_year_label()
        if self.real_mode:
            self._build_velocity()
            self._build_change_layer()
            self._compose()
            self.canvas.update()
            self._update_status()

    def _on_real_month(self, value):
        if self.real is None:
            return
        self.real_month_idx = int(value)
        self._set_year_label()
        if self.real_mode:
            self._build_velocity()
            self._build_change_layer()
            self._compose()
            self.canvas.update()
            self._update_status()

    def _on_real_auto(self, on):
        self.real_autoplay = bool(on)
        self._year_tick = 0

    # ---------- 地形 ----------
    def _on_terrain(self, value):
        self.terrain_strength = value / 100.0
        self.terrain_lbl.setText(f"{value}%")
        self._terrain_timer.start(140)

    def _apply_terrain(self):
        if self.real_mode:
            return
        self._build_velocity()
        self._compose()
        self.canvas.update()

    def _update_status(self):
        if self.real_mode and self.real is not None:
            year = int(self.real.year_values[self.real_year_idx])
            month = int(self.real.month_values[self.real_month_idx])
            if self.real_layer == "anomaly":
                text = (f"真实月际变化：{year} 年 {month} 月 − 同月 2015–2024 平均；"
                        "蓝=顺时针，橙=逆时针（±30°）")
            elif self.real_layer == "trend":
                text = "真实十年趋势：2015–2024 年均值的线性拟合，流线为每十年变化（m/s）"
            else:
                text = (f"真实月均风场：{year} 年 {month} 月 · "
                        "NCEP/NCAR Reanalysis 1（10 m）")
        elif self.show_pressure:
            text = "气压：橙=低压，蓝=高压；流线为合成风场。"
        else:
            text = "细线是气流：每像素一个粒子，拖尾 0.9 衰减；勾选「气压分布」看高压低压区"
        self.lbl_status.setText(text)

    def _on_season(self, month):
        if self.real_mode:
            return
        self.auto_season = False
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(False)
        self.chk_auto.blockSignals(False)
        self._season_t = float(month)
        self._day = (month - 1) * 365.0 / 12.0 + 1.0
        self.season_lbl.setText(f"{month} 月")
        self.lbl_month.setText(f"{month} 月")
        self.wind.update(self._season_t, self._contrast)
        self._build_velocity()
        if self.show_pressure:
            self._build_pressure()
        self._compose()
        self.canvas.update()

    def _on_land(self, v):
        self.land_lbl.setText(f"{v}%")
        self._contrast = v / 100.0
        self._land_timer.start(160)          # 拖动时先攒着，松手再重算

    def _apply_contrast(self):
        if self.real_mode:
            return
        self.wind.update(self._season_t, self._contrast)
        self._build_velocity()
        if self.show_pressure:
            self._build_pressure()
        self.canvas.update()

    def _on_speed(self, v):
        self.speed_scale = v / 100.0
        self.speed_lbl.setText(f"{v}%")

    def _on_auto(self, on):
        self.auto_season = bool(on)

    def _toggle(self, attr, checked):
        setattr(self, attr, bool(checked))
        if attr == "show_pressure" and checked and not self.real_mode:
            self._build_pressure()
        self._build_base()
        self._compose()
        self.canvas.update()

    def _reseed(self):
        for a in ("_px_w", "_py_w"):
            getattr(self, a)[:] = -1
        self._dens_w[:] = 0
        self.canvas.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    raise SystemExit(run_app(CirculationApp))
