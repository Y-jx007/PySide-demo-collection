"""
Caustic — 光线追踪焦散演示。

两个场景：
1. 折射焦散（默认）：平行光穿过带波纹的水面，按 Snell 定律折射后落到池底，
   逐条光线把能量累加成地板光照图，形成真实的焦散网纹；
2. 反射肾形线：平行光照射圆镜，反射线的包络是肾形线（几何演示）。
"""

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from common.app import run_app
from common.imaging import ndarray_to_qimage
from common.lab_window import LabWindow
from common.qt_base import LabeledSlider


# ==================== 配色 ====================
BG       = QColor(250, 250, 247)
PANEL    = QColor(245, 244, 239)
BORDER   = QColor(222, 221, 214)
INK_DARK = QColor(45, 50, 65)
INK_MID  = QColor(120, 125, 140)
INK_LT   = QColor(205, 206, 212)
INK_XLT  = QColor(232, 232, 234)
FOCUS    = QColor(125, 85, 165)
ARROW_A  = QColor(65, 105, 165)
ARROW_B  = QColor(175, 100, 95)
HALO     = QColor(255, 255, 255, 210)

F_TITLE = QFont("Sans Serif", 13, QFont.Bold)
F_SUB   = QFont("Sans Serif", 9)
F_LABEL = QFont("Sans Serif", 9, QFont.Bold)
F_LEG   = QFont("Sans Serif", 9)

# 密度 colormap：白 → 深蓝黑（浅色系）
CMAP = np.array([
    [1.00, 1.00, 1.00],
    [0.94, 0.95, 0.97],
    [0.80, 0.83, 0.90],
    [0.55, 0.60, 0.73],
    [0.28, 0.34, 0.50],
    [0.10, 0.14, 0.24],
    [0.04, 0.06, 0.12],
], dtype=np.float32)
CMAP_POS = np.linspace(0, 1, len(CMAP))


def colorize(v):
    out = np.empty((*v.shape, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, CMAP_POS, CMAP[:, c])
    return out


def glow(d, factor=8, iters=2):
    h, w = d.shape
    sh, sw = h // factor, w // factor
    if sh < 2 or sw < 2:
        return d.copy()
    dh = d[:sh * factor, :sw * factor]
    ds = dh.reshape(sh, factor, sw, factor).mean(axis=(1, 3))
    for _ in range(iters):
        ds = 0.25 * (np.roll(ds, 1, 0) + np.roll(ds, -1, 0) +
                     np.roll(ds, 1, 1) + np.roll(ds, -1, 1))
    up = np.repeat(np.repeat(ds, factor, axis=0), factor, axis=1)
    if up.shape[0] < h:
        up = np.pad(up, ((0, h - up.shape[0]), (0, 0)), mode="edge")
    else:
        up = up[:h]
    if up.shape[1] < w:
        up = np.pad(up, ((0, 0), (0, w - up.shape[1])), mode="edge")
    else:
        up = up[:, :w]
    return up


# =====================================================================
class MainView(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._density = None
        self._density_key = None
        self._caustic = None
        self._caustic_key = None
        self.setFixedSize(640, 640)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: #ffffff;")

    def _bounds(self):
        return 1.9, -1.5, 2.7

    def _transform(self):
        W, H = self.width(), self.height()
        wx, wy_min, wy_max = self._bounds()
        scale = min(W / (2 * wx), H / (wy_max - wy_min)) * 0.94
        return W / 2, H / 2, scale

    def world_to_screen(self, x, y):
        cx, cy, scale = self._transform()
        _, wy_min, wy_max = self._bounds()
        wy_c = (wy_min + wy_max) / 2
        return QPointF(cx + x * scale, cy - (y - wy_c) * scale)

    def invalidate(self):
        self._density_key = None
        self._caustic_key = None

    # ============================================================
    # 折射焦散：平行光 → 波纹水面 → 池底光照图
    # ============================================================
    def _caustic_key_now(self):
        s = self.state
        return (round(s["amp"], 4), round(s["wavelength"], 3),
                round(s["index_n"], 3), round(s["depth"], 3),
                round(s["phase"], 3), bool(s.get("playing")))

    def _ensure_caustic(self):
        key = self._caustic_key_now()
        if self._caustic_key == key and self._caustic is not None:
            return
        self._caustic = self._compute_caustic()
        self._caustic_key = key

    def _compute_caustic(self):
        """正向光子追踪：均匀射线穿过水面，按 Snell 折射后落在池底。"""
        s = self.state
        playing = bool(self.state.get("playing"))
        n = 380 if playing else 560
        res = 620 if playing else 900
        axis = np.linspace(-1.0, 1.0, n)
        X, Z = np.meshgrid(axis, axis, indexing="ij")
        step = axis[1] - axis[0]

        k = 2.0 * math.pi / max(s["wavelength"], 1e-3)
        phase = s["phase"]
        height = s["amp"] * (
            np.sin(k * X + phase) * np.cos(1.15 * k * Z - 0.4 * phase)
            + 0.55 * np.sin(1.7 * k * X - 0.8 * k * Z + 0.7 * phase)
            + 0.32 * np.sin(0.62 * k * X + 1.35 * k * Z + 1.1 * phase)
            + 0.2 * np.sin(2.25 * k * X + 0.45 * k * Z - 0.5 * phase)
        )
        hx, hz = np.gradient(height, step)

        inv = 1.0 / np.sqrt(hx * hx + 1.0 + hz * hz)
        nx, ny, nz = -hx * inv, inv, -hz * inv

        # 入射方向 (0,-1,0)，n1=1、n2=折射率
        cos_i = np.clip(ny, 0.0, 1.0)
        eta = 1.0 / max(s["index_n"], 1.0001)
        sin2_t = eta * eta * (1.0 - cos_i * cos_i)
        cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, 1.0))
        scale_d = eta * cos_i - cos_t

        dx = scale_d * nx
        dy = -eta + scale_d * ny
        dz = scale_d * nz

        travel = s["depth"] / np.maximum(-dy, 1e-6)
        hit_x = X + dx * travel
        hit_z = Z + dz * travel

        # Fresnel 透射率（自然光 = s/p 平均）
        n1, n2 = 1.0, max(s["index_n"], 1.0001)
        rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
        rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
        energy = (1.0 - 0.5 * (rs + rp)) * np.clip(-dy, 0.0, 1.0)

        span = 1.15
        u = (hit_x + span) / (2 * span) * (res - 1)
        v = (hit_z + span) / (2 * span) * (res - 1)
        iu = np.floor(u).astype(np.int64)
        iv = np.floor(v).astype(np.int64)
        fu = u - iu
        fv = v - iv
        m = (iu >= 0) & (iu < res - 1) & (iv >= 0) & (iv < res - 1)
        iu, iv, fu, fv = iu[m], iv[m], fu[m], fv[m]
        w = energy[m]

        flat = np.zeros(res * res, dtype=np.float64)
        base = iv * res + iu
        for off, weight in ((0, (1 - fu) * (1 - fv)),
                            (1, fu * (1 - fv)),
                            (res, (1 - fu) * fv),
                            (res + 1, fu * fv)):
            flat += np.bincount(base + off, weights=weight * w, minlength=res * res)

        d = flat.reshape(res, res).astype(np.float32)
        # 按高分位数归一化，避免个别焦点把整张图压暗
        positive = d[d > 0]
        if positive.size:
            ref = np.percentile(positive, 99.5)
            if ref > 0:
                d = d / ref
        d = np.clip(d, 0.0, 1.0)

        # 轻微模糊 + bloom，让焦散线柔和
        pad = np.pad(d, 1, mode="edge")
        d = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
             + pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:] + d) / 9.0
        render = d + 0.35 * glow(d, factor=8, iters=2)
        render /= max(float(render.max()), 1e-12)
        render = np.power(render, 1.15)

        rgb = colorize(render)
        arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        return ndarray_to_qimage(arr)

    def _paint_refract(self, p):
        self._ensure_caustic()
        W, H = self.width(), self.height()
        margin = 22
        if self._caustic is not None and not self._caustic.isNull():
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF(margin, margin, W - 2 * margin, H - 2 * margin),
                        self._caustic)
        p.setPen(INK_DARK)
        p.setFont(F_TITLE)
        p.drawText(24, 34, "折射焦散")
        p.setPen(INK_MID)
        p.setFont(F_SUB)
        p.drawText(24, 52, "平行光 · 波纹水面 · Snell 折射 · 池底光照图")

        p.setFont(F_LEG)
        p.setPen(INK_MID)
        s = self.state
        p.drawText(24, H - 22,
                   f"波幅 {s['amp']:.3f} · 波长 {s['wavelength']:.2f} · "
                   f"n = {s['index_n']:.2f} · 水深 {s['depth']:.2f}")

    # ============================================================
    # 反射肾形线（原有几何演示）
    # ============================================================
    def _ensure_density(self):
        key = (round(self.state['R'], 4), self.state['n_rays'])
        if self._density_key == key and self._density is not None:
            return
        self._density = self._compute_density()
        self._density_key = key

    def _compute_density(self):
        R = self.state['R']; n = self.state['n_rays']
        wx, wy_min, wy_max = self._bounds()
        iw = 800
        ih = int(iw * (wy_max - wy_min) / (2 * wx))

        th = np.linspace(np.pi + 1e-3, 2 * np.pi - 1e-3, n)
        Px, Py = R * np.cos(th), R * np.sin(th)
        dx = np.sin(2 * th); dy = -np.cos(2 * th)
        t_len = 3.4
        Ex = Px + t_len * dx
        Ey = Py + t_len * dy

        smp = 80
        tt = np.linspace(0, 1, smp)
        xs = (Px[:, None] + (Ex - Px)[:, None] * tt).ravel()
        ys = (Py[:, None] + (Ey - Py)[:, None] * tt).ravel()

        px = (xs + wx) / (2 * wx) * iw - 0.5
        py = (wy_max - ys) / (wy_max - wy_min) * ih - 0.5
        ix = np.floor(px).astype(np.int64); iy = np.floor(py).astype(np.int64)
        fx = px - ix; fy = py - iy
        m = (ix >= 0) & (ix < iw - 1) & (iy >= 0) & (iy < ih - 1)
        ix, iy, fx, fy = ix[m], iy[m], fx[m], fy[m]

        flat = np.zeros(iw * ih, dtype=np.float64)
        base = iy * iw + ix
        for off, w in ((0, (1 - fx) * (1 - fy)),
                       (1, fx * (1 - fy)),
                       (iw, (1 - fx) * fy),
                       (iw + 1, fx * fy)):
            flat += np.bincount(base + off, weights=w, minlength=iw * ih)

        d = flat.reshape(ih, iw).astype(np.float32)
        if d.max() > 0:
            d /= d.max()

        d_pad = np.pad(d, 1, mode='edge')
        d = (d_pad[:-2, 1:-1] + d_pad[2:, 1:-1] +
             d_pad[1:-1, :-2] + d_pad[1:-1, 2:] +
             d_pad[:-2, :-2] + d_pad[:-2, 2:] +
             d_pad[2:, :-2] + d_pad[2:, 2:] + d) / 9.0

        render = d + 0.3 * glow(d, factor=8, iters=2)
        render /= max(render.max(), 1e-12)
        render = np.power(render, 0.6)
        rgb = colorize(render)
        arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        return ndarray_to_qimage(arr)

    # ============================================================
    # 绘制辅助
    # ============================================================
    def _arrow(self, p, p0, p1, color, w=2.0):
        pen = QPen(color, w); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.drawLine(p0, p1)
        ddx, ddy = p1.x() - p0.x(), p1.y() - p0.y()
        L = math.hypot(ddx, ddy)
        if L < 1e-6:
            return
        ux, uy = ddx / L, ddy / L
        nx, ny = -uy, ux
        al, aw = 9, 4.5
        bx, by = p1.x() - ux * al, p1.y() - uy * al
        tri = QPainterPath()
        tri.moveTo(p1)
        tri.lineTo(bx + nx * aw, by + ny * aw)
        tri.lineTo(bx - nx * aw, by - ny * aw)
        tri.closeSubpath()
        p.setBrush(color); p.setPen(Qt.NoPen)
        p.drawPath(tri)

    def _draw_point(self, p, pos, color, label=None, dx=10, dy=-6):
        p.setPen(QPen(BG, 1.6))
        p.setBrush(color)
        p.drawEllipse(pos, 6, 6)
        if label:
            p.setPen(color)
            p.setFont(F_LABEL)
            p.drawText(pos + QPointF(dx, dy), label)

    def _halo_line(self, p, p0, p1, color, width=1.8, halo_w=4):
        p.setPen(QPen(HALO, halo_w))
        p.drawLine(p0, p1)
        p.setPen(QPen(color, width))
        p.drawLine(p0, p1)

    # ============================================================
    # 主绘制
    # ============================================================
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)

        if self.state["mode"] == "refract":
            self._paint_refract(p)
            p.end()
            return

        self._paint_nephroid(p)
        p.end()

    def _paint_nephroid(self, p):
        W, H = self.width(), self.height()
        R = self.state['R']; theta = self.state['theta']
        wx, wy_min, wy_max = self._bounds()

        self._ensure_density()
        if self._density is not None and not self._density.isNull():
            tl = self.world_to_screen(-wx, wy_max)
            br = self.world_to_screen(wx, wy_min)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF(tl, br), self._density)

        def S(x, y):
            return self.world_to_screen(x, y)

        O_screen = S(0, 0)
        _, _, scale = self._transform()
        Rp = R * scale

        Px, Py = R * math.cos(theta), R * math.sin(theta)
        dx, dy = math.sin(2 * theta), -math.cos(2 * theta)
        Mx, My = Px / 2, Py / 2
        t_H = -R * math.sin(theta)
        Hx, Hy = Px + t_H * dx, Py + t_H * dy
        Qx, Qy = (Px + Hx) / 2, (Py + Hy) / 2
        a_ = R / 4.0; rho = R / 2.0
        Cx = (rho + a_) * math.cos(theta)
        Cy = (rho + a_) * math.sin(theta)
        y_top = wy_max - 0.10

        p.setPen(QPen(QColor(150, 120, 140, 55), 0.5))
        for ang in np.linspace(np.pi + 0.01, 2 * np.pi - 0.01, 220):
            px_ = R * math.cos(ang); py_ = R * math.sin(ang)
            p.drawLine(S(px_, py_),
                       S(px_ + 2.1 * math.sin(2 * ang),
                         py_ - 2.1 * math.cos(2 * ang)))

        p.setPen(QPen(INK_LT, 1.0, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(O_screen, Rp, Rp)
        path = QPainterPath()
        for i, ang in enumerate(np.linspace(np.pi, 2 * np.pi, 140)):
            pt = S(R * math.cos(ang), R * math.sin(ang))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(INK_DARK, 2.0))
        p.drawPath(path)

        p.setPen(QPen(INK_XLT, 0.6))
        for xr in np.linspace(-R * 0.97, R * 0.97, 30):
            yb = -math.sqrt(max(R * R - xr * xr, 1e-4))
            p.drawLine(S(xr, y_top), S(xr, yb))

        p.setPen(QPen(INK_LT, 1.0, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(O_screen, rho * scale, rho * scale)

        path = QPainterPath()
        for i, t_ in enumerate(np.linspace(0, 2 * np.pi, 420)):
            qx = a_ * (3 * math.cos(t_) + math.cos(3 * t_))
            qy = a_ * (3 * math.sin(t_) + math.sin(3 * t_))
            path.moveTo(S(qx, qy)) if i == 0 else path.lineTo(S(qx, qy))
        p.setPen(QPen(QColor(200, 195, 215), 1.0, Qt.DashLine))
        p.drawPath(path)

        path = QPainterPath(); first = True
        for t_ in np.linspace(np.pi, theta, 160):
            qx = a_ * (3 * math.cos(t_) + math.cos(3 * t_))
            qy = a_ * (3 * math.sin(t_) + math.sin(3 * t_))
            if first:
                path.moveTo(S(qx, qy)); first = False
            else:
                path.lineTo(S(qx, qy))
        p.setPen(QPen(FOCUS, 2.2))
        p.drawPath(path)

        p.setPen(QPen(INK_LT, 1.0, Qt.DashLine))
        p.drawLine(O_screen, S(Px, Py))
        p.drawLine(O_screen, S(Hx, Hy))

        C_screen = S(Cx, Cy)
        p.setPen(QPen(QColor(165, 180, 205), 1.2))
        p.setBrush(QColor(205, 215, 235, 55))
        p.drawEllipse(C_screen, a_ * scale, a_ * scale)

        tcx = rho * math.cos(theta); tcy = rho * math.sin(theta)
        p.setPen(QPen(BG, 1.2)); p.setBrush(INK_MID)
        p.drawEllipse(S(tcx, tcy), 3, 3)

        self._halo_line(p, S(Px, y_top), S(Px, Py), INK_DARK, 1.8, 4)

        tl_x, tl_y = -math.sin(theta), math.cos(theta)
        p.setPen(QPen(INK_LT, 1.1, Qt.DashLine))
        p.drawLine(S(Px - 0.55 * tl_x, Py - 0.55 * tl_y),
                   S(Px + 0.55 * tl_x, Py + 0.55 * tl_y))

        self._halo_line(p, S(Px, Py),
                        S(Px + 2.3 * dx, Py + 2.3 * dy), INK_DARK, 1.8, 4)

        p.setPen(QPen(INK_DARK, 2.0))
        p.drawLine(S(Px, Py), S(Hx, Hy))

        p.setPen(QPen(FOCUS, 1.4, Qt.DashLine))
        p.drawLine(S(Mx, My), S(Qx, Qy))

        self._arrow(p, O_screen, C_screen, ARROW_A, 2.0)
        self._arrow(p, C_screen, S(Qx, Qy), ARROW_B, 2.0)

        self._draw_point(p, O_screen, INK_DARK, "O", -20, 4)
        self._draw_point(p, S(Px, Py), INK_DARK, "P", 10, -6)
        self._draw_point(p, S(Hx, Hy), INK_DARK, "H", 10, -6)
        self._draw_point(p, S(Mx, My), INK_DARK, "M", 10, -6)
        self._draw_point(p, S(Qx, Qy), FOCUS, "Q", 10, 16)

        p.setPen(INK_DARK); p.setFont(F_TITLE)
        p.drawText(24, 32, "反射焦散 = 肾形线")
        p.setPen(INK_MID); p.setFont(F_SUB)
        p.drawText(24, 50, "光线追踪渲染 · 周转圆构造")

        self._draw_legend(p, W, H)

    def _draw_legend(self, p, W, H):
        entries = [
            (INK_DARK, "O   圆心"),
            (INK_DARK, "P   入射点"),
            (INK_DARK, "H   垂足"),
            (INK_DARK, "M   OP 中点"),
            (FOCUS,    "Q   焦散点"),
            (ARROW_A,  "A   公转 θ"),
            (ARROW_B,  "B   自转 3θ"),
        ]
        row_h = 18
        box_w, box_h = 152, len(entries) * row_h + 20
        x0 = W - box_w - 24
        y0 = H - box_h - 24

        rect = QRectF(x0, y0, box_w, box_h)
        p.setPen(QPen(BORDER, 1.0))
        p.setBrush(QColor(252, 252, 250, 235))
        p.drawRoundedRect(rect, 6, 6)

        p.setFont(F_LEG)
        for i, (color, text) in enumerate(entries):
            y = y0 + 14 + i * row_h
            p.setPen(QPen(BG, 1.4))
            p.setBrush(color)
            p.drawEllipse(QPointF(x0 + 14, y), 5, 5)
            p.setPen(INK_MID)
            p.drawText(QPointF(x0 + 26, y + 4), text)


# =====================================================================
class MainWindow(LabWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caustic — 光线追踪焦散")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 720)

        self.state = {
            "mode": "refract",
            "amp": 0.035, "wavelength": 0.55, "index_n": 1.33,
            "depth": 0.6, "phase": 0.0, "playing": False,
            "R": 1.0, "theta": 4 * math.pi / 3, "n_rays": 15000,
        }
        self.view = MainView(self.state)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(70)
        self.playing = False

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(330)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)

        scene = QGroupBox("场景")
        sv = QVBoxLayout(scene)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["折射焦散", "反射肾形线"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        sv.addWidget(self.mode_combo)
        side.addWidget(scene)

        self.refract_group = QGroupBox("折射焦散")
        rv = QVBoxLayout(self.refract_group)
        rv.setSpacing(6)
        self.amp = LabeledSlider("波幅", 0, 100, 35,
                                 formatter=lambda v: f"{v / 1000:.3f}",
                                 label_width=42)
        self.wavelength = LabeledSlider("波长", 20, 100, 55,
                                        formatter=lambda v: f"{v / 100:.2f}",
                                        label_width=42)
        self.index = LabeledSlider("折射率", 105, 160, 133,
                                   formatter=lambda v: f"{v / 100:.2f}",
                                   label_width=42)
        self.depth = LabeledSlider("水深", 20, 100, 60,
                                   formatter=lambda v: f"{v / 100:.2f}",
                                   label_width=42)
        for slider, handler in ((self.amp, self._on_amp),
                                (self.wavelength, self._on_wavelength),
                                (self.index, self._on_index),
                                (self.depth, self._on_depth)):
            slider.bind(handler)
            rv.addWidget(slider)
        side.addWidget(self.refract_group)

        self.reflect_group = QGroupBox("反射肾形线")
        nv = QVBoxLayout(self.reflect_group)
        nv.setSpacing(6)
        self.theta = LabeledSlider("θ", 1812, 3588, 2400,
                                   formatter=lambda v: f"{v / 10:.1f}°",
                                   label_width=42)
        self.radius = LabeledSlider("R", 60, 150, 100,
                                    formatter=lambda v: f"{v / 100:.2f}",
                                    label_width=42)
        self.theta.bind(self._on_theta)
        self.radius.bind(self._on_radius)
        nv.addWidget(self.theta)
        nv.addWidget(self.radius)
        side.addWidget(self.reflect_group)
        self.reflect_group.setVisible(False)

        controls = QGroupBox("控制")
        ch = QHBoxLayout(controls)
        self.play_btn = QPushButton("播放")
        self.reset_btn = QPushButton("重置")
        self.play_btn.clicked.connect(self._toggle_play)
        self.reset_btn.clicked.connect(self._reset)
        ch.addWidget(self.play_btn)
        ch.addWidget(self.reset_btn)
        side.addWidget(controls)

        info = QGroupBox("说明")
        iv = QVBoxLayout(info)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        iv.addWidget(self.info_label)
        side.addWidget(info)
        side.addStretch(1)

        layout.addWidget(panel)
        layout.addWidget(self.view)
        layout.addStretch(1)
        self.setCentralWidget(root)

        self._update_info()
        self._update_status()

    # ------------------------------------------------------------------
    def _update_info(self):
        if self.state["mode"] == "refract":
            self.info_label.setText(
                "平行光垂直穿过带波纹的水面，按 Snell 定律折射后落到池底；"
                "逐条光线把能量累加成地板光照图，汇聚处即焦散。\n"
                "波幅/波长改变水面起伏，折射率与水深改变聚焦位置。")
        else:
            self.info_label.setText(
                "平行光照射球形反射面，反射线的包络形成肾形焦散线。\n"
                "θ 改变光源方向，R 改变反射球半径；播放会连续扫角。")

    def _update_status(self):
        s = self.state
        mode = "折射焦散" if s["mode"] == "refract" else "反射肾形线"
        state = "播放" if self.playing else "暂停"
        if s["mode"] == "refract":
            detail = (f"波幅 {s['amp']:.3f} · 波长 {s['wavelength']:.2f} · "
                      f"n = {s['index_n']:.2f} · 水深 {s['depth']:.2f}")
        else:
            detail = (f"R={s['R']:.2f} · θ={self.theta.slider.value() / 10:.1f}°"
                      f" · {s['n_rays']:,} 条射线")
        self.statusBar().showMessage(f"{mode} · {detail} · {state}")

    # ------------------------------------------------------------------
    def _on_mode(self, idx):
        self.state["mode"] = "refract" if idx == 0 else "nephroid"
        refract = self.state["mode"] == "refract"
        self.refract_group.setVisible(refract)
        self.reflect_group.setVisible(not refract)
        if not refract:
            self.view.invalidate()
        self.view.update()
        self._update_info()
        self._update_status()

    def _on_amp(self, v):
        self.state["amp"] = v / 1000.0
        self.view.update()
        self._update_status()

    def _on_wavelength(self, v):
        self.state["wavelength"] = v / 100.0
        self.view.update()
        self._update_status()

    def _on_index(self, v):
        self.state["index_n"] = v / 100.0
        self.view.update()
        self._update_status()

    def _on_depth(self, v):
        self.state["depth"] = v / 100.0
        self.view.update()
        self._update_status()

    def _on_theta(self, v):
        self.state["theta"] = v * math.pi / 1800.0
        self.view.update()
        self._update_status()

    def _on_radius(self, v):
        self.state["R"] = v / 100.0
        self.view.invalidate()
        self.view.update()
        self._update_status()

    # ------------------------------------------------------------------
    def _toggle_play(self):
        self.playing = not self.playing
        self.state["playing"] = self.playing
        if self.playing:
            self.play_btn.setText("暂停")
            self.timer.start()
        else:
            self.play_btn.setText("播放")
            self.timer.stop()
        self.view.invalidate()
        self.view.update()
        self._update_status()

    def _reset(self):
        self.timer.stop()
        self.playing = False
        self.state["playing"] = False
        self.play_btn.setText("播放")
        self.theta.setValue(2400)
        self.radius.setValue(100)
        self.amp.setValue(35)
        self.wavelength.setValue(55)
        self.index.setValue(133)
        self.depth.setValue(60)
        self.state.update(R=1.0, theta=4 * math.pi / 3, n_rays=15000,
                          amp=0.035, wavelength=0.55, index_n=1.33,
                          depth=0.6, phase=0.0)
        self.view.invalidate()
        self.view.update()
        self._update_status()

    def _tick(self):
        if self.state["mode"] == "refract":
            self.state["phase"] = (self.state["phase"] + 0.18) % (2 * math.pi)
            self.view.update()
            return
        v = self.theta.slider.value() + 10
        if v > 3588:
            v = 1812
        self.theta.setValue(v)


if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow))
