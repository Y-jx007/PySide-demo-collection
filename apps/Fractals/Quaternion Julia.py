import numpy as np
import taichi_forge as ti
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QSlider, QLineEdit, QGroupBox,
    QSplitter, QSizePolicy,
)
from OpenGL.GL import *

from common.app import run_app
from common.utils import OrbitCamera
from common.widgets import LabeledSlider

# ==================== Taichi 初始化 ====================
# 默认离线缓存目录 C:/taichi_cache 可能不可写，导致每次启动都重新 JIT
# 编译几十秒。这里显式指到用户可写目录，首帧之后再次启动可直接命中缓存。
import os
import tempfile

_ti_candidates = []
if os.environ.get("QJ_TAICHI_CACHE"):
    _ti_candidates.append(os.environ["QJ_TAICHI_CACHE"])
_base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
_ti_candidates.append(os.path.join(_base, "PySideDrawingTool", "taichi_cache"))
_ti_candidates.append(os.path.join(tempfile.gettempdir(), "PySideDrawingTool_taichi_cache"))

_ti_cache = None
for _candidate in _ti_candidates:
    try:
        os.makedirs(_candidate, exist_ok=True)
        _probe = os.path.join(_candidate, ".write_test")
        with open(_probe, "w") as _f:
            _f.write("1")
        os.remove(_probe)
        _ti_cache = _candidate
        break
    except OSError:
        continue
if _ti_cache is None:
    _ti_cache = tempfile.mkdtemp(prefix="PySideDrawingTool_taichi_")

ti.init(arch=ti.gpu, offline_cache_file_path=_ti_cache)

W, H = 640, 640
FOCAL_LENGTH = 3.0
BOUNDING_RADIUS = 3.5               # 扩大包围球
DEFAULT_MAX_ITER = 250
DEFAULT_SURFACE_PRECISION = 0.00025
DEFAULT_SAMPLES = 1
DEFAULT_ACCUM = 48                  # 累积帧数：到上限后停止重绘
RAY_TMAX = 7.0
DEFAULT_POWER = 2

curr = ti.Vector.field(3, ti.f32, (W, H))
accum = ti.Vector.field(3, ti.f32, (W, H))
acc_cnt = ti.field(ti.i32, ())
display = ti.Vector.field(3, ti.u8, (W, H))

section_plane = ti.Vector.field(4, ti.f32, ())
section_plane[None] = ti.Vector([0.0, 1.0, 0.0, 0.0])

# ---------- 四元数工具 ----------
@ti.func
def qmul(a, b):
    return ti.Vector([a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
                      a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                      a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
                      a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x])

@ti.func
def qsq(q):
    return ti.Vector([q.x*q.x - q.y*q.y - q.z*q.z - q.w*q.w,
                      2.0*q.x*q.y, 2.0*q.x*q.z, 2.0*q.x*q.w])

@ti.func
def qlen2(q): return q.dot(q)

@ti.func
def smoothstep(e0, e1, x):
    t = ti.max(0.0, ti.min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


# ---------- 廉价 3D 值噪声 / fbm：给岩壳、火山灰和熔岩裂纹加细节 ----------
@ti.func
def hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177) & 0x7fffffff
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7fffffff
    h = (h ^ (h >> 16)) & 0x00ffffff
    return ti.cast(h, ti.f32) * (1.0 / 16777216.0)


@ti.func
def vnoise(p):
    ip = ti.cast(ti.floor(p), ti.i32)
    f = p - ti.cast(ip, ti.f32)
    u = f * f * (3.0 - 2.0 * f)
    n000 = hash3(ip.x, ip.y, ip.z)
    n100 = hash3(ip.x + 1, ip.y, ip.z)
    n010 = hash3(ip.x, ip.y + 1, ip.z)
    n110 = hash3(ip.x + 1, ip.y + 1, ip.z)
    n001 = hash3(ip.x, ip.y, ip.z + 1)
    n101 = hash3(ip.x + 1, ip.y, ip.z + 1)
    n011 = hash3(ip.x, ip.y + 1, ip.z + 1)
    n111 = hash3(ip.x + 1, ip.y + 1, ip.z + 1)
    nx00 = n000 * (1.0 - u.x) + n100 * u.x
    nx10 = n010 * (1.0 - u.x) + n110 * u.x
    nx01 = n001 * (1.0 - u.x) + n101 * u.x
    nx11 = n011 * (1.0 - u.x) + n111 * u.x
    nxy0 = nx00 * (1.0 - u.y) + nx10 * u.y
    nxy1 = nx01 * (1.0 - u.y) + nx11 * u.y
    return nxy0 * (1.0 - u.z) + nxy1 * u.z


@ti.func
def fbm(p):
    return (vnoise(p) * 0.55
            + vnoise(p * 2.03 + ti.Vector([13.1, 7.7, 3.3])) * 0.30
            + vnoise(p * 4.07 + ti.Vector([41.7, 19.3, 5.9])) * 0.15)


@ti.func
def fbm2(p):
    """两倍频版本，给 bump 用：少 1/3 的噪声开销，视觉差别很小。"""
    return (vnoise(p) * 0.68
            + vnoise(p * 2.03 + ti.Vector([13.1, 7.7, 3.3])) * 0.32)


@ti.func
def bump_normal(pos, nor, scale, strength):
    """用 fbm 的有限差分扰动法线，让岩壳不再像塑料。"""
    base = fbm2(pos * scale)
    tang = ti.Vector([0.0, 0.0, 1.0]).cross(nor)
    if tang.norm() < 1e-3:
        tang = ti.Vector([1.0, 0.0, 0.0]).cross(nor)
    tang = tang.normalized()
    bitang = nor.cross(tang).normalized()
    e = 0.035
    nx = fbm2((pos + tang * e) * scale) - base
    ny = fbm2((pos + bitang * e) * scale) - base
    return (nor + (tang * nx + bitang * ny) * (strength / e)).normalized()


# ---------- 四元数快速幂 ----------
@ti.func
def qpow(z, p):
    """计算四元数 z 的 p 次幂 (p >= 0)"""
    res = ti.Vector([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    base = z
    k = p
    while k > 0:
        if k & 1:
            res = qmul(res, base)
        base = qmul(base, base)
        k >>= 1
    return res

# ---------- Julia 距离估计（任意正整数幂次）----------
@ti.func
def dist_estim(p, c, plane, max_iter, power):
    z = ti.Vector([p.x, p.y, p.z, 0.0])
    dz2 = 1.0
    m2 = 0.0
    n = 0.0
    for _ in range(max_iter):
        zm1 = qpow(z, power - 1)          # z^{power-1}
        dz2 *= (float(power) ** 2) * qlen2(zm1)
        z_pow = qmul(zm1, z)              # z^power
        z = z_pow + c
        m2 = qlen2(z)
        if m2 > 256.0:
            break
        n += 1.0
    d = 0.25 * ti.log(m2) * ti.sqrt(m2 / dz2)
    plane_dist = plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w
    d = ti.max(d, plane_dist)
    return ti.Vector([d, n])

# ---------- Mandelbrot 距离估计（任意正整数幂次）----------
@ti.func
def dist_estim_mandel(p, plane, max_iter, power):
    c = ti.Vector([p.x, p.y, p.z, 0.0])
    z = ti.Vector([0.0, 0.0, 0.0, 0.0])
    j = ti.Vector([0.0, 0.0, 0.0, 0.0])   # dz/dc
    m2 = 0.0
    n = 0.0
    one = ti.Vector([1.0, 0.0, 0.0, 0.0])

    for _ in range(max_iter):
        if m2 < 256.0:
            zm1 = qpow(z, power - 1)
            j = float(power) * qmul(zm1, j) + one
            z = qmul(zm1, z) + c
            m2 = qlen2(z)
            n += 1.0
        else:
            break

    dz2 = qlen2(j)
    d = 0.25 * ti.log(ti.max(m2, 1e-10)) * ti.sqrt(m2 / ti.max(dz2, 1e-20))
    plane_dist = plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w
    d = ti.max(d, plane_dist)
    return ti.Vector([d, n])

@ti.func
def normal(pos, c, plane, max_iter, power, prec, mode: ti.i32):
    e = ti.Vector([1.0, -1.0]) * 0.5773 * prec
    off = [ti.Vector([e.x, e.y, e.y]), ti.Vector([e.y, e.y, e.x]),
           ti.Vector([e.y, e.x, e.y]), ti.Vector([e.x, e.x, e.x])]
    grad = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(4)):
        if mode == 0:
            grad += off[k] * dist_estim(pos + off[k], c, plane, max_iter, power).x
        else:
            grad += off[k] * dist_estim_mandel(pos + off[k], plane, max_iter, power).x
    return grad.normalized()

@ti.func
def sphere_hit(ro, rd, rad):
    b = ro.dot(rd); c = ro.dot(ro) - rad*rad; h = b*b - c
    res = ti.Vector([-1.0, -1.0])
    if h >= 0.0:
        sq = ti.sqrt(h); res = ti.Vector([-b - sq, -b + sq])
    return res

@ti.func
def ray_march(ro, rd, c, plane, max_iter, power, rad, prec, mode: ti.i32):
    tmin = prec
    bv = sphere_hit(ro, rd, rad)
    hit, niters = -1.0, 0.0
    res = ti.Vector([0.0,0.0])
    if bv.y >= 0.0:
        tmin = ti.max(tmin, bv.x)
        tmax = ti.min(RAY_TMAX, bv.y)
        t = tmin
        for _ in range(1024):
            if mode == 0:
                res = dist_estim(ro + rd * t, c, plane, max_iter, power)
            else:
                res = dist_estim_mandel(ro + rd * t, plane, max_iter, power)
            if res.x < prec:
                hit = t; niters = res.y; break
            t += res.x * (0.5 + 0.5 * ti.random())
            if t > tmax: break
    return ti.Vector([hit, niters])

@ti.func
def background(rd):
    # 保持原版中性灰背景，不做烟尘/余烬染色
    return ti.Vector([0.2, 0.2, 0.22])


@ti.func
def shade(pos, nor, rd, t_norm, mode):
    # 多尺度噪声：大块火山灰、中频岩壳、细颗粒
    n_big = fbm(pos * 2.4 + ti.Vector([4.2, 1.7, 8.3]))
    n_mid = fbm(pos * 7.0 + ti.Vector([11.1, 3.9, 5.2]))
    n_fine = fbm(pos * 19.0 + ti.Vector([2.3, 9.4, 1.8]))

    up = smoothstep(-0.15, 0.95, nor.y)
    ash = smoothstep(0.38, 0.78, n_big + 0.25 * up) * (0.25 + 0.75 * up)
    ash *= smoothstep(0.35, 0.75, n_mid)

    basalt = ti.Vector([0.030, 0.026, 0.024])
    ash_col = ti.Vector([0.235, 0.208, 0.182])
    albedo = basalt * (1.0 - ash) + ash_col * ash
    albedo *= 0.55 + 0.65 * n_fine
    albedo *= 0.85 + 0.30 * smoothstep(0.55, 0.85, n_mid)
    # 细颗粒火山灰：迎风面更明显
    dust = smoothstep(0.60, 0.78, n_fine) * up
    albedo += ti.Vector([0.075, 0.066, 0.058]) * dust * (1.0 - ash * 0.5)

    # 迭代次数作为“温度”：越靠内部越热，再用噪声打碎成裂纹
    heat = smoothstep(0.48, 0.98, t_norm)
    vein = smoothstep(0.52, 0.66, n_mid * 0.7 + n_fine * 0.3)
    lava = smoothstep(0.25, 0.9, heat) * (0.30 + 0.70 * vein)
    core = smoothstep(0.86, 1.0, t_norm + 0.12 * (n_mid - 0.5))
    if mode == 1:
        # Mandelbrot 的 t_norm 在集合边界附近普遍接近 1，
        # 不压低的话会整片烧成白色，看不到岩壳。
        lava *= 0.30
        core *= 0.05
    # 细裂缝：即使没有高温核心，也在岩壳间透出暗红/橙色
    crack = smoothstep(0.60, 0.72, n_fine * 0.68 + n_mid * 0.32)

    deep_red = ti.Vector([0.85, 0.055, 0.004])
    orange = ti.Vector([4.2, 0.72, 0.045])
    white_hot = ti.Vector([6.0, 2.4, 0.55])
    emission = (deep_red * lava * 0.8
                + orange * (lava * lava) * 1.25
                + deep_red * crack * (0.18 + 0.82 * heat) * 0.9
                + orange * crack * heat * 0.55
                + white_hot * core)

    key = ti.Vector([-0.55, 0.35, 0.76]).normalized()
    sky = ti.Vector([0.25, 0.9, -0.25]).normalized()
    ndl = ti.max(nor.dot(key), 0.0)
    nds = ti.max(nor.dot(sky), 0.0)

    ambient = 0.035 + 0.22 * nds
    hvec = (key - rd).normalized()
    spec = ti.pow(ti.max(nor.dot(hvec), 0.0), 64.0) * 0.04 * (1.0 - ash * 0.9)
    bounce = ti.max(-nor.y, 0.0) * 0.14 * lava

    col = albedo * (ambient + ndl * 0.85)
    col += ti.Vector([1.0, 0.52, 0.22]) * (spec + bounce)
    col += emission
    rim = ti.pow(1.0 - ti.max(nor.dot(-rd), 0.0), 3.0)
    col += ti.Vector([1.0, 0.26, 0.045]) * rim * (0.08 + 0.45 * heat)
    return col

@ti.kernel
def render(ro: ti.types.vector(3, ti.f32), right: ti.types.vector(3, ti.f32),
           up: ti.types.vector(3, ti.f32), fwd: ti.types.vector(3, ti.f32),
           c_val: ti.types.vector(4, ti.f32), plane: ti.types.vector(4, ti.f32),
           max_iter: ti.i32, power: ti.i32, samples: ti.i32, prec: ti.f32,
           mode: ti.i32):
    ti.loop_config(block_dim=256)
    for i, j in curr:
        col = ti.Vector([0.0, 0.0, 0.0])
        for _ in range(samples):
            uv = ti.Vector([(i+ti.random())/W, (j+ti.random())/H])
            p = ti.Vector([2.0*uv.x-1.0, 2.0*uv.y-1.0]) * ti.Vector([W/H, 1.0])
            rd = (right*p.x + up*p.y + fwd*FOCAL_LENGTH).normalized()
            tn = ray_march(ro, rd, c_val, plane, max_iter, power, BOUNDING_RADIUS, prec, mode)
            t = tn.x
            if t < 0.0:
                col += background(rd)
            else:
                pos = ro + rd*t
                nor = normal(pos, c_val, plane, max_iter, power, prec, mode)
                nor = bump_normal(pos, nor, 9.0, 0.38)
                t_norm = ti.log(tn.y+1) / ti.log(ti.cast(max_iter, ti.f32)+1)
                surf = shade(pos, nor, rd, t_norm, mode)
                col += surf
        curr[i,j] = col / ti.cast(samples, ti.f32)
        accum[i,j] += curr[i,j]
    acc_cnt[None] += 1

@ti.kernel
def reset():
    for i, j in accum: accum[i,j] = ti.Vector([0.0,0.0,0.0])
    acc_cnt[None] = 0

@ti.kernel
def tonemap():
    # 原版色调映射：保持背景亮度和暗角与旧版本一致
    cnt = ti.cast(acc_cnt[None], ti.f32); inv = 1.0 / ti.max(cnt, 1.0)
    for i, j in display:
        avg = accum[i,j] * inv
        mapped = avg * 2.0 / (ti.Vector([1.0,1.0,1.0]) + avg)
        gamma = ti.pow(mapped, 0.4545)
        enhanced = gamma*0.5 + 0.5*gamma*gamma*(3.0-2.0*gamma)
        u = (ti.cast(i,ti.f32)+0.5)/W; v = (ti.cast(j,ti.f32)+0.5)/H
        enhanced *= 0.5 + 0.5 * ti.pow(16.0*u*v*(1.0-u)*(1.0-v), 0.1)
        display[i,j] = ti.cast(ti.min(ti.max(enhanced,0.0),1.0)*255, ti.u8)

# ==================== OpenGL 窗口 ====================
class FractalWidget(QOpenGLWidget):
    first_frame_ready = Signal()

    def __init__(self, cam):
        super().__init__()
        self.setMinimumSize(W, H)
        self.cam = cam; self.last_pos = None
        self.cx, self.cy, self.cz, self.cw = 0.0, 0.0, 0.675, 0.0
        self.max_iter = DEFAULT_MAX_ITER; self.samples = DEFAULT_SAMPLES
        self.prec = DEFAULT_SURFACE_PRECISION; self.power = DEFAULT_POWER
        self.max_accum = DEFAULT_ACCUM
        self.mode = 0                      # 0=Julia, 1=Mandelbrot
        self.need_reset = True; self.tex = None
        # 首帧前先让窗口显示出来，再在事件循环里编译/渲染；
        # 否则 show() 会同步卡在几十秒的 Taichi JIT 上。
        self._render_armed = False
        self._defer_scheduled = False
        self._first_frame_emitted = False
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(1)
        self.render_timer.timeout.connect(self.update)

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.22, 1.0)
        self.tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, None)
        glDisable(GL_DEPTH_TEST)

    def _arm_render(self):
        self._render_armed = True
        self.kick()

    def kick(self):
        """重新开始累积（相机/参数变化后调用）。"""
        self.need_reset = True
        if self._render_armed and not self.render_timer.isActive():
            self.render_timer.start(0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if not self._render_armed:
            # 让窗口先出现，把 JIT 放到事件循环里执行。
            if not self._defer_scheduled:
                self._defer_scheduled = True
                QTimer.singleShot(50, self._arm_render)
            return

        self.cam._update_vectors()
        if self.need_reset:
            reset(); self.need_reset = False

        if acc_cnt[None] < self.max_accum:
            m = self.cam.model_matrix()
            ro = m.map(self.cam.position - self.cam.target) + self.cam.target
            rd = m.mapVector(self.cam.front); ri = m.mapVector(self.cam.right); up = m.mapVector(self.cam.up)
            ro_ti = ti.Vector([ro.x(), ro.y(), ro.z()]); ri_ti = ti.Vector([ri.x(), ri.y(), ri.z()])
            up_ti = ti.Vector([up.x(), up.y(), up.z()]); rd_ti = ti.Vector([rd.x(), rd.y(), rd.z()])
            c_ti = ti.Vector([self.cx, self.cy, self.cz, self.cw]); pl_ti = section_plane[None]
            render(ro_ti, ri_ti, up_ti, rd_ti, c_ti, pl_ti, self.max_iter,
                   self.power, self.samples, self.prec, self.mode)
            tonemap(); ti.sync()
            arr = display.to_numpy().transpose(1, 0, 2)
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB,
                            GL_UNSIGNED_BYTE, arr.tobytes())
            if not self._first_frame_emitted:
                self._first_frame_emitted = True
                self.first_frame_ready.emit()
            if acc_cnt[None] < self.max_accum:
                self.render_timer.start(1)

        ratio = self.devicePixelRatioF(); w, h = int(self.width()*ratio), int(self.height()*ratio)
        glViewport(0,0,w,h); glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glEnable(GL_TEXTURE_2D); glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0); glTexCoord2f(1,0); glVertex2f(w,0)
        glTexCoord2f(1,1); glVertex2f(w,h); glTexCoord2f(0,1); glVertex2f(0,h)
        glEnd(); glDisable(GL_TEXTURE_2D)

    def resizeEvent(self, e):
        s = min(self.width(), self.height()); self.setFixedSize(s, s)
        super().resizeEvent(e)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.last_pos = e.position()
            self.cam.start_arcball(e.position().x(), e.position().y(), self.width(), self.height())
        elif e.button() == Qt.MiddleButton: self.last_pos = e.position()

    def mouseMoveEvent(self, e):
        if not self.last_pos: return
        dx = e.position().x() - self.last_pos.x(); dy = e.position().y() - self.last_pos.y()
        if e.buttons() & Qt.LeftButton:
            if e.modifiers() & Qt.ControlModifier: self.cam.roll_model(dx*0.5)
            else: self.cam.update_arcball(e.position().x(), e.position().y(), self.width(), self.height())
            self.kick()
        elif e.buttons() & Qt.MiddleButton:
            self.cam.pan(-dx*0.1, dy*0.1)
            self.kick()
        self.last_pos = e.position()

    def closeEvent(self, e):
        self.render_timer.stop()
        super().closeEvent(e)

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("四元数 Julia / Mandelbrot - 熔岩与火山灰")
        self.resize(1100, 750)
        self.cam = OrbitCamera(QVector3D(0,0,0), 3.6); self.cam.pitch = -30; self.cam._update_vectors()
        self._ui()

    def _ui(self):
        splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(splitter)
        panel = QWidget(); panel.setFixedWidth(300)
        lay = QVBoxLayout(panel); lay.setContentsMargins(10,10,10,10); lay.setSpacing(6)

        # 分形参数组（类型 + 幂次）
        param_group = QGroupBox("分形参数")
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Julia", "Mandelbrot"])
        self.type_combo.setCurrentIndex(0)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        hbox.addWidget(self.type_combo)

        hbox.addWidget(QLabel("幂次:"))
        self.power_spin = QSpinBox()
        self.power_spin.setRange(2, 50)
        self.power_spin.setValue(DEFAULT_POWER)
        self.power_spin.valueChanged.connect(lambda v: (setattr(self.gl, 'power', v), self._reset()))
        hbox.addWidget(self.power_spin)
        hbox.addStretch()
        param_group.setLayout(hbox)
        lay.addWidget(param_group)

        # 渲染参数
        param = QGroupBox("渲染参数"); g = QGridLayout(); g.setSpacing(4)
        g.addWidget(QLabel("迭代次数"),0,0)
        self.iter_s = LabeledSlider("", 50, 500, DEFAULT_MAX_ITER)
        self.iter_s.valueChanged.connect(lambda _v: self._reset())
        g.addWidget(self.iter_s,0,1,1,2)

        g.addWidget(QLabel("精度"),1,0)
        self.prec_s = LabeledSlider("", 1, 50, int(DEFAULT_SURFACE_PRECISION*100000),
                                    formatter=lambda v: f"{v/100000:.5f}", value_width=52)
        self.prec_s.valueChanged.connect(lambda _v: self._reset())
        g.addWidget(self.prec_s,1,1,1,2)

        g.addWidget(QLabel("采样"),2,0)
        self.samp_s = LabeledSlider("", 1, 8, DEFAULT_SAMPLES)
        self.samp_s.valueChanged.connect(lambda _v: self._reset())
        g.addWidget(self.samp_s,2,1,1,2)

        g.addWidget(QLabel("累积帧"),3,0)
        self.accum_s = LabeledSlider("", 1, 128, DEFAULT_ACCUM)
        self.accum_s.valueChanged.connect(self._on_accum_changed)
        g.addWidget(self.accum_s,3,1,1,2)
        param.setLayout(g); lay.addWidget(param)

        # C 常数 (仅 Julia 可用)
        self.c_group = QGroupBox("C 常数")
        cg = QGridLayout()
        c_vals = [0.0,0.0,0.675,0.0]; self.c_sliders=[]; self.c_inputs=[]
        for i,(nm,v) in enumerate(zip(["Cx","Cy","Cz","Cw"],c_vals)):
            lb=QLabel(nm); sl=QSlider(Qt.Horizontal); sl.setRange(-1000,1000); sl.setValue(int(v*1000))
            inp=QLineEdit(); inp.setFixedWidth(65); inp.setText(f"{v:.4f}")
            sl.valueChanged.connect(lambda val, idx=i, s=sl, inp=inp: (self._on_c_slider(idx, val, inp)))
            inp.textEdited.connect(lambda txt, idx=i, s=sl: (self._on_c_input(idx, txt, s)))
            self.c_sliders.append(sl); self.c_inputs.append(inp)
            cg.addWidget(lb,i,0); cg.addWidget(sl,i,1); cg.addWidget(inp,i,2)
        self.c_group.setLayout(cg); lay.addWidget(self.c_group)

        # 截面平面
        pl_group = QGroupBox("截面平面"); pg = QGridLayout()
        p_vals = [0.0,1.0,0.0,0.0]; self.pl_sliders=[]; self.pl_inputs=[]; self.pl_vals=p_vals.copy()
        for i,nm in enumerate("ABCD"):
            lb=QLabel(nm); sl=QSlider(Qt.Horizontal); sl.setRange(-1000,1000); sl.setValue(int(p_vals[i]*1000))
            inp=QLineEdit(); inp.setFixedWidth(65); inp.setText(f"{p_vals[i]:.4f}")
            sl.valueChanged.connect(lambda val, idx=i, s=sl, inp=inp: (self._on_plane_slider(idx, val, inp)))
            inp.textEdited.connect(lambda txt, idx=i, s=sl: (self._on_plane_input(idx, txt, s)))
            self.pl_sliders.append(sl); self.pl_inputs.append(inp)
            pg.addWidget(lb,i,0); pg.addWidget(sl,i,1); pg.addWidget(inp,i,2)
        self.pl_norm_l = QLabel("归一化: (0.00,1.00,0.00,0.00)")
        pg.addWidget(self.pl_norm_l,4,0,1,3)
        btn_reset_pl = QPushButton("重置 y=0"); btn_reset_pl.clicked.connect(self._reset_plane)
        pg.addWidget(btn_reset_pl,5,0,1,3)
        pl_group.setLayout(pg); lay.addWidget(pl_group)

        # 相机距离
        dist_g = QGroupBox("相机距离"); dh = QHBoxLayout()
        self.dist_s = LabeledSlider("距离:", 1, 200, 36,
                                    formatter=lambda v: f"{v/10:.1f}", value_width=40)
        self.dist_s.valueChanged.connect(self._set_dist)
        dh.addWidget(self.dist_s)
        btn_reset_view = QPushButton("重置视角"); btn_reset_view.clicked.connect(self._reset_cam)
        dh.addWidget(btn_reset_view); dist_g.setLayout(dh); lay.addWidget(dist_g)
        lay.addStretch()

        self.gl = FractalWidget(self.cam)
        self.gl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.gl.first_frame_ready.connect(self._on_first_frame)
        splitter.addWidget(panel); splitter.addWidget(self.gl)
        self.statusBar().showMessage("正在编译 GPU 着色器（首次启动可能需要几十秒）…")

    def _on_first_frame(self):
        self.statusBar().clearMessage()

    def _on_type_changed(self, idx):
        self.gl.mode = idx
        self.c_group.setEnabled(idx == 0)   # Julia 时启用 C 常数
        self._reset()

    def _on_c_slider(self, idx, val, inp):
        v = val / 1000.0
        inp.setText(f"{v:.4f}")
        setattr(self.gl, ['cx','cy','cz','cw'][idx], v)
        self._reset()

    def _on_c_input(self, idx, txt, sl):
        try:
            v = float(txt); v = max(-1.0, min(1.0, v))
            sl.setValue(int(v * 1000))
            setattr(self.gl, ['cx','cy','cz','cw'][idx], v)
            self._reset()
        except ValueError: pass

    def _on_plane_slider(self, idx, val, inp):
        v = val / 1000.0
        inp.setText(f"{v:.4f}")
        self.pl_vals[idx] = v
        self._update_plane()

    def _on_plane_input(self, idx, txt, sl):
        try:
            v = float(txt); v = max(-1.0, min(1.0, v))
            sl.setValue(int(v * 1000))
            self.pl_vals[idx] = v
            self._update_plane()
        except ValueError: pass

    def _update_plane(self):
        a,b,c,d = self.pl_vals
        L = np.sqrt(a*a + b*b + c*c)
        if L < 1e-12:
            self.pl_norm_l.setText("无效")
            return
        na,nb,nc = a/L, b/L, c/L
        nd = d/L
        section_plane[None] = ti.Vector([na, nb, nc, nd])
        self.pl_norm_l.setText(f"归一化: ({na:.2f},{nb:.2f},{nc:.2f},{nd:.2f})")
        self._reset()

    def _reset_plane(self):
        self.pl_vals = [0, 1, 0, 0]
        for sl, inp, v in zip(self.pl_sliders, self.pl_inputs, self.pl_vals):
            sl.setValue(int(v * 1000))
            inp.setText(f"{v:.4f}")
        self._update_plane()

    def _on_accum_changed(self, v):
        if getattr(self, "gl", None) is not None:
            self.gl.max_accum = v
            self.gl.kick()

    def _reset(self):
        self.gl.max_iter = self.iter_s.value()
        self.gl.prec = self.prec_s.value() / 100000
        self.gl.samples = self.samp_s.value()
        self.gl.power = self.power_spin.value()
        self.gl.max_accum = self.accum_s.value()
        self.gl.kick()

    def _set_dist(self, v):
        self.cam.distance = v / 10.0
        self.cam._update_vectors()
        self._reset()

    def _reset_cam(self):
        self.cam.reset()
        self.dist_s.setValue(36)
        self._reset()

    def closeEvent(self, e):
        e.accept()

if __name__ == "__main__":
    # before_exec：先把窗口画出来再跑首帧 JIT，避免启动时白屏
    raise SystemExit(run_app(MainWindow, gl=True,
                             before_exec=lambda app: app.processEvents()))
