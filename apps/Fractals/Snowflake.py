from common.field_canvas import ArrayCanvas
from common.qt_base import (ColorButton, LabeledSlider, QCheckBox, QColor, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                            QImage, QLabel, QMainWindow, QPainter, QPushButton, QSpinBox, QTimer, QVBoxLayout,
                            QWidget, Qt, math, run_app)
from common.numpy_base import (np)
import json as _json
import os as _os
from PySide6.QtCore import QPointF, QRectF, QSize, Signal
from PySide6.QtGui import QIcon, QImage, QPen
from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QToolButton
from PySide6.QtGui import QImage

SQ3 = math.sqrt(3.0)

# 六角格 6 个邻居方向（行偏移=Δr, 列偏移=Δq）
HEX_DIRS = ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0))

PRESETS = {
    "枝晶 Dendrite": (0.90, 0.40, 0.0008),
    "羽毛 Feather": (0.45, 0.38, 0.0003),
    "宽枝 Wide": (0.60, 0.42, 0.0015),
    "密实 Dense": (0.70, 0.50, 0.0040),
    "板状 Plate": (1.00, 0.45, 0.0000),
}


def _blur3(a):
    """3×3 盒式模糊（软边用）。"""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


class SnowCrystal:
    """Reiter 雪花模型，方形域 + 六角格。

    每格含水量 s 按类别拆成参与扩散的 u 与只累积的 v：
        u = 0, v = s                （可接收格：已冻结或与冻结格相邻）
        u = s, v = 0                （其余格）
    每步 v += γ（仅可接收格）、u += α/2·(⟨u⟩−u)、方形边界格 u = β，
    s = u + v，s ≥ 1 即冻结。扩散供水的快慢决定长成枝晶还是实心六边形。
    """

    def __init__(self, size=420, alpha=0.9, beta=0.40, gamma=0.0008, cluster=False):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.steps = 0

        # 方形域半边长（格），r 是行、q 是列
        S = size / 2.0
        rr_max = int(S / (SQ3 / 2.0)) + 1
        qq_max = int(S + rr_max * 0.5) + 2
        rows = np.arange(-rr_max, rr_max + 1)
        cols = np.arange(-qq_max - 1, qq_max + 2)
        R, Q = np.meshgrid(rows, cols, indexing="ij")
        X = Q + R * 0.5
        Y = R * (SQ3 / 2.0)
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.mask = (np.abs(X) <= S) & (np.abs(Y) <= S)
        self.nrows, self.ncols = self.mask.shape

        # 方形边界（离边界 1 格以内）作为水汽入口
        self.edge = self.mask & ((np.abs(X) >= S - 1.5) | (np.abs(Y) >= S - 1.5))

        self.nb_cnt = np.maximum(self._shift_sum(self.mask.astype(np.float32)), 1.0)
        self._inv_cnt = (1.0 / self.nb_cnt).astype(np.float32)
        self._mask_f = self.mask.astype(np.float32)

        self.s = np.where(self.mask, beta, 0.0).astype(np.float32)
        centre = (self.nrows // 2, self.ncols // 2)
        if cluster:
            d = np.sqrt(X.astype(np.float32) ** 2 + Y.astype(np.float32) ** 2)
            self.s[(d <= 2.5) & self.mask] = 1.0
        else:
            self.s[centre] = 1.0

        # 增量维护“已冻结 / 可接收”两张表，省掉每步重算邻里冻结数
        self._frozen = self.s >= 1.0
        self._receptive = np.zeros(self.s.shape, dtype=bool)
        self._receptive_f = np.zeros(self.s.shape, dtype=np.float32)
        _rs, _cs = np.nonzero(self._frozen)
        self._mark_receptive(_rs, _cs)
        self._xmin, self._xmax = float(self.X[_rs, _cs].min()), float(self.X[_rs, _cs].max())
        self._ymin, self._ymax = float(self.Y[_rs, _cs].min()), float(self.Y[_rs, _cs].max())
        self._rmin, self._rmax = int(_rs.min()), int(_rs.max())
        self._cmin, self._cmax = int(_cs.min()), int(_cs.max())

    # ---------- 计算 ----------
    def _shift_sum(self, a):
        acc = np.zeros_like(a)
        nr, nc = a.shape
        for dr, dc in HEX_DIRS:
            r0 = max(0, -dr)
            r1 = nr - max(0, dr)
            c0 = max(0, -dc)
            c1 = nc - max(0, dc)
            acc[r0:r1, c0:c1] += a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        return acc

    def _mark_receptive(self, rs, cs):
        # 把 (rs, cs) 及其六个邻格标为可接收水汽
        nr, nc = self.s.shape
        self._receptive[rs, cs] = True
        self._receptive_f[rs, cs] = 1.0
        for dr, dc in HEX_DIRS:
            rr = rs + dr
            cc = cs + dc
            ok = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
            self._receptive[rr[ok], cc[ok]] = True
            self._receptive_f[rr[ok], cc[ok]] = 1.0

    @property
    def frozen(self):
        return self.s >= 1.0

    @property
    def ice_count(self):
        return int((self.s >= 1.0).sum())

    @property
    def extent(self):
        # 方形计算域的“长满”程度：冻结区外接框跨度 / 域边长
        if self._xmax <= self._xmin:
            return 0.0
        # 两个方向都要顶到边界才算长满
        span = min(self._xmax - self._xmin, self._ymax - self._ymin)
        return float(span / self.size)

    MARGIN = 22          # 只在晶体周围的窗口内演化，窗口外恒为 β

    def step(self, n=1):
        s = self.s
        nr, nc = s.shape
        gamma = self.gamma
        half_alpha = 0.5 * self.alpha
        beta = self.beta
        margin = self.MARGIN
        for _ in range(n):
            r0 = max(0, self._rmin - margin)
            r1 = min(nr, self._rmax + margin + 1)
            c0 = max(0, self._cmin - margin)
            c1 = min(nc, self._cmax + margin + 1)

            # 外扩 1 格做缓冲，保证写回区域内的邻域和是完整的
            rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
            rr1, cc1 = min(nr, r1 + 1), min(nc, c1 + 1)
            sub = s[rr0:rr1, cc0:cc1]

            rmask = self._receptive_f[rr0:rr1, cc0:cc1]
            u = sub * (1.0 - rmask)
            u += half_alpha * (self._shift_sum(u) * self._inv_cnt[rr0:rr1, cc0:cc1] - u)
            u *= self._mask_f[rr0:rr1, cc0:cc1]

            new = u + (sub + gamma) * rmask
            new[self.edge[rr0:rr1, cc0:cc1]] = beta
            np.clip(new, 0.0, 4.0, out=new)

            s[r0:r1, c0:c1] = new[r0 - rr0:r1 - rr0, c0 - cc0:c1 - cc0]
            self.steps += 1

            # 只处理新冻结的格子，把它们的邻域并入可接收区
            fresh = s[r0:r1, c0:c1] >= 1.0
            fresh &= ~self._frozen[r0:r1, c0:c1]
            if fresh.any():
                rs, cs = np.nonzero(fresh)
                rs += r0
                cs += c0
                self._frozen[rs, cs] = True
                self._mark_receptive(rs, cs)
                _x = self.X[rs, cs]
                _y = self.Y[rs, cs]
                self._xmin = min(self._xmin, float(_x.min()))
                self._xmax = max(self._xmax, float(_x.max()))
                self._ymin = min(self._ymin, float(_y.min()))
                self._ymax = max(self._ymax, float(_y.max()))
                self._rmin = min(self._rmin, int(rs.min()))
                self._rmax = max(self._rmax, int(rs.max()))
                self._cmin = min(self._cmin, int(cs.min()))
                self._cmax = max(self._cmax, int(cs.max()))

class SnowCanvas(ArrayCanvas):
    """像素→格子查表 + 冰晶着色（软边、假光照、辉光）。"""

    def __init__(self, crystal, main_window=None):
        # aspect=0：跟随渲染图像自身的比例（图像尺寸随控件变化）
        super().__init__(minimum=(420, 420), renderer=self._render_image, aspect=0.0)
        self.crystal = crystal
        self.main_window = main_window
        self.ice_color = QColor(150, 205, 255)
        self.bg_color = QColor(238, 244, 250)
        self.vapor_color = QColor(96, 132, 190)
        self.glow_color = QColor(120, 180, 255)
        self.show_vapor = True
        self._map = None
        self._lut = None

    def set_crystal(self, crystal):
        self.crystal = crystal
        self._map = None
        self.invalidate()

    # ---------- 像素映射 ----------
    RENDER_CAP = 640

    def _build_map(self):
        c = self.crystal
        w, h = max(1, self.width()), max(1, self.height())
        cap = self.RENDER_CAP
        if max(w, h) > cap:
            f = cap / float(max(w, h))
            w, h = max(1, int(w * f)), max(1, int(h * f))
        side = min(w, h)
        scale = (side / 2.0 - 4) / (c.size / 2.0)
        xs = (np.arange(w, dtype=np.float32) - w / 2.0 + 0.5) / scale
        ys = (np.arange(h, dtype=np.float32) - h / 2.0 + 0.5) / scale
        r2 = ys[:, None] / (SQ3 / 2.0)
        q2 = xs[None, :] - r2 / 2.0
        ri = np.rint(r2).astype(np.int32) + c.nrows // 2
        ci = np.rint(q2).astype(np.int32) + c.ncols // 2
        inside = (ri >= 0) & (ri < c.nrows) & (ci >= 0) & (ci < c.ncols)
        ri = np.clip(ri, 0, c.nrows - 1)
        ci = np.clip(ci, 0, c.ncols - 1)
        idx = np.where(inside & c.mask[ri, ci], ri * c.ncols + ci, c.mask.size)
        self._map = idx.astype(np.int32)

    # ---------- 着色 ----------
    def _build_lut(self):
        """256 级渐变：底色 → 水汽 → 冰 → 高光。"""
        stops = [
            (0.00, np.array([self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()], np.float32)),
            (0.30, np.array([self.vapor_color.red(), self.vapor_color.green(), self.vapor_color.blue()], np.float32) * 0.55
             + np.array([self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()], np.float32) * 0.45),
            (0.62, np.array([self.ice_color.red(), self.ice_color.green(), self.ice_color.blue()], np.float32) * 0.82),
            (0.88, np.array([self.ice_color.red(), self.ice_color.green(), self.ice_color.blue()], np.float32)),
            (1.00, np.array([252, 253, 255], np.float32)),
        ]
        xs = np.linspace(0.0, 1.0, 256)
        lut = np.empty((256, 3), dtype=np.uint8)
        for ch in range(3):
            xs_stop = [s[0] for s in stops]
            ys_stop = [s[1][ch] for s in stops]
            lut[:, ch] = np.interp(xs, xs_stop, ys_stop)
        self._lut = lut

    def _render_image(self):
        c = self.crystal
        if self._map is None:
            self._build_map()
        if self._lut is None:
            self._build_lut()

        frozen = (c.s >= 1.0).astype(np.float32)
        soft = _blur3(frozen)                  # 冰边缘：一次模糊，保留细枝
        glow = _blur3(_blur3(soft))            # 外层辉光
        thick = np.clip(c.s - 1.0, 0.0, 1.2) * (1.0 / 1.2)

        # 软边梯度 → 假光照（光来自左上）
        gy, gx = np.gradient(soft)
        light = np.clip(1.0 - 1.5 * gx + 1.5 * gy, 0.15, 1.9)

        if self.show_vapor:
            val = np.clip(c.s, 0.0, 1.0) * 0.30 * (1.0 - soft)
        else:
            val = np.zeros_like(soft)
        val += soft * 0.74
        val += soft * (light - 1.0) * 0.22 + soft * thick * 0.18
        val += glow * (1.0 - soft) * 0.22
        val = np.clip(val, 0.0, 1.0)

        code = (val * 255).astype(np.uint8)
        cells = self._lut[code.ravel()]
        cells = np.vstack([cells, self._lut[0][None, :]])

        return cells[self._map]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._map = None
        self.invalidate()


from common.paths import PRESETS_DIR

_ALPHA_DIR = PRESETS_DIR / 'snowflake_alpha'
_TERRAIN_DIR = PRESETS_DIR / 'snowflake_terrain'
_TERRAIN_NPZ = _TERRAIN_DIR / 'terrain.npz'
_META_JSON = _TERRAIN_DIR / 'meta.json'


class SnowMapView(QWidget):
    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.meta = _json.load(open(_META_JSON, encoding='utf-8')) if _os.path.exists(_META_JSON) else None
        self.alphas = np.array(self.meta['alphas']) if self.meta else np.array([0.1])
        self.betas = np.array(self.meta['betas']) if self.meta else np.array([0.1, 0.9])
        self.gammas = np.array(self.meta['gammas']) if self.meta else np.array([0.0, 0.006])
        self.aidx = 0
        self._img = None
        self._bm = float(self.betas[0])
        self._gm = float(self.gammas[0])
        self.setFixedSize(250, 250)
        self.setCursor(Qt.CrossCursor)
        self._min_scale = 1.0
        self._scale = 1.0
        self._center = QPointF(0.5, 0.5)
        self._press = None
        self._moved = False
        self._load()

    def _load(self):
        path = _os.path.join(_TERRAIN_DIR, 'sheet_%d.png' % self.aidx)
        self._img = QImage(path) if _os.path.exists(path) else None
        self.update()

    def set_alpha_index(self, i):
        i = max(0, min(int(i), len(self.alphas) - 1))
        if i != self.aidx:
            self.aidx = i
            self._load()

    def set_marker(self, beta, gamma):
        self._bm, self._gm = float(beta), float(gamma)
        self.update()

    def reset_view(self):
        self._scale = self._min_scale
        self._center = QPointF(0.5, 0.5)
        self.update()

    def _src(self):
        half = 0.5 / self._scale
        cx = min(max(self._center.x(), half), 1.0 - half)
        cy = min(max(self._center.y(), half), 1.0 - half)
        return QRectF(cx - half, cy - half, 2 * half, 2 * half)

    def _to_norm(self, pos):
        r = self._src()
        return r.left() + pos.x() / self.width() * r.width(), r.top() + pos.y() / self.height() * r.height()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 20))
        if self._img is not None:
            r = self._src()
            src = QRectF(r.left() * self._img.width(), r.top() * self._img.height(),
                         r.width() * self._img.width(), r.height() * self._img.height())
            p.setRenderHint(QPainter.SmoothPixmapTransform, self._scale < 1.0)
            p.drawImage(QRectF(self.rect()), self._img, src)
        # 参数 -> 屏幕位置：格心在 (j+0.5)/N，首末格心之间线性映射
        nb, ng = len(self.betas), len(self.gammas)
        tx = (self._bm - self.betas[0]) / (self.betas[-1] - self.betas[0])
        ty = (self._gm - self.gammas[0]) / (self.gammas[-1] - self.gammas[0])
        bx = 0.5 / nb + tx * (nb - 1) / nb
        by = 1.0 - (0.5 / ng + ty * (ng - 1) / ng)
        r = self._src()
        px = (bx - r.left()) / r.width() * self.width()
        py = (by - r.top()) / r.height() * self.height()
        if -20 <= px <= self.width() + 20 and -20 <= py <= self.height() + 20:
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 80, 80), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(px, py), 5, 5)

    def wheelEvent(self, event):
        f = 1.25 ** (event.angleDelta().y() / 120.0)
        max_scale = max(1.0, (self._img.width() / float(self.width())) if self._img is not None else 1.0)
        ns = min(max(self._scale * f, 1.0), max_scale)
        if ns == self._scale:
            return
        nx, ny = self._to_norm(event.position())
        self._scale = ns
        self._center = QPointF(nx - (event.position().x() / self.width() - 0.5) / ns,
                               ny - (event.position().y() / self.height() - 0.5) / ns)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = event.position()
            self._moved = False

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        d = event.position() - self._press
        if abs(d.x()) + abs(d.y()) > 2:
            self._moved = True
        if self._moved:
            self._center = QPointF(self._center.x() - d.x() / self.width() / self._scale,
                                   self._center.y() - d.y() / self.height() / self._scale)
            self._press = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._press is None:
            return
        was = self._moved
        self._press = None
        if was:
            return
        nx, ny = self._to_norm(event.position())
        nb, ng = len(self.betas), len(self.gammas)
        tx = min(max((nx - 0.5 / nb) / ((nb - 1) / nb), 0.0), 1.0)
        ty = min(max((1.0 - ny - 0.5 / ng) / ((ng - 1) / ng), 0.0), 1.0)
        b = self.betas[0] + tx * (self.betas[-1] - self.betas[0])
        g = self.gammas[0] + ty * (self.gammas[-1] - self.gammas[0])
        self.picked.emit(float(min(max(b, self.betas[0]), self.betas[-1])),
                         float(min(max(g, self.gammas[0]), self.gammas[-1])))

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

class SnowApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雪花生长 Reiter Snowflake")
        self.setGeometry(120, 70, 1120, 800)

        self.grid_size = 420
        self.alpha, self.beta, self.gamma = PRESETS["枝晶 Dendrite"]
        self.speed = 30
        self.auto_stop = True

        self.crystal = SnowCrystal(self.grid_size, self.alpha, self.beta, self.gamma)
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.update_stats()

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        panel = QWidget()
        panel.setFixedWidth(268)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(6)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = SnowCanvas(self.crystal, self)

        self._build_preset_group(panel_layout)
        self._build_param_group(panel_layout)
        self._build_map_group(panel_layout)
        panel_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(292)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        root.addWidget(scroll)
        root.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("就绪")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_ice = QLabel("冰格 0")
        self.lbl_steps = QLabel("步数 0")
        self.statusBar().addPermanentWidget(self.lbl_ice)
        self.statusBar().addPermanentWidget(self.lbl_steps)

    def _build_preset_group(self, layout):
        group = QGroupBox('控制')
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        vbox.addWidget(self.preset_combo)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.run_btn = QPushButton('开始')
        self.run_btn.clicked.connect(self.toggle_run)
        row.addWidget(self.run_btn, 1)
        btn_reset = QPushButton('重新生长')
        btn_reset.clicked.connect(self.reset)
        row.addWidget(btn_reset, 1)
        step_btn = QPushButton('单步')
        step_btn.clicked.connect(self.single_step)
        row.addWidget(step_btn, 1)
        vbox.addLayout(row)

        form = QFormLayout()
        form.setSpacing(3)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speed_slider = LabeledSlider("", 1, 200, self.speed,
                                         formatter=lambda v: f"{v} 步/帧", value_width=66)
        self.speed_slider.valueChanged.connect(self.on_speed)
        form.addRow('速度', self.speed_slider)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(240, 640)
        self.size_spin.setSingleStep(60)
        self.size_spin.setValue(self.grid_size)
        self.size_spin.editingFinished.connect(self.apply_grid_size)
        form.addRow('网格', self.size_spin)
        self.auto_check = QCheckBox('长满自动停')
        self.auto_check.setChecked(self.auto_stop)
        self.auto_check.toggled.connect(lambda c: setattr(self, 'auto_stop', bool(c)))
        form.addRow('', self.auto_check)
        vbox.addLayout(form)

        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self.vapor_check = QCheckBox('水汽')
        self.vapor_check.setChecked(True)
        self.vapor_check.toggled.connect(self.toggle_vapor)
        row3.addWidget(self.vapor_check)
        row3.addWidget(QLabel('冰色'))
        self.ice_btn = ColorButton(self.canvas.ice_color)
        self.ice_btn.colorChanged.connect(lambda c: self._set_color('ice_color', c))
        row3.addWidget(self.ice_btn)
        row3.addWidget(QLabel('底色'))
        self.bg_btn = ColorButton(self.canvas.bg_color)
        self.bg_btn.colorChanged.connect(lambda c: self._set_color('bg_color', c))
        row3.addWidget(self.bg_btn)
        row3.addStretch()
        vbox.addLayout(row3)

        layout.addWidget(group)

    def _build_map_group(self, layout):
        group = QGroupBox('beta-gamma 地图')
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(3)
        self.map_view = SnowMapView()
        self.map_view.picked.connect(self.apply_bg)
        vbox.addWidget(self.map_view, 0, Qt.AlignHCenter)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.map_alpha = LabeledSlider('alpha', 0, max(0, len(self.map_view.alphas) - 1), 0,
                                       formatter=lambda i: f"{self.map_view.alphas[i]:.2f}",
                                       value_width=34)
        self.map_alpha.valueChanged.connect(self._on_map_alpha)
        row.addWidget(self.map_alpha, 1)
        vbox.addLayout(row)
        hint = QLabel('滚轮缩放 · 拖动平移 · 双击复位')
        hint.setStyleSheet('color: #8a8a94;')
        vbox.addWidget(hint)
        layout.addWidget(group)
        i = int(np.argmin(np.abs(self.map_view.alphas - self.alpha)))
        self.map_alpha.setValue(i)
        self._on_map_alpha(i)
        self.map_view.set_marker(self.beta, self.gamma)

    def _on_map_alpha(self, i):
        self.map_view.set_alpha_index(i)

    def apply_bg(self, beta, gamma):
        running = self.timer.isActive()
        alpha = float(self.map_view.alphas[self.map_view.aidx])
        self.alpha_slider.setValue(int(round(alpha * 100)))
        self.beta_slider.setValue(int(round(beta * 100)))
        self.gamma_slider.setValue(int(round(gamma * 10000)))
        self.reset()
        self._start_run()
        self.map_view.set_marker(beta, gamma)

    def _build_param_group(self, layout):
        group = QGroupBox("参数")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.alpha_slider = self._add_slider(form, "扩散 α", 10, 100, int(self.alpha * 100),
                                             lambda v: v / 100.0)
        self.beta_slider = self._add_slider(form, "湿度 β", 10, 90, int(self.beta * 100),
                                            lambda v: v / 100.0)
        self.gamma_slider = self._add_slider(form, "沉积 γ", 0, 100, int(self.gamma * 10000),
                                             lambda v: v / 10000.0)
        layout.addWidget(group)
    def _add_slider(self, form, text, lo, hi, value, transform):
        widget = LabeledSlider("", lo, hi, value, formatter=lambda v: f"{transform(v):.4g}",
                               value_width=44)
        widget.valueChanged.connect(lambda _v: self.apply_params())
        form.addRow(text, widget)
        return widget

    # ---------- 逻辑 ----------
    def _set_color(self, attr, color):
        setattr(self.canvas, attr, color)
        self.canvas._lut = None
        self.canvas.invalidate()

    def apply_preset(self, name):
        if name not in PRESETS:
            return
        self.alpha, self.beta, self.gamma = PRESETS[name]
        for slider, value in ((self.alpha_slider, int(self.alpha * 100)),
                              (self.beta_slider, int(self.beta * 100)),
                              (self.gamma_slider, int(self.gamma * 10000))):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self.reset()
        self._start_run()

    def _start_run(self):
        if not self.timer.isActive():
            self.timer.start(33)
        self.run_btn.setText('暂停')
        self.lbl_status.setText('生长中…')

    def apply_params(self):
        self.alpha = self.alpha_slider.value() / 100.0
        self.beta = self.beta_slider.value() / 100.0
        self.gamma = self.gamma_slider.value() / 10000.0
        if self.crystal is not None:
            self.crystal.alpha = self.alpha
            self.crystal.beta = self.beta
            self.crystal.gamma = self.gamma

    def apply_grid_size(self):
        size = self.size_spin.value()
        if size == self.grid_size:
            return
        self.grid_size = size
        self.reset()
        self.lbl_status.setText(f"网格已改为 {size}×{size}")

    def reset(self):
        self.timer.stop()
        self.run_btn.setText("开始")
        self.crystal = SnowCrystal(self.grid_size, self.alpha, self.beta, self.gamma)
        self.canvas.set_crystal(self.crystal)
        self.update_stats()
        self.lbl_status.setText("已重置")

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已暂停")
        else:
            self.timer.start(33)
            self.run_btn.setText("暂停")
            self.lbl_status.setText("生长中…")

    def single_step(self):
        self.crystal.step(max(1, self.speed))
        self.canvas.invalidate()
        self.update_stats()

    def tick(self):
        self.crystal.step(self.speed)
        self.canvas.invalidate()
        self.update_stats()
        # 到不了 0.95 的参数也不能无限长：和缩略图一样给 12000 步上限
        if self.auto_stop and self.crystal.extent > 0.95:
            self.timer.stop()
            self.run_btn.setText("开始")
            self.lbl_status.setText("已长满，自动停止")

    def on_speed(self, value):
        self.speed = value

    def toggle_vapor(self, checked):
        self.canvas.show_vapor = bool(checked)
        self.canvas.invalidate()

    def update_stats(self):
        self.lbl_ice.setText(f"冰格 {self.crystal.ice_count:,}")
        self.lbl_steps.setText(f"步数 {self.crystal.steps:,}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    raise SystemExit(run_app(SnowApp))
