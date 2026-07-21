from custom_import import *

# ==================== Taichi 初始化 ====================
ti.init(arch=ti.gpu)

W, H = 640, 640
FOCAL_LENGTH = 3.0
BOUNDING_RADIUS = 3.5               # 扩大包围球
DEFAULT_MAX_ITER = 250
DEFAULT_SURFACE_PRECISION = 0.00025
DEFAULT_SAMPLES = 2
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
def shade(t_norm):
    rock = ti.Vector([140, 140, 140]) / 255.0
    dk_red = ti.Vector([160, 30, 10]) / 255.0
    orange = ti.Vector([255, 100, 0]) / 255.0
    gold = ti.Vector([255, 220, 0]) / 255.0
    t = ti.pow(t_norm, 1.5)
    col = ti.Vector([0.0, 0.0, 0.0])
    if t < 0.5:
        s = smoothstep(0.0, 0.5, t)
        col = rock * (1 - s) + dk_red * s
    elif t < 0.8:
        s = smoothstep(0.5, 0.8, t)
        col = dk_red * (1 - s) + orange * s
    else:
        s = smoothstep(0.8, 1.0, t)
        col = orange * (1 - s) + gold * s
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
                col += ti.Vector([0.2, 0.2, 0.22])
            else:
                pos = ro + rd*t
                nor = normal(pos, c_val, plane, max_iter, power, prec, mode)
                light = ti.Vector([0.5, 0.8, 0.3]).normalized()
                diff = ti.max(nor.dot(light), 0.0); amb = 0.4
                t_norm = ti.log(tn.y+1) / ti.log(ti.cast(max_iter,ti.f32)+1)
                col += shade(t_norm) * (amb + diff*0.6)
        curr[i,j] = col / ti.cast(samples, ti.f32)
        accum[i,j] += curr[i,j]
    acc_cnt[None] += 1

@ti.kernel
def reset():
    for i, j in accum: accum[i,j] = ti.Vector([0.0,0.0,0.0])
    acc_cnt[None] = 0

@ti.kernel
def tonemap():
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
    def __init__(self, cam):
        super().__init__()
        self.setMinimumSize(W, H)
        self.cam = cam; self.last_pos = None
        self.cx, self.cy, self.cz, self.cw = 0.0, 0.0, 0.675, 0.0
        self.max_iter = DEFAULT_MAX_ITER; self.samples = DEFAULT_SAMPLES
        self.prec = DEFAULT_SURFACE_PRECISION; self.power = DEFAULT_POWER
        self.mode = 0                      # 0=Julia, 1=Mandelbrot
        self.need_reset = True; self.tex = None

    def initializeGL(self):
        self.tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glDisable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT); self.cam._update_vectors()
        m = self.cam.model_matrix()
        ro = m.map(self.cam.position - self.cam.target) + self.cam.target
        rd = m.mapVector(self.cam.front); ri = m.mapVector(self.cam.right); up = m.mapVector(self.cam.up)
        ro_ti = ti.Vector([ro.x(), ro.y(), ro.z()]); ri_ti = ti.Vector([ri.x(), ri.y(), ri.z()])
        up_ti = ti.Vector([up.x(), up.y(), up.z()]); rd_ti = ti.Vector([rd.x(), rd.y(), rd.z()])
        c_ti = ti.Vector([self.cx, self.cy, self.cz, self.cw]); pl_ti = section_plane[None]
        if self.need_reset: reset(); self.need_reset = False
        render(ro_ti, ri_ti, up_ti, rd_ti, c_ti, pl_ti, self.max_iter, self.power, self.samples, self.prec, self.mode)
        tonemap(); ti.sync()
        arr = display.to_numpy().transpose(1,0,2)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, arr.tobytes())
        ratio = self.devicePixelRatioF(); w, h = int(self.width()*ratio), int(self.height()*ratio)
        glViewport(0,0,w,h); glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glEnable(GL_TEXTURE_2D); glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0); glTexCoord2f(1,0); glVertex2f(w,0)
        glTexCoord2f(1,1); glVertex2f(w,h); glTexCoord2f(0,1); glVertex2f(0,h)
        glEnd(); glDisable(GL_TEXTURE_2D); self.update()

    def resizeEvent(self, e):
        s = min(self.width(), self.height()); self.setFixedSize(s, s); super().resizeEvent(e)

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
            self.need_reset = True
        elif e.buttons() & Qt.MiddleButton: self.cam.pan(-dx*0.1, dy*0.1); self.need_reset = True
        self.last_pos = e.position()

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("四元数 Julia / Mandelbrot - 熔岩")
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
        g.addWidget(QLabel("迭代次数"),0,0); self.iter_s = QSlider(Qt.Horizontal); self.iter_s.setRange(50,500); self.iter_s.setValue(DEFAULT_MAX_ITER)
        self.iter_l = QLabel(str(DEFAULT_MAX_ITER))
        self.iter_s.valueChanged.connect(lambda v: (self.iter_l.setText(str(v)), self._reset()))
        g.addWidget(self.iter_s,0,1); g.addWidget(self.iter_l,0,2)

        g.addWidget(QLabel("精度"),1,0); self.prec_s = QSlider(Qt.Horizontal); self.prec_s.setRange(1,50); self.prec_s.setValue(int(DEFAULT_SURFACE_PRECISION*100000))
        self.prec_l = QLabel(f"{DEFAULT_SURFACE_PRECISION:.5f}")
        self.prec_s.valueChanged.connect(lambda v: (self.prec_l.setText(f"{v/100000:.5f}"), self._reset()))
        g.addWidget(self.prec_s,1,1); g.addWidget(self.prec_l,1,2)

        g.addWidget(QLabel("采样"),2,0); self.samp_s = QSlider(Qt.Horizontal); self.samp_s.setRange(1,8); self.samp_s.setValue(DEFAULT_SAMPLES)
        self.samp_l = QLabel(str(DEFAULT_SAMPLES))
        self.samp_s.valueChanged.connect(lambda v: (self.samp_l.setText(str(v)), self._reset()))
        g.addWidget(self.samp_s,2,1); g.addWidget(self.samp_l,2,2)
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
        self.dist_s = QSlider(Qt.Horizontal); self.dist_s.setRange(1,200); self.dist_s.setValue(36)
        self.dist_l = QLabel("3.6")
        self.dist_s.valueChanged.connect(lambda v: (self.dist_l.setText(f"{v/10:.1f}"), self._set_dist(v)))
        dh.addWidget(QLabel("距离:")); dh.addWidget(self.dist_s); dh.addWidget(self.dist_l)
        btn_reset_view = QPushButton("重置视角"); btn_reset_view.clicked.connect(self._reset_cam)
        dh.addWidget(btn_reset_view); dist_g.setLayout(dh); lay.addWidget(dist_g)
        lay.addStretch()

        self.gl = FractalWidget(self.cam)
        self.gl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        splitter.addWidget(panel); splitter.addWidget(self.gl)

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

    def _reset(self):
        self.gl.max_iter = self.iter_s.value()
        self.gl.prec = self.prec_s.value() / 100000
        self.gl.samples = self.samp_s.value()
        self.gl.power = self.power_spin.value()
        self.gl.need_reset = True

    def _set_dist(self, v):
        self.cam.distance = v / 10.0
        self.cam._update_vectors()
        self._reset()

    def _reset_cam(self):
        self.cam.reset()
        self.dist_s.setValue(36)
        self._reset()

    def closeEvent(self, e):
        ti.reset(); e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat(); fmt.setVersion(2,1); fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    w = MainWindow(); w.resize(1100, 750); w.show()
    sys.exit(app.exec())