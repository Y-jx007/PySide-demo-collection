from custom_import import *

ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=False, offline_cache=True)

# ========== 网格 ==========
nx, ny = 800, 250
cylinder_cx = int(nx * 0.2)
cylinder_cy = ny // 2
default_re = 200
default_u0 = 0.1
default_D = 20
default_nu = default_u0 * default_D / default_re

# ========== D2Q9 常量 ==========
e = ti.Matrix([
    [ 0,  0],
    [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
    [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]
])
w = ti.Vector([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opposite = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])

# ========== MRT 变换矩阵 ==========
M_np = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1],
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1],
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1]
], dtype=np.float32)
M_inv_np = np.linalg.inv(M_np)
M = ti.Matrix(M_np.tolist())
M_inv = ti.Matrix(M_inv_np.tolist())

# ========== 物理 / 松弛参数 ==========
s_vec = ti.Vector.field(9, ti.f32, shape=())
s1 = ti.field(ti.f32, shape=())
s2 = ti.field(ti.f32, shape=())
s4 = ti.field(ti.f32, shape=())
s6 = ti.field(ti.f32, shape=())

u0  = ti.field(ti.f32, shape=())
D   = ti.field(ti.f32, shape=())
Re  = ti.field(ti.f32, shape=())
nu  = ti.field(ti.f32, shape=())
cs  = ti.field(ti.f32, shape=())

bc_type  = ti.field(ti.i32, shape=4)
bc_value = ti.Vector.field(2, ti.f32, shape=4)

vort_vmax  = ti.field(ti.f32, shape=())
speed_vmax = ti.field(ti.f32, shape=())

rho   = ti.field(ti.f32, shape=(nx, ny))
vel   = ti.Vector.field(2, ti.f32, shape=(nx, ny))
mask  = ti.field(ti.f32, shape=(nx, ny))

f_old = ti.Vector.field(9, ti.f32, shape=(nx, ny))
f_new = ti.Vector.field(9, ti.f32, shape=(nx, ny))

vort  = ti.field(ti.f32, shape=(nx, ny))
speed = ti.field(ti.f32, shape=(nx, ny))

output_vort  = ti.Vector.field(4, ti.f32, shape=(nx, ny))
output_speed = ti.Vector.field(4, ti.f32, shape=(nx, ny))

lut_size = 256
vort_lut  = ti.Vector.field(3, ti.f32, shape=lut_size)
speed_lut = ti.Vector.field(3, ti.f32, shape=lut_size)

# ========== 初始化与更新 ==========
@ti.kernel
def init_constants():
    bc_type[0] = 0; bc_type[1] = 0; bc_type[2] = 1; bc_type[3] = 0
    bc_value[0] = ti.Vector([u0[None], 0.0])
    bc_value[1] = ti.Vector([0.0, 0.0])
    bc_value[2] = ti.Vector([0.0, 0.0])
    bc_value[3] = ti.Vector([0.0, 0.0])

@ti.kernel
def update_relaxation():
    nu_val = u0[None] * D[None] / Re[None]
    nu[None] = nu_val
    tau = 3.0 * nu_val + 0.5
    if tau < 0.5001: tau = 0.5001
    inv_tau = 1.0 / tau

    s_vec[None][0] = 0.0
    s_vec[None][1] = s1[None]
    s_vec[None][2] = s2[None]
    s_vec[None][3] = 0.0
    s_vec[None][4] = s4[None]
    s_vec[None][5] = 0.0
    s_vec[None][6] = s6[None]
    s_vec[None][7] = inv_tau
    s_vec[None][8] = inv_tau

@ti.func
def f_eq(i, j):
    eu = e @ vel[i, j]
    uv = vel[i, j].dot(vel[i, j])
    return w * rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

@ti.kernel
def init_fields():
    for i, j in ti.ndrange(nx, ny):
        rho[i, j] = 1.0
        vel[i, j] = ti.Vector([u0[None], 0.0])
        f_old[i, j] = f_eq(i, j)
        f_new[i, j] = f_old[i, j]
        mask[i, j] = 0.0
    disturb_start = int(nx * 0.15); disturb_end = int(nx * 0.19)
    for i, j in ti.ndrange((disturb_start, disturb_end), (0, ny)):
        if mask[i, j] == 0.0:
            vel[i, j] += ti.Vector([0.0, (ti.random() - 0.5) * 0.002])
            f_old[i, j] = f_eq(i, j)
            f_new[i, j] = f_old[i, j]

@ti.kernel
def update_mask_from_array(mask_arr: ti.types.ndarray()):
    for i, j in ti.ndrange(nx, ny):
        mask[i, j] = mask_arr[i, j]

@ti.kernel
def reset_solid_points():
    for i, j in ti.ndrange(nx, ny):
        if mask[i, j] == 1.0:
            rho[i, j] = 1.0
            vel[i, j] = ti.Vector([0.0, 0.0])
            f_tmp = f_eq(i, j)
            f_old[i, j] = f_tmp
            f_new[i, j] = f_tmp

# ========== 边界条件 ==========
@ti.kernel
def zouhe_inlet():
    u_in = u0[None]
    for j in range(1, ny-1):
        v_pert = (ti.random() - 0.5) * 0.01 * u_in
        ux_target = u_in
        uy_target = v_pert

        f0 = f_old[1, j][0]
        f2 = f_old[1, j][2]
        f3 = f_old[1, j][3]
        f4 = f_old[1, j][4]
        f6 = f_old[1, j][6]
        f7 = f_old[1, j][7]
        rho_0 = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - ux_target)
        f_old[0, j][1] = f3 + (2.0/3.0)*rho_0*ux_target
        f_old[0, j][5] = f7 - 0.5*(f2 - f4) + (1.0/6.0)*rho_0*ux_target + 0.5*rho_0*uy_target
        f_old[0, j][8] = f6 + 0.5*(f2 - f4) + (1.0/6.0)*rho_0*ux_target - 0.5*rho_0*uy_target
        vel[0, j] = ti.Vector([ux_target, uy_target])
        rho[0, j] = rho_0

@ti.kernel
def apply_corner_bc():
    # 左下角
    rho[0, 0] = rho[1, 1]
    vel[0, 0] = ti.Vector([0.0, 0.0])
    f_old[0, 0] = f_eq(0, 0)
    # 左上角
    rho[0, ny-1] = rho[1, ny-2]
    vel[0, ny-1] = ti.Vector([0.0, 0.0])
    f_old[0, ny-1] = f_eq(0, ny-1)
    # 右下角
    rho[nx-1, 0] = rho[nx-2, 1]
    vel[nx-1, 0] = ti.Vector([0.0, 0.0])
    f_old[nx-1, 0] = f_eq(nx-1, 0)
    # 右上角
    rho[nx-1, ny-1] = rho[nx-2, ny-2]
    vel[nx-1, ny-1] = ti.Vector([0.0, 0.0])
    f_old[nx-1, ny-1] = f_eq(nx-1, ny-1)

@ti.kernel
def apply_bc():
    for i in range(1, nx-1):
        apply_bc_core(1, 1, i, ny-1, i, ny-2)
    for i in range(1, nx-1):
        apply_bc_core(1, 3, i, 0, i, 1)
    for j in range(1, ny-1):
        apply_bc_core(1, 2, nx-1, j, nx-2, j)

@ti.func
def apply_bc_core(outer, dr, ibc, jbc, inb, jnb):
    if outer == 1:
        if bc_type[dr] == 0:
            vel[ibc, jbc] = bc_value[dr]
        else:
            vel[ibc, jbc] = vel[inb, jnb]
    rho[ibc, jbc] = rho[inb, jnb]
    f_old[ibc, jbc] = f_eq(ibc, jbc) - f_eq(inb, jnb) + f_old[inb, jnb]

# ========== 碰撞与流 ==========
@ti.kernel
def stream_and_collide():
    cs2 = 1.0 / 3.0
    for i, j in ti.ndrange((1, nx-1), (1, ny-1)):
        if mask[i, j] == 1.0:
            continue

        f = ti.Vector.zero(ti.f32, 9)
        for k in ti.static(range(9)):
            ip = i - e[k, 0]
            jp = j - e[k, 1]
            if mask[ip, jp] == 1.0:
                k_op = opposite[k]
                f[k] = f_old[i, j][k_op]
            else:
                f[k] = f_old[ip, jp][k]

        rho_tmp = 0.0; ux_tmp = 0.0; uy_tmp = 0.0
        for k in ti.static(range(9)):
            rho_tmp += f[k]
            ux_tmp += e[k, 0] * f[k]
            uy_tmp += e[k, 1] * f[k]
        inv_rho = 1.0 / rho_tmp
        ux_tmp *= inv_rho
        uy_tmp *= inv_rho
        uv = ux_tmp*ux_tmp + uy_tmp*uy_tmp

        tau_m = 3.0 * nu[None] + 0.5
        if tau_m < 0.5001: tau_m = 0.5001
        inv_2rho_cs2_tau = 1.0 / (2.0 * rho_tmp * cs2 * tau_m)

        Pi_xx = 0.0; Pi_yy = 0.0; Pi_xy = 0.0
        for k in ti.static(range(9)):
            eu = e[k,0]*ux_tmp + e[k,1]*uy_tmp
            feq = w[k] * rho_tmp * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*uv)
            neq = f[k] - feq
            Pi_xx += e[k,0]*e[k,0] * neq
            Pi_yy += e[k,1]*e[k,1] * neq
            Pi_xy += e[k,0]*e[k,1] * neq

        S_xx = -Pi_xx * inv_2rho_cs2_tau
        S_yy = -Pi_yy * inv_2rho_cs2_tau
        S_xy = -Pi_xy * inv_2rho_cs2_tau
        S_mag = ti.sqrt(2.0*(S_xx*S_xx + S_yy*S_yy + 2.0*S_xy*S_xy))

        nu_t = cs[None]*cs[None] * S_mag
        nu_total = nu[None] + nu_t
        tau_total = 3.0 * nu_total + 0.5
        if tau_total < 0.5001: tau_total = 0.5001

        inv_2rho_cs2_tau_eff = 1.0 / (2.0 * rho_tmp * cs2 * tau_total)
        S_xx = -Pi_xx * inv_2rho_cs2_tau_eff
        S_yy = -Pi_yy * inv_2rho_cs2_tau_eff
        S_xy = -Pi_xy * inv_2rho_cs2_tau_eff
        S_mag = ti.sqrt(2.0*(S_xx*S_xx + S_yy*S_yy + 2.0*S_xy*S_xy))

        nu_t = cs[None]*cs[None] * S_mag
        nu_total = nu[None] + nu_t
        tau_total = 3.0 * nu_total + 0.5
        if tau_total < 0.5001: tau_total = 0.5001
        inv_tau_total = 1.0 / tau_total

        m = M @ f
        m_eq = ti.Vector.zero(ti.f32, 9)
        m_eq[0] = rho_tmp
        m_eq[1] = -2.0*rho_tmp + 3.0*rho_tmp*uv
        m_eq[2] =  rho_tmp - 3.0*rho_tmp*uv
        m_eq[3] =  rho_tmp * ux_tmp
        m_eq[4] = -rho_tmp * ux_tmp
        m_eq[5] =  rho_tmp * uy_tmp
        m_eq[6] = -rho_tmp * uy_tmp
        m_eq[7] =  rho_tmp * (ux_tmp*ux_tmp - uy_tmp*uy_tmp)
        m_eq[8] =  rho_tmp * ux_tmp * uy_tmp

        s = s_vec[None]
        m_post = ti.Vector.zero(ti.f32, 9)
        for k in ti.static(range(9)):
            sk = s[k]
            if k == 7 or k == 8:
                sk = inv_tau_total
            m_post[k] = m[k] - sk * (m[k] - m_eq[k])

        f_post = M_inv @ m_post
        for k in ti.static(range(9)):
            f_new[i, j][k] = f_post[k]

@ti.kernel
def update_macro():
    for i, j in ti.ndrange((1, nx-1), (1, ny-1)):
        if mask[i, j] == 1.0:
            continue
        rho[i, j] = 0.0
        vel[i, j] = ti.Vector([0.0, 0.0])
        for k in ti.static(range(9)):
            f_old[i, j][k] = f_new[i, j][k]
            rho[i, j] += f_new[i, j][k]
            vel[i, j] += ti.Vector([e[k, 0], e[k, 1]]) * f_new[i, j][k]
        vel[i, j] /= rho[i, j]

# ========== 后处理 ==========
@ti.kernel
def calc_vorticity():
    for i, j in ti.ndrange((1, nx-1), (1, ny-1)):
        if mask[i, j] == 1.0 or mask[i+1, j]==1.0 or mask[i-1, j]==1.0 or mask[i, j+1]==1.0 or mask[i, j-1]==1.0:
            vort[i, j] = 0.0
        else:
            du_dy = (vel[i, j+1][0] - vel[i, j-1][0]) * 0.5
            dv_dx = (vel[i+1, j][1] - vel[i-1, j][1]) * 0.5
            vort[i, j] = du_dy - dv_dx

@ti.kernel
def calc_speed():
    for i, j in ti.ndrange(nx, ny):
        speed[i, j] = ti.sqrt(vel[i, j][0]**2 + vel[i, j][1]**2)

@ti.kernel
def init_color_luts():
    for i in range(lut_size):
        t = i/(lut_size-1)
        if t<0.5: vort_lut[i] = ti.Vector([t*2, t*2, 1.0])
        else:     vort_lut[i] = ti.Vector([1.0, 2*(1-t), 2*(1-t)])
    for i in range(lut_size):
        t = i/(lut_size-1)
        if t<0.33:     speed_lut[i] = ti.Vector([0.0, 0.0, t*3])
        elif t<0.66:   speed_lut[i] = ti.Vector([(t-0.33)*3, 0.0, 1.0])
        else:          speed_lut[i] = ti.Vector([1.0, (t-0.66)*3, 1.0-(t-0.66)*3])

@ti.kernel
def visualize_vort():
    vmax = vort_vmax[None]
    for i, j in ti.ndrange(nx, ny):
        val = max(-vmax, min(vmax, vort[i, j]))
        t = (val+vmax)/(2.0*vmax)
        idx = ti.cast(t*(lut_size-1), ti.i32)
        output_vort[i, j] = ti.Vector([vort_lut[idx][0], vort_lut[idx][1], vort_lut[idx][2], 1.0])

@ti.kernel
def visualize_speed():
    vmax = speed_vmax[None]
    for i, j in ti.ndrange(nx, ny):
        val = min(speed[i, j], vmax)
        t = val/vmax
        idx = ti.cast(t*(lut_size-1), ti.i32)
        output_speed[i, j] = ti.Vector([speed_lut[idx][0], speed_lut[idx][1], speed_lut[idx][2], 1.0])

# ========== GUI 部分 ==========
class GLDisplayWidget(QOpenGLWidget):
    def __init__(self, output_field, parent=None):
        super().__init__(parent)
        self.output_field = output_field
        self.texture_id = None
        self.setMinimumSize(400, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def paintGL(self):
        if self.texture_id is None: return
        dpr = self.devicePixelRatio()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)
        if w <= 0 or h <= 0: return
        tex_ratio = nx / ny
        if w/h > tex_ratio:
            draw_h = h; draw_w = int(h * tex_ratio)
            ox = (w - draw_w)//2; oy = 0
        else:
            draw_w = w; draw_h = int(w / tex_ratio)
            ox = 0; oy = (h - draw_h)//2

        data = self.output_field.to_numpy()
        data = np.ascontiguousarray(data)
        data = np.clip(data, 0, 1) * 255
        data = data.astype(np.uint8).transpose(1,0,2).copy()

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glViewport(0, 0, w, h)
        glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2i(ox, oy)
        glTexCoord2f(1,0); glVertex2i(ox+draw_w, oy)
        glTexCoord2f(1,1); glVertex2i(ox+draw_w, oy+draw_h)
        glTexCoord2f(0,1); glVertex2i(ox, oy+draw_h)
        glEnd()

    def resizeGL(self, w, h): pass

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.simulation_running = True
        self.barrier_thickness = 2
        self.single_slit_width = 10.0
        self.double_slit_width = 10.0
        self.double_slit_sep = 20.0
        self._updating_re_nu = False
        self._init_ui()

    def _init_ui(self):
        main_h = QHBoxLayout(self)
        left = QVBoxLayout()

        # 物理参数
        phys = QGroupBox("物理参数")
        g = QGridLayout(phys)
        self.re_slider = QSlider(Qt.Horizontal); self.re_slider.setRange(10,500); self.re_slider.setValue(default_re)
        self.re_label = QLabel(str(default_re))
        g.addWidget(QLabel("Re"),0,0); g.addWidget(self.re_slider,0,1); g.addWidget(self.re_label,0,2)
        self.re_slider.valueChanged.connect(self._on_re_changed)

        self.nu_slider = QSlider(Qt.Horizontal); self.nu_slider.setRange(1,500); self.nu_slider.setValue(int(default_nu*2000))
        self.nu_label = QLabel(f"{default_nu:.4f}")
        g.addWidget(QLabel("ν"),1,0); g.addWidget(self.nu_slider,1,1); g.addWidget(self.nu_label,1,2)
        self.nu_slider.valueChanged.connect(self._on_nu_changed)

        self.u_slider = QSlider(Qt.Horizontal); self.u_slider.setRange(1,30); self.u_slider.setValue(int(default_u0*100))
        self.u_label = QLabel(f"{default_u0:.2f}")
        g.addWidget(QLabel("u0"),2,0); g.addWidget(self.u_slider,2,1); g.addWidget(self.u_label,2,2)
        self.u_slider.valueChanged.connect(self._on_u_changed)

        self.tau_label = QLabel("0.5001")
        g.addWidget(QLabel("τ"),3,0); g.addWidget(self.tau_label,3,1,1,2)

        self.cs_slider = QSlider(Qt.Horizontal); self.cs_slider.setRange(0,20); self.cs_slider.setValue(10)
        self.cs_label = QLabel("0.10")
        g.addWidget(QLabel("Smag. Cs"),4,0); g.addWidget(self.cs_slider,4,1); g.addWidget(self.cs_label,4,2)
        self.cs_slider.valueChanged.connect(lambda v: (cs.__setitem__(None, v/100.0), self.cs_label.setText(f"{v/100.0:.2f}")))

        self.steps_slider = QSlider(Qt.Horizontal); self.steps_slider.setRange(1,20); self.steps_slider.setValue(10)
        self.steps_label = QLabel("10")
        g.addWidget(QLabel("步数"),5,0); g.addWidget(self.steps_slider,5,1); g.addWidget(self.steps_label,5,2)
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        left.addWidget(phys)

        # MRT 松弛参数
        mrt_group = QGroupBox("MRT 松弛参数")
        mg = QGridLayout(mrt_group)
        self.s1_slider = QSlider(Qt.Horizontal); self.s1_slider.setRange(10,200); self.s1_slider.setValue(119)
        self.s1_label = QLabel("1.19")
        mg.addWidget(QLabel("能量 s1"),0,0); mg.addWidget(self.s1_slider,0,1); mg.addWidget(self.s1_label,0,2)
        self.s1_slider.valueChanged.connect(lambda v: (s1.__setitem__(None, v/100.0), self.s1_label.setText(f"{v/100.0:.2f}")))

        self.s2_slider = QSlider(Qt.Horizontal); self.s2_slider.setRange(10,200); self.s2_slider.setValue(140)
        self.s2_label = QLabel("1.40")
        mg.addWidget(QLabel("能量平方 s2"),1,0); mg.addWidget(self.s2_slider,1,1); mg.addWidget(self.s2_label,1,2)
        self.s2_slider.valueChanged.connect(lambda v: (s2.__setitem__(None, v/100.0), self.s2_label.setText(f"{v/100.0:.2f}")))

        self.s4_slider = QSlider(Qt.Horizontal); self.s4_slider.setRange(10,200); self.s4_slider.setValue(120)
        self.s4_label = QLabel("1.20")
        mg.addWidget(QLabel("x动量通量 s4"),2,0); mg.addWidget(self.s4_slider,2,1); mg.addWidget(self.s4_label,2,2)
        self.s4_slider.valueChanged.connect(lambda v: (s4.__setitem__(None, v/100.0), self.s4_label.setText(f"{v/100.0:.2f}")))

        self.s6_slider = QSlider(Qt.Horizontal); self.s6_slider.setRange(10,200); self.s6_slider.setValue(120)
        self.s6_label = QLabel("1.20")
        mg.addWidget(QLabel("y动量通量 s6"),3,0); mg.addWidget(self.s6_slider,3,1); mg.addWidget(self.s6_label,3,2)
        self.s6_slider.valueChanged.connect(lambda v: (s6.__setitem__(None, v/100.0), self.s6_label.setText(f"{v/100.0:.2f}")))
        left.addWidget(mrt_group)

        # 障碍物选择
        obs = QGroupBox("障碍物")
        ol = QVBoxLayout(obs)
        self.obs_combo = QComboBox(); self.obs_combo.addItems(["圆柱","方柱","单缝","双缝"])
        self.obs_combo.currentIndexChanged.connect(self._on_obs_preset_changed)
        ol.addWidget(self.obs_combo)
        left.addWidget(obs)
        main_h.addLayout(left)

        # 右侧几何、可视化控件
        right = QVBoxLayout()
        geo = QGroupBox("几何尺寸")
        gg = QGridLayout(geo)
        self.thick_slider = QSlider(Qt.Horizontal); self.thick_slider.setRange(1,10); self.thick_slider.setValue(2)
        self.thick_label = QLabel("2")
        gg.addWidget(QLabel("厚度"),0,0); gg.addWidget(self.thick_slider,0,1); gg.addWidget(self.thick_label,0,2)
        self.thick_slider.valueChanged.connect(self._on_thickness_changed)
        self.ss_slider = QSlider(Qt.Horizontal); self.ss_slider.setRange(2,100); self.ss_slider.setValue(10)
        self.ss_label = QLabel("10")
        gg.addWidget(QLabel("单缝宽"),1,0); gg.addWidget(self.ss_slider,1,1); gg.addWidget(self.ss_label,1,2)
        self.ss_slider.valueChanged.connect(self._on_ss_width_changed)
        self.ds_w_slider = QSlider(Qt.Horizontal); self.ds_w_slider.setRange(2,50); self.ds_w_slider.setValue(10)
        self.ds_w_label = QLabel("10")
        gg.addWidget(QLabel("双缝宽"),2,0); gg.addWidget(self.ds_w_slider,2,1); gg.addWidget(self.ds_w_label,2,2)
        self.ds_w_slider.valueChanged.connect(self._on_ds_width_changed)
        self.ds_s_slider = QSlider(Qt.Horizontal); self.ds_s_slider.setRange(4,150); self.ds_s_slider.setValue(20)
        self.ds_s_label = QLabel("20")
        gg.addWidget(QLabel("双缝距"),3,0); gg.addWidget(self.ds_s_slider,3,1); gg.addWidget(self.ds_s_label,3,2)
        self.ds_s_slider.valueChanged.connect(self._on_ds_sep_changed)
        self.d_slider = QSlider(Qt.Horizontal); self.d_slider.setRange(10,50); self.d_slider.setValue(default_D)
        self.d_label = QLabel(str(default_D))
        gg.addWidget(QLabel("圆方尺寸"),4,0); gg.addWidget(self.d_slider,4,1); gg.addWidget(self.d_label,4,2)
        self.d_slider.valueChanged.connect(lambda v: (self.d_label.setText(str(v)), self._on_D_changed(v)))
        right.addWidget(geo)

        # 可视化范围 (修正滑块归属)
        vis = QGroupBox("可视化范围")
        vg = QGridLayout(vis)
        self.vort_vmax_slider = QSlider(Qt.Horizontal); self.vort_vmax_slider.setRange(1,100); self.vort_vmax_slider.setValue(20)
        self.vort_vmax_label = QLabel("0.020")
        vg.addWidget(QLabel("涡量"),0,0); vg.addWidget(self.vort_vmax_slider,0,1); vg.addWidget(self.vort_vmax_label,0,2)
        self.vort_vmax_slider.valueChanged.connect(lambda v: (vort_vmax.__setitem__(None, v/1000.0), self.vort_vmax_label.setText(f"{v/1000.0:.3f}")))
        self.speed_vmax_slider = QSlider(Qt.Horizontal); self.speed_vmax_slider.setRange(1,300); self.speed_vmax_slider.setValue(150)
        self.speed_vmax_label = QLabel("0.150")
        vg.addWidget(QLabel("速度"),1,0); vg.addWidget(self.speed_vmax_slider,1,1); vg.addWidget(self.speed_vmax_label,1,2)
        self.speed_vmax_slider.valueChanged.connect(lambda v: (speed_vmax.__setitem__(None, v/1000.0), self.speed_vmax_label.setText(f"{v/1000.0:.3f}")))
        right.addWidget(vis)

        run = QHBoxLayout()
        self.pause_btn = QPushButton("暂停"); self.pause_btn.setCheckable(True); self.pause_btn.setChecked(True)
        self.reset_btn = QPushButton("重置流场")
        run.addWidget(self.pause_btn); run.addWidget(self.reset_btn)
        right.addLayout(run); right.addStretch()
        main_h.addLayout(right)

        self.setMaximumWidth(400); self.setMinimumWidth(340)
        self.pause_btn.clicked.connect(self.toggle_sim)
        self.reset_btn.clicked.connect(self.reset_sim)

        # 初始化参数
        s1[None] = 1.19; s2[None] = 1.4; s4[None] = 1.2; s6[None] = 1.2
        vort_vmax[None] = 0.02
        speed_vmax[None] = 0.15
        cs[None] = 0.1
        self._update_tau_display()

    def _sync_re_nu(self, changed):
        if self._updating_re_nu: return
        self._updating_re_nu = True
        u = self.u_slider.value() / 100.0
        D_val = float(self.d_slider.value())
        if changed == 're':
            re = self.re_slider.value()
            nu_val = u * D_val / re
            nu_slider_val = int(nu_val * 2000)
            nu_slider_val = max(1, min(500, nu_slider_val))
            nu_val = nu_slider_val / 2000.0
            new_re = int(u * D_val / nu_val)
            new_re = max(10, min(500, new_re))
            self.re_slider.setValue(new_re); self.re_label.setText(str(new_re))
            self.nu_slider.setValue(nu_slider_val); self.nu_label.setText(f"{nu_val:.4f}")
        elif changed == 'nu':
            nu_val = self.nu_slider.value() / 2000.0
            re = int(u * D_val / nu_val)
            re = max(10, min(500, re))
            self.re_slider.setValue(re); self.re_label.setText(str(re))
            self.nu_label.setText(f"{nu_val:.4f}")
        elif changed == 'u':
            re = self.re_slider.value()
            nu_val = u * D_val / re
            nu_slider_val = int(nu_val * 2000)
            nu_slider_val = max(1, min(500, nu_slider_val))
            nu_val = nu_slider_val / 2000.0
            self.nu_slider.setValue(nu_slider_val); self.nu_label.setText(f"{nu_val:.4f}")
        elif changed == 'D':
            re = self.re_slider.value()
            nu_val = u * D_val / re
            nu_slider_val = int(nu_val * 2000)
            nu_slider_val = max(1, min(500, nu_slider_val))
            nu_val = nu_slider_val / 2000.0
            self.nu_slider.setValue(nu_slider_val); self.nu_label.setText(f"{nu_val:.4f}")
        self._updating_re_nu = False
        self._update_tau_display()

    def _on_re_changed(self, v): self.re_label.setText(str(v)); self._sync_re_nu('re')
    def _on_nu_changed(self, v): self.nu_label.setText(f"{v/2000.0:.4f}"); self._sync_re_nu('nu')
    def _on_u_changed(self, v): self.u_label.setText(f"{v/100.0:.2f}"); self._sync_re_nu('u')
    def _on_D_changed(self, v): self.d_label.setText(str(v)); self._sync_re_nu('D'); self._reapply_current_obstacle()

    def _update_tau_display(self):
        u = self.u_slider.value()/100.0; D_val = float(self.d_slider.value()); re = self.re_slider.value()
        nu_val = u * D_val / re; tau = 3.0*nu_val + 0.5
        if tau < 0.5001: tau = 0.5001
        self.tau_label.setText(f"{tau:.4f}")

    def _on_thickness_changed(self, v): self.barrier_thickness=v; self.thick_label.setText(str(v)); self._reapply_current_obstacle()
    def _on_ss_width_changed(self, v): self.single_slit_width=float(v); self.ss_label.setText(str(v)); self._reapply_current_obstacle()
    def _on_ds_width_changed(self, v): self.double_slit_width=float(v); self.ds_w_label.setText(str(v)); self._reapply_current_obstacle()
    def _on_ds_sep_changed(self, v): self.double_slit_sep=float(v); self.ds_s_label.setText(str(v)); self._reapply_current_obstacle()

    def _reapply_current_obstacle(self):
        idx = self.obs_combo.currentIndex()
        if idx==0: self.apply_cylindrical_obstacle()
        elif idx==1: self.apply_square_obstacle()
        elif idx==2: self.apply_single_slit()
        elif idx==3: self.apply_double_slit()

    def _on_obs_preset_changed(self, idx):
        self._reapply_current_obstacle()

    def apply_cylindrical_obstacle(self):
        D_val = float(self.d_slider.value()); mask_arr = np.zeros((nx, ny), dtype=np.float32)
        cx, cy = cylinder_cx, cylinder_cy
        for i in range(nx):
            for j in range(ny):
                if (i-cx)**2 + (j-cy)**2 < (D_val/2)**2: mask_arr[i,j]=1.0
        update_mask_from_array(mask_arr)
        reset_solid_points()

    def apply_square_obstacle(self):
        D_val = float(self.d_slider.value()); mask_arr = np.zeros((nx, ny), dtype=np.float32)
        cx, cy = cylinder_cx, cylinder_cy; half = D_val/2
        for i in range(nx):
            for j in range(ny):
                if abs(i-cx)<half and abs(j-cy)<half: mask_arr[i,j]=1.0
        update_mask_from_array(mask_arr)
        reset_solid_points()

    def apply_single_slit(self):
        gap = max(2, int(self.single_slit_width)); thick = max(1, self.barrier_thickness)
        cx, cy = cylinder_cx, cylinder_cy; mask_arr = np.zeros((nx, ny), dtype=np.float32)
        tb = cy - gap//2; bt = cy + gap//2
        for i in range(nx):
            for j in range(ny):
                if abs(i-cx)<thick and (j<tb or j>bt): mask_arr[i,j]=1.0
        update_mask_from_array(mask_arr)
        reset_solid_points()

    def apply_double_slit(self):
        gap = max(2, int(self.double_slit_width)); sep = max(gap+2, int(self.double_slit_sep))
        thick = max(1, self.barrier_thickness); cx, cy = cylinder_cx, cylinder_cy
        mask_arr = np.zeros((nx, ny), dtype=np.float32)
        uc = cy - sep//2; lc = cy + sep//2
        for i in range(nx):
            for j in range(ny):
                if abs(i-cx)<thick:
                    in_u = abs(j-uc)<gap//2; in_l = abs(j-lc)<gap//2
                    if not (in_u or in_l): mask_arr[i,j]=1.0
        update_mask_from_array(mask_arr)
        reset_solid_points()

    def toggle_sim(self):
        self.simulation_running = not self.simulation_running
        self.pause_btn.setText("暂停" if self.simulation_running else "开始")

    def reset_sim(self):
        self.apply_params(); update_relaxation(); init_fields()
        self._reapply_current_obstacle()
        calc_vorticity(); calc_speed()

    def apply_params(self):
        Re[None] = float(self.re_slider.value())
        u0[None] = self.u_slider.value()/100.0
        D[None] = float(self.d_slider.value())
        init_constants()

    def get_steps(self):
        return self.steps_slider.value()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卡门涡街 MRT + 非平衡 Smagorinsky + Zou‑He 入口 (修正版)")
        self.setGeometry(100,100,1280,560)
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        self.control = ControlPanel()
        right = QVBoxLayout()
        self.vort_display = GLDisplayWidget(output_vort)
        self.speed_display = GLDisplayWidget(output_speed)
        right.addWidget(self.vort_display); right.addWidget(self.speed_display)
        main_layout.addWidget(self.control,0); main_layout.addLayout(right,1)

        self.control.apply_params(); D[None]=default_D; update_relaxation(); init_fields()
        self.control.apply_cylindrical_obstacle()
        init_color_luts(); calc_vorticity(); calc_speed(); visualize_vort(); visualize_speed()
        self.vort_display.update(); self.speed_display.update()

        self.timer = QTimer(); self.timer.timeout.connect(self._update_sim); self.timer.start(0)

    def _update_sim(self):
        if self.control.simulation_running:
            self.control.apply_params(); update_relaxation()
            for _ in range(self.control.get_steps()):
                stream_and_collide()
                update_macro()
                apply_bc()
                zouhe_inlet()
                apply_corner_bc()
            calc_vorticity(); calc_speed(); visualize_vort(); visualize_speed()
            self.vort_display.update(); self.speed_display.update()

    def closeEvent(self, event):
        self.timer.stop(); ti.reset(); event.accept()

def main():
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()