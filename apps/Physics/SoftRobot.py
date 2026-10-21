from custom_import import *
from PySide6.QtCore import QPointF, QRectF, QThread, Signal
from PySide6.QtGui import QPolygonF

import numpy as np

# ==================== 可微 MPM 软体机器人 ====================
N_GRID = 48
MAX_STEPS = 300
DT = 2e-3
E = 10.0
MU = E
LA = E
GRAVITY = 5.0
ACT_OMEGA = 20.0
ACT_STRENGTH = 4.0
N_SIN = 4

real = ti.f32


class SoftRobot:
    def __init__(self, n_grid=N_GRID, steps=MAX_STEPS):
        ti.reset()
        ti.init(arch=ti.cpu, default_fp=real)

        self.n_grid = n_grid
        self.steps = steps
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.p_vol = 1.0

        # 场景：身体 + 四条腿，每条腿一个驱动器
        self.scene = []
        self.act_ids = []
        self._build_robot()
        self.n_particles = len(self.scene)
        self.n_actuators = 4

        self.x = ti.Vector.field(2, real)
        self.v = ti.Vector.field(2, real)
        self.C = ti.Matrix.field(2, 2, real)
        self.F = ti.Matrix.field(2, 2, real)
        self.grid_v_in = ti.Vector.field(2, real)
        self.grid_m_in = ti.field(real)
        self.grid_v_out = ti.Vector.field(2, real)
        self.actuator_id = ti.field(ti.i32)
        self.actuation = ti.field(real)
        self.loss = ti.field(real)
        self.x_avg = ti.Vector.field(2, real)
        self.weights = ti.field(real)
        self.bias = ti.field(real)
        self.weights_grad = ti.field(real)
        self.bias_grad = ti.field(real)

        ti.root.dense(ti.ij, (self.n_actuators, N_SIN)).place(self.weights, self.weights_grad)
        ti.root.dense(ti.i, self.n_actuators).place(self.bias, self.bias_grad)
        ti.root.dense(ti.ij, (steps, self.n_actuators)).place(self.actuation)
        ti.root.dense(ti.i, self.n_particles).place(self.actuator_id)
        ti.root.dense(ti.k, steps).dense(ti.l, self.n_particles).place(
            self.x, self.v, self.C, self.F)
        ti.root.dense(ti.ij, n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.place(self.loss, self.x_avg)
        ti.root.lazy_grad()

        self._init_state()
        self._compile()

    # ---------- 场景 ----------
    def _add_rect(self, x, y, w, h, act):
        w_count = max(1, int(w / self.dx) * 2)
        h_count = max(1, int(h / self.dx) * 2)
        rdx, rdy = w / w_count, h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.scene.append([x + (i + 0.5) * rdx, y + (j + 0.5) * rdy])
                self.act_ids.append(act)

    def _build_robot(self):
        ox, oy = 0.14, 0.04
        self._add_rect(ox + 0.00, oy + 0.08, 0.24, 0.08, -1)   # 身体
        self._add_rect(ox + 0.00, oy + 0.00, 0.04, 0.08, 0)    # 四条腿
        self._add_rect(ox + 0.04, oy + 0.00, 0.04, 0.08, 1)
        self._add_rect(ox + 0.16, oy + 0.00, 0.04, 0.08, 2)
        self._add_rect(ox + 0.20, oy + 0.00, 0.04, 0.08, 3)

    def _init_state(self):
        rng = np.random.default_rng(0)
        xs = np.array(self.scene, np.float32)
        for i in range(self.n_particles):
            self.x[0, i] = xs[i]
            self.F[0, i] = [[1.0, 0.0], [0.0, 1.0]]
            self.actuator_id[i] = self.act_ids[i]
        for i in range(self.n_actuators):
            for j in range(N_SIN):
                self.weights[i, j] = rng.normal() * 0.01
            self.bias[i] = 0.0

    # ---------- 内核 ----------
    def _compile(self):
        n_grid, dx, inv_dx = self.n_grid, self.dx, self.inv_dx
        n_particles, steps = self.n_particles, self.steps
        x, v, C, F = self.x, self.v, self.C, self.F
        gvi, gmi, gvo = self.grid_v_in, self.grid_m_in, self.grid_v_out
        act, aid = self.actuation, self.actuator_id
        weights, bias = self.weights, self.bias
        loss, x_avg = self.loss, self.x_avg
        dt, p_vol, mu, la, gravity = DT, self.p_vol, MU, LA, GRAVITY
        bound, coeff = 3, 0.6

        @ti.kernel
        def clear_and_act(t: ti.i32):
            for i, j in gmi:
                gvi[i, j] = ti.Vector([0.0, 0.0])
                gmi[i, j] = 0.0
            for i in range(4):
                s = 0.0
                for j in ti.static(range(N_SIN)):
                    s += weights[i, j] * ti.sin(ACT_OMEGA * t * dt + 2 * 3.14159265 / N_SIN * j)
                s += bias[i]
                act[t, i] = ti.tanh(s)

        @ti.kernel
        def p2g(f: ti.i32):
            for p in range(n_particles):
                base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
                fx = x[f, p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_F = (ti.Matrix.identity(real, 2) + dt * C[f, p]) @ F[f, p]
                F[f + 1, p] = new_F
                a = act[f, ti.max(0, aid[p])] * ACT_STRENGTH * ti.cast(aid[p] >= 0, real)
                A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * a
                # StVK：eps = (FᵀF - I)/2，P = F(2με + λtr(ε)I)
                eps = 0.5 * (new_F.transpose() @ new_F - ti.Matrix.identity(real, 2))
                tr = eps[0, 0] + eps[1, 1]
                cauchy = (new_F @ (2 * mu * eps + ti.Matrix.identity(real, 2) * (la * tr))
                          @ new_F.transpose())
                cauchy += new_F @ A @ new_F.transpose()
                stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
                affine = stress + C[f, p]
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = (ti.Vector([i, j]) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    gvi[base + ti.Vector([i, j])] += weight * (v[f, p] + affine @ dpos)
                    gmi[base + ti.Vector([i, j])] += weight

        @ti.kernel
        def grid_op():
            for i, j in gmi:
                inv_m = 1.0 / (gmi[i, j] + 1e-10)
                v_out = gvi[i, j] * inv_m
                v_out[1] -= dt * gravity
                if i < bound and v_out[0] < 0.0:
                    v_out[0] = 0.0
                if i > n_grid - bound and v_out[0] > 0.0:
                    v_out[0] = 0.0
                if j < bound:
                    if v_out[1] < 0.0:
                        v_out[1] = 0.0
                    v_out[0] *= coeff          # 地面摩擦
                if j > n_grid - bound and v_out[1] > 0.0:
                    v_out[1] = 0.0
                gvo[i, j] = v_out

        @ti.kernel
        def g2p(f: ti.i32):
            for p in range(n_particles):
                base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
                fx = x[f, p] * inv_dx - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector([0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]) - fx
                    g_v = gvo[base + ti.Vector([i, j])]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
                v[f + 1, p] = new_v
                x[f + 1, p] = x[f, p] + dt * new_v
                C[f + 1, p] = new_C

        @ti.kernel
        def compute_x_avg():
            for i in range(n_particles):
                ti.atomic_add(x_avg[None], (1.0 / n_particles) * x[steps - 1, i])

        self._clear_and_act = clear_and_act
        self._p2g = p2g
        self._grid_op = grid_op
        self._g2p = g2p
        self._compute_x_avg = compute_x_avg

    # ---------- 前向 ----------
    def forward(self):
        for s in range(self.steps - 1):
            self._clear_and_act(s)
            self._p2g(s)
            self._grid_op()
            self._g2p(s)
        self.x_avg[None] = [0.0, 0.0]
        self._compute_x_avg()
        return float(self.x_avg[None][0])

    def set_theta(self, theta):
        th = np.asarray(theta, np.float32).reshape(4, N_SIN + 1)
        for i in range(4):
            for j in range(N_SIN):
                self.weights[i, j] = th[i, j]
            self.bias[i] = th[i, N_SIN]

    def get_theta(self):
        th = np.zeros((4, N_SIN + 1), np.float32)
        for i in range(4):
            for j in range(N_SIN):
                th[i, j] = self.weights[i, j]
            th[i, N_SIN] = self.bias[i]
        return th.reshape(-1)

    def rollout(self):
        """返回 (steps, n, 2) 的轨迹快照，供界面回放。"""
        return self.x.to_numpy()[:, :, :]

    def rollout(self):
        """返回 (steps, n, 2) 的轨迹快照，供界面回放。"""
        return self.x.to_numpy()[:, :, :]


# ==================== 画布 ====================
class RobotCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(560, 460)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 249, 246))
        app = self.app
        traj = app.traj
        if traj is None or len(traj) < 2:
            p.setPen(QColor(150, 145, 138))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignCenter,
                       "点「开始优化」：梯度下降会教它走路")
            p.end()
            return

        w, h = self.width(), self.height()
        idx = int(app.frame_idx) % len(traj)
        pos = traj[idx]
        # 相机跟随：视野宽度约 0.55 个网格单位
        cx = float(pos[:, 0].mean())
        view_w = 1.35
        scale = min(w / view_w, (h - 90) / (view_w * 0.55))
        left = cx - view_w / 2.0

        def X(v):
            return (v - left) * scale

        floor_y0 = 3.0 / self.app.robot.n_grid

        def Y(v):
            return h * 0.80 - (v - floor_y0) * scale

        # 地板：MPM 的墙是网格底部 3 格
        floor_y = floor_y0
        p.setPen(QPen(QColor(120, 116, 110), 2))
        p.drawLine(QPointF(0, Y(floor_y)), QPointF(w, Y(floor_y)))
        # 米尺
        p.setPen(QColor(160, 155, 148))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        import math
        m0 = math.floor(left * 10) / 10
        for k in range(12):
            v = m0 + k * 0.05
            xx = X(v)
            if 0 < xx < w:
                p.setPen(QPen(QColor(226, 222, 214), 1))
                p.drawLine(QPointF(xx, Y(floor_y)), QPointF(xx, Y(floor_y) + 6))
                if abs(v * 10 - round(v * 10)) < 1e-6:
                    p.setPen(QColor(150, 145, 138))
                    p.drawText(QPointF(xx + 2, Y(0.0) + 18), f"{v:.1f}")

        acts = app.acts
        aid = app.aid
        p.setPen(Qt.NoPen)
        for i in range(len(pos)):
            if aid[i] < 0 or acts is None:
                col = QColor(46, 48, 54)
            else:
                a = float(acts[idx, aid[i]])
                a = max(-1.0, min(1.0, a))
                rr = max(0.0, min(1.0, 0.5 - a))
                gg = max(0.0, min(1.0, 0.5 - abs(a)))
                bb = max(0.0, min(1.0, 0.5 + a))
                col = QColor(int(rr * 255), int(gg * 255), int(bb * 255))
            p.setBrush(col)
            r = 2.0
            p.drawEllipse(QPointF(X(float(pos[i][0])), Y(float(pos[i][1]))), r, r)

        p.setPen(QColor(196, 84, 58))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(QPointF(14, 22), f"第 {idx} 步 / {len(traj)}")
        p.end()


# ==================== 训练线程（进化策略） ====================
class ESTrainer(QThread):
    progress = Signal(int, float, float, object)

    def __init__(self, robot, pop=6, sigma=0.10, lr=0.12):
        super().__init__()
        self.robot = robot
        self.pop = pop
        self.sigma = sigma
        self.lr = lr
        self.running = True
        self.rng = np.random.default_rng(1)
        self.theta = robot.get_theta() * 0.0

    def run(self):
        gen = 0
        while self.running:
            eps = self.rng.normal(0, 1, (self.pop, self.theta.size)).astype(np.float32)
            fits = np.zeros(self.pop, np.float32)
            snaps = [None] * self.pop
            actsnap = [None] * self.pop
            for k in range(self.pop):
                if not self.running:
                    return
                self.robot.set_theta(self.theta + self.sigma * eps[k])
                fits[k] = self.robot.forward()
                if fits[k] >= fits.max():
                    snaps[k] = self.robot.rollout()
                    actsnap[k] = self.robot.actuation.to_numpy()
            # 用排名做权重（比原始适应度稳）
            order = np.argsort(fits)
            w = np.zeros(self.pop, np.float32)
            for rank, idx in enumerate(order):
                w[idx] = rank / (self.pop - 1) - 0.5
            self.theta = self.theta + self.lr / (self.pop * self.sigma) * (w[:, None] * eps).sum(0)
            self.robot.set_theta(self.theta)
            base = self.robot.forward()          # 用更新后的参数记录回放
            snap = self.robot.rollout()
            act = self.robot.actuation.to_numpy()
            gen += 1
            self.progress.emit(gen, float(base), float(fits.mean()), (snap, act))
        return

    def stop(self):
        self.running = False


def _unused():
    pass


# ==================== 主窗口 ====================
class SoftRobotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("演化软体 · MPM 软体机器人")
        self.setGeometry(110, 70, 1120, 780)

        self.robot = SoftRobot()
        self.traj = None
        self.acts = None
        self.aid = np.array(self.robot.act_ids, np.int32)
        self.frame_idx = 0.0
        self.history = []
        self.thread = None

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def tick(self):
        if self.traj is not None:
            self.frame_idx = (self.frame_idx + 1.8) % max(1, len(self.traj))
        self.canvas.update()

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

        title = QLabel("MPM 软体机器人")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel(f"{self.robot.n_particles} 个 MPM 粒子构成的软体\n"
                     "4 个驱动器的正弦权重由进化策略搜索")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = RobotCanvas(self)

        g1 = QGroupBox("训练")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(6)
        v1.setContentsMargins(8, 8, 8, 8)
        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.toggle)
        v1.addWidget(self.btn_train)
        self.lbl_gen = QLabel("第 0 代")
        self.lbl_dist = QLabel("距离 –")
        self.lbl_mean = QLabel("平均 –")
        for wdg in (self.lbl_gen, self.lbl_dist, self.lbl_mean):
            v1.addWidget(wdg)
        pv.addWidget(g1)

        g2 = QGroupBox("参数")
        f2 = QFormLayout(g2)
        f2.setSpacing(6)
        f2.setContentsMargins(8, 8, 8, 8)
        self.pop_slider = QSlider(Qt.Horizontal)
        self.pop_slider.setRange(4, 16)
        self.pop_slider.setValue(6)
        self.pop_lbl = QLabel("8")
        self.pop_lbl.setFixedWidth(40)
        self.pop_slider.valueChanged.connect(lambda v: self.pop_lbl.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(self.pop_slider, 1)
        row.addWidget(self.pop_lbl)
        f2.addRow("种群", row)
        pv.addWidget(g2)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("每代评估 8 个扰动样本，每样本 110 步 MPM 模拟")
        self.statusBar().addWidget(self.lbl_status, 1)

    def toggle(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.thread = None
            self.btn_train.setText("继续训练")
            return
        self.thread = ESTrainer(self.robot, pop=self.pop_slider.value())
        self.thread.progress.connect(self.on_progress)
        self.thread.start()
        self.btn_train.setText("暂停")

    def on_progress(self, gen, best, mean, payload):
        snap, acts = payload
        self.traj = snap
        self.acts = acts
        self.history.append((best, mean))
        self.lbl_gen.setText(f"第 {gen} 代")
        self.lbl_dist.setText(f"最远 {best:.3f}")
        self.lbl_mean.setText(f"平均 {mean:.3f}")
        self.lbl_status.setText(
            f"第 {gen} 代 · 最远 {best:.3f} · 平均 {mean:.3f} · "
            f"{self.robot.n_particles} 粒子 / {MAX_STEPS} 步")

    def closeEvent(self, event):
        self.timer.stop()
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(3000)
        try:
            ti.reset()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SoftRobotApp()
    window.show()
    sys.exit(app.exec())
