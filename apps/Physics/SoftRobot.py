from custom_import import *
from PySide6.QtCore import QPointF, QRectF, QThread, Signal

import numpy as np

# ==================== 参数 ====================
N_GRID = 64
MAX_STEPS = 300
DT = 2e-3
MU = 10.0
LA = 10.0
GRAVITY = 5.0
ACT_OMEGA = 20.0
ACT_STRENGTH = 6.0
N_SIN = 4
N_AGENTS = 4               # 同屏 4 个候选机器人并排，一次 rollout 全部评估
N_ACT = 4

real = ti.f32


class Swarm:
    """6 个独立软体机器人共享一个 MPM 网格，互不接触。

    一次前向就得到 6 个样本的适应度，等于把进化策略的批大小搬进了物理。
    """

    def __init__(self, n_grid=N_GRID, steps=MAX_STEPS, n_agents=N_AGENTS):
        ti.reset()
        ti.init(arch=ti.cpu, default_fp=real)

        self.n_grid = n_grid
        self.steps = steps
        self.n_agents = n_agents
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)

        pts, ids, acts = [], [], []
        for a in range(n_agents):
            self._layout(a, pts, ids, acts)
        self.scene = np.array(pts, np.float32)
        self.act_ids = np.array(acts, np.int32)
        self.rid = np.array(ids, np.int32)
        self.n_particles = len(self.scene)
        self.per_agent = self.n_particles // n_agents

        self.x = ti.Vector.field(2, real, shape=(steps * self.n_particles,))
        self.v = ti.Vector.field(2, real, shape=(steps * self.n_particles,))
        self.C = ti.Matrix.field(2, 2, real, shape=(steps * self.n_particles,))
        self.F = ti.Matrix.field(2, 2, real, shape=(steps * self.n_particles,))
        self.gvi = ti.Vector.field(2, real, shape=(n_grid * n_grid,))
        self.gmi = ti.field(real, shape=(n_grid * n_grid,))
        self.gvo = ti.Vector.field(2, real, shape=(n_grid * n_grid,))
        self.aid = ti.field(ti.i32, shape=(self.n_particles,))
        self.rid_f = ti.field(ti.i32, shape=(self.n_particles,))
        self.weights = ti.field(real, shape=(n_agents, N_ACT, N_SIN))
        self.bias = ti.field(real, shape=(n_agents, N_ACT))
        self.act = ti.field(real, shape=(steps, n_agents, N_ACT))
        self.dist = ti.field(real, shape=(n_agents,))
        self.x0f = ti.field(real, shape=(n_agents,))

        self._compile()
        self._init_state()
        for a in range(n_agents):
            m = self.rid == a
            self.x0f[a] = float(self.scene[m, 0].mean())
        self.theta = np.zeros((n_agents, N_ACT * (N_SIN + 1)), np.float32)

    # ---------- 场景 ----------
    def _layout(self, agent, pts, ids, acts):
        x0 = 0.05 + agent * 0.235
        y0 = 0.05
        dx = self.dx

        def rect(px, py, w, h, act):
            wc = max(1, int(w / dx) * 2)
            hc = max(1, int(h / dx) * 2)
            rdx, rdy = w / wc, h / hc
            for i in range(wc):
                for j in range(hc):
                    pts.append([px + (i + 0.5) * rdx, py + (j + 0.5) * rdy])
                    ids.append(agent)
                    acts.append(act)

        rect(x0, y0 + 0.08, 0.24, 0.08, -1)          # 身体
        for k, off in enumerate((0.00, 0.06, 0.16, 0.20)):
            rect(x0 + off, y0, 0.04, 0.08, k)        # 四条腿

    def _init_state(self):
        for p in range(self.n_particles):
            self.x[p] = ti.Vector([self.scene[p, 0], self.scene[p, 1]])
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            self.aid[p] = int(self.act_ids[p])
            self.rid_f[p] = int(self.rid[p])

    # ---------- kernel ----------
    def _compile(self):
        n, np_ = self.n_grid, self.n_particles
        steps, per = self.steps, self.per_agent
        x, v, C, F = self.x, self.v, self.C, self.F
        gvi, gmi, gvo = self.gvi, self.gmi, self.gvo
        aid, rid = self.aid, self.rid_f
        weights, bias, act, dist = self.weights, self.bias, self.act, self.dist
        dt, inv, dx = DT, self.inv_dx, self.dx

        @ti.kernel
        def clear_and_act(s: ti.i32):
            for k in range(n * n):
                gvi[k] = ti.Vector([0.0, 0.0])
                gmi[k] = 0.0
            for a, i in ti.ndrange(N_AGENTS, N_ACT):
                acc = bias[a, i]
                for j in ti.static(range(N_SIN)):
                    acc += weights[a, i, j] * ti.sin(ACT_OMEGA * s * dt
                                                     + 2 * 3.14159265 / N_SIN * j)
                act[s, a, i] = ti.tanh(acc)

        @ti.kernel
        def p2g(s: ti.i32):
            for p in range(np_):
                i0 = s * np_ + p
                base = ti.min(ti.max(ti.cast(x[i0] * inv - 0.5, ti.i32), 0), n - 3)
                fx = x[i0] * inv - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                nF = (ti.Matrix.identity(real, 2) + dt * C[i0]) @ F[i0]
                F[i0 + np_] = nF
                idx_a = ti.max(0, aid[p])
                drive = act[s, rid[p], idx_a] * ACT_STRENGTH * ti.cast(aid[p] >= 0, real)
                A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * drive
                eps = 0.5 * (nF.transpose() @ nF - ti.Matrix.identity(real, 2))
                tr = eps[0, 0] + eps[1, 1]
                cauchy = nF @ (2 * MU * eps + ti.Matrix.identity(real, 2) * (LA * tr)) @ nF.transpose()
                cauchy += nF @ A @ nF.transpose()
                stress = -(dt * 4 * inv * inv) * cauchy
                affine = stress + C[i0]
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = (ti.Vector([i, j]) - fx) * dx
                    wt = w[i][0] * w[j][1]
                    k = (base[0] + i) * n + (base[1] + j)
                    ti.atomic_add(gvi[k], wt * (v[i0] + affine @ dpos))
                    ti.atomic_add(gmi[k], wt)

        @ti.kernel
        def grid_op():
            for k in range(n * n):
                m = ti.max(gmi[k], 1e-3)
                vo = gvi[k] * (1.0 / m)
                vo[1] -= dt * GRAVITY
                gi = k // n
                gj = k % n
                if gi < 2 and vo[0] < 0.0:
                    vo = ti.Vector([0.0, 0.0])
                if gi > n - 3 and vo[0] > 0.0:
                    vo = ti.Vector([0.0, 0.0])
                if gj > n - 3 and vo[1] > 0.0:
                    vo = ti.Vector([0.0, 0.0])
                # 地面：参考里的库仑摩擦（法向冲量与切向速度按系数耦合）
                if gj < 2 and vo[1] < 0.0:
                    normal = ti.Vector([0.0, 1.0])
                    lin = vo.dot(normal)
                    if lin < 0.0:
                        vit = vo - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + 0.6 * lin <= 0.0:
                            vo = ti.Vector([0.0, 0.0])
                        else:
                            vo = (1.0 + 0.6 * lin / lit) * vit
                gvo[k] = vo

        @ti.kernel
        def g2p(s: ti.i32):
            for p in range(np_):
                i0 = s * np_ + p
                base = ti.min(ti.max(ti.cast(x[i0] * inv - 0.5, ti.i32), 0), n - 3)
                fx = x[i0] * inv - ti.cast(base, real)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                nv = ti.Vector([0.0, 0.0])
                nC = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                for i, j in ti.static(ti.ndrange(3, 3)):
                    dpos = ti.Vector([i, j]) - fx
                    k = (base[0] + i) * n + (base[1] + j)
                    gv = gvo[k]
                    wt = w[i][0] * w[j][1]
                    nv += wt * gv
                    nC += 4 * wt * gv.outer_product(dpos) * inv
                v[i0 + np_] = nv
                nx = x[i0] + dt * nv
                x[i0 + np_] = ti.Vector([
                    ti.min(ti.max(nx[0], 2.0 * dx), 1.0 - 3.0 * dx),
                    ti.min(ti.max(nx[1], dx), 1.0 - dx)])
                C[i0 + np_] = nC

        @ti.kernel
        def measure():
            for a in range(N_AGENTS):
                dist[a] = -self.x0f[a]
            for p in range(np_):
                ti.atomic_add(dist[rid[p]], x[(steps - 1) * np_ + p][0] / per)

        self._clear_and_act = clear_and_act
        self._p2g = p2g
        self._grid_op = grid_op
        self._g2p = g2p
        self._measure = measure

    # ---------- 前向 ----------
    def set_thetas(self, thetas):
        th = np.asarray(thetas, np.float32).reshape(self.n_agents, N_ACT, N_SIN + 1)
        self.theta = np.asarray(thetas, np.float32).copy()
        for a in range(self.n_agents):
            for i in range(N_ACT):
                for j in range(N_SIN):
                    self.weights[a, i, j] = th[a, i, j]
                self.bias[a, i] = th[a, i, N_SIN]

    def forward(self):
        for s in range(self.steps - 1):
            self._clear_and_act(s)
            self._p2g(s)
            self._grid_op()
            self._g2p(s)
        self._measure()
        return self.dist.to_numpy().copy()

    def rollout(self):
        # 扁平布局 (steps*n, 2) → (steps, n, 2)
        return self.x.to_numpy().reshape(self.steps, self.n_particles, 2)

    def act_numpy(self):
        return self.act.to_numpy()


# ==================== 画布 ====================
class SwarmCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(620, 460)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 249, 246))
        app = self.app
        traj = app.traj
        if traj is None:
            p.setPen(QColor(150, 145, 138))
            p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignCenter,
                       "点「开始训练」：6 个候选机器人会同时跑")
            p.end()
            return

        w, h = self.width(), self.height()
        floor_y0 = 2.0 / app.sim.n_grid
        view_w = 1.02
        scale = min(w / view_w, (h - 90) / 0.5)
        left = 0.0

        def X(v):
            return (v - left) * scale

        def Y(v):
            return h * 0.80 - (v - floor_y0) * scale

        # 地板
        p.setPen(QPen(QColor(150, 145, 138), 2))
        p.drawLine(QPointF(0, Y(floor_y0)), QPointF(max(w, X(1.0)), Y(floor_y0)))

        idx = int(app.frame_idx) % traj.shape[0]
        pos = traj[idx]
        acts = app.acts
        best = app.best_agent
        p.setPen(Qt.NoPen)
        for i in range(len(pos)):
            a_id = int(app.sim.act_ids[i])
            r_id = int(app.sim.rid[i])
            if a_id < 0:
                col = QColor(44, 46, 52)
                r = 1.6
            else:
                act = float(acts[idx, r_id, a_id]) if acts is not None else 0.0
                act = max(-1.0, min(1.0, act))
                rr = int(max(0.0, min(1.0, 0.5 - act)) * 255)
                gg = int(max(0.0, min(1.0, 0.5 - abs(act))) * 255)
                bb = int(max(0.0, min(1.0, 0.5 + act)) * 255)
                col = QColor(rr, gg, bb)
                r = 1.8
                if r_id != best:
                    col.setAlpha(80)
            p.setBrush(col)
            p.drawEllipse(QPointF(X(float(pos[i][0])), Y(float(pos[i][1]))), r, r)

        # 每个机器人下方的距离
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        for a in range(app.sim.n_agents):
            x0 = 0.05 + a * 0.235
            col = QColor(196, 84, 58) if a == best else QColor(150, 145, 138)
            p.setPen(col)
            p.drawText(QPointF(X(x0) - 4, Y(floor_y0) + 22), f"{app.dists[a]:.2f}")
        p.end()


# ==================== 训练线程 ====================
class ESTrainer(QThread):
    progress = Signal(int, float, float, object, object, int, object)

    def __init__(self, sim, sigma=0.10, lr=0.14):
        super().__init__()
        self.sim = sim
        self.sigma = sigma
        self.lr = lr
        self.running = True
        self.rng = np.random.default_rng(1)
        self.theta = np.zeros(N_ACT * (N_SIN + 1), np.float32)   # 共享参数

    def run(self):
        gen = 0
        while self.running:
            eps = self.rng.normal(0, 1, (self.sim.n_agents, self.theta.size)).astype(np.float32)
            eps[0] = 0.0                                    # 候选 0 是当前参数本身
            cand = self.theta[None, :] + self.sigma * eps
            self.sim.set_thetas(cand.reshape(-1))
            fits = self.sim.forward()                       # 一次 rollout 评估 6 个
            order = np.argsort(fits)
            w = np.zeros(self.sim.n_agents, np.float32)
            for rank, k in enumerate(order):
                w[k] = rank / (self.sim.n_agents - 1) - 0.5
            self.theta = self.theta + self.lr / (self.sim.n_agents * self.sigma) * (
                w[:, None] * eps).sum(0)
            self.sim.set_thetas(np.tile(self.theta, (self.sim.n_agents, 1)).reshape(-1))
            best = int(np.argmax(fits))
            gen += 1
            self.progress.emit(gen, float(fits.max()), float(fits.mean()),
                               self.sim.rollout(), self.sim.act_numpy(), best, fits)
            time.sleep(0.005)

    def stop(self):
        self.running = False


# ==================== 主窗口 ====================
class SoftRobotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("软体机器人 · 6 个候选同场竞技")
        self.setGeometry(110, 70, 1120, 700)

        self.sim = Swarm()
        self.traj = None
        self.acts = None
        self.dists = np.zeros(N_AGENTS, np.float32)
        self.best_agent = 0
        self.frame_idx = 0.0
        self.thread = None

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def tick(self):
        if self.traj is not None:
            self.frame_idx = (self.frame_idx + 2.0) % max(1, self.traj.shape[0])
        self.canvas.update()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(268)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("软体机器人竞技")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel(f"{N_AGENTS} 个候选并排放在同一网格里\n"
                     "一次 MPM 前向就评估全部")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = SwarmCanvas(self)

        g1 = QGroupBox("训练")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(6)
        v1.setContentsMargins(8, 8, 8, 8)
        self.btn = QPushButton("开始训练")
        self.btn.clicked.connect(self.toggle)
        v1.addWidget(self.btn)
        self.lbl_gen = QLabel("第 0 代")
        self.lbl_best = QLabel("最远 –")
        self.lbl_mean = QLabel("平均 –")
        for wdg in (self.lbl_gen, self.lbl_best, self.lbl_mean):
            v1.addWidget(wdg)
        pv.addWidget(g1)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel(
            f"{self.sim.n_particles} 粒子 / {MAX_STEPS} 步 · 6 个候选一次评估")
        self.statusBar().addWidget(self.lbl_status, 1)

    def toggle(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.thread = None
            self.btn.setText("继续训练")
            return
        self.thread = ESTrainer(self.sim)
        self.thread.progress.connect(self.on_progress)
        self.thread.start()
        self.btn.setText("暂停")

    def on_progress(self, gen, best, mean, snap, acts, best_agent, dists):
        self.traj = snap
        self.acts = acts
        self.best_agent = best_agent
        self.dists = dists
        self.lbl_gen.setText(f"第 {gen} 代")
        self.lbl_best.setText(f"最远 {best:.3f}")
        self.lbl_mean.setText(f"平均 {mean:.3f}")

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
