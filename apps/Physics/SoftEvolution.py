from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF

import numpy as np

# ==================== 参数 ====================
GRAVITY = 9.81
N_NODES = 14                 # 每条虫子 14 个节点
SPACING = 0.34
NODE_R = 0.11
DT = 1.0 / 240.0
SUBSTEPS = 6                 # 每帧推进 6 个物理步 → 1/40 s
ITERS = 8                    # 位置约束迭代次数
GEN_TIME = 7.0               # 每代模拟时长（秒）
FPS = 40.0


class Genome:
    """基因：每条肌肉的振幅与相位 + 全局频率。"""

    def __init__(self, pop, n_muscles, rng):
        self.amp = rng.random((pop, n_muscles), dtype=np.float32)
        self.phase = (rng.random((pop, n_muscles)) * 2 * np.pi).astype(np.float32)
        self.freq = rng.uniform(0.6, 2.2, pop).astype(np.float32)   # 每秒周期数

    def copy(self):
        g = Genome.__new__(Genome)
        g.amp = self.amp.copy()
        g.phase = self.phase.copy()
        g.freq = self.freq.copy()
        return g

    def select(self, idx):
        g = Genome.__new__(Genome)
        g.amp = self.amp[idx].copy()
        g.phase = self.phase[idx].copy()
        g.freq = self.freq[idx].copy()
        return g


class Swarm:
    """整个种群一起做位置约束（PBD）软体模拟。"""

    def __init__(self, pop=24, seed=3):
        self.pop = pop
        self.rng = np.random.default_rng(seed)
        self.n = N_NODES

        # 拓扑：相邻弹簧（可收缩）+ 隔一个的加强弹簧
        pairs = [(i, i + 1) for i in range(self.n - 1)]
        self.muscles = len(pairs)
        stiff = [(i, i + 2) for i in range(self.n - 2)]
        self.springs = pairs + stiff
        self.rest = np.array([SPACING if j == i + 1 else 2 * SPACING
                              for i, j in self.springs], np.float32)
        self.stiff_mask = np.array([1.0 if j == i + 1 else 0.0
                                    for i, j in self.springs], np.float32)
        self.si = np.array([s[0] for s in self.springs], np.int32)
        self.sj = np.array([s[1] for s in self.springs], np.int32)

        # 奇数节点是脚，压在地上；偶数节点是躯干，可以抬离地面
        self.foot = np.ones(self.n, np.float32)
        self.foot[0::2] = 1.0
        self.foot[1::2] = 1.0
        self.foot = np.where(np.arange(self.n) % 2 == 1, 1.0, 0.55).astype(np.float32)

        self.genome = Genome(pop, self.muscles, self.rng)
        self._refresh_muscles()
        self.reset()

    def _refresh_muscles(self):
        m = len(self.springs)
        self.amp_full = np.zeros((self.pop, m), np.float32)
        self.phase_full = np.zeros((self.pop, m), np.float32)
        self.amp_full[:, :self.muscles] = self.genome.amp
        self.phase_full[:, :self.muscles] = self.genome.phase

    # ---------- 状态 ----------
    def reset(self):
        pop, n = self.pop, self.n
        self.x = np.zeros((pop, n, 2), np.float64)
        self.x[:, :, 0] = np.arange(n) * SPACING
        self.x[:, :, 1] = 0.6
        self.v = np.zeros((pop, n, 2), np.float64)
        self.t = 0.0
        self.start_x = self.x[:, :, 0].mean(axis=1).copy()
        self.done = np.zeros(pop, bool)

    def step(self, dt=DT):
        p, n = self.pop, self.n
        g = self.genome
        # 肌肉目标长度
        w = 2 * np.pi * g.freq[:, None]                     # (P,1)
        drive = 0.5 + 0.5 * np.sin(w * self.t + self.phase_full)   # (P,M)
        target = self.rest[None, :] * (1.0 - 0.45 * self.amp_full * drive)

        self.v[:, :, 1] -= GRAVITY * dt
        self.v *= 0.994
        prev = self.x.copy()
        self.x += self.v * dt

        # 位置约束
        for _ in range(ITERS):
            i, j = self.si, self.sj
            d = self.x[:, j] - self.x[:, i]
            dist = np.linalg.norm(d, axis=2) + 1e-9
            corr = (dist - target) / dist * 0.5
            off = d * corr[..., None]
            self.x[:, i] += off
            self.x[:, j] -= off
            # 抗弯：每个内节点朝邻居中点靠，身体才会起伏而不是像根棍子
            self.x[:, 1:-1] += 0.10 * (0.5 * (self.x[:, :-2] + self.x[:, 2:]) - self.x[:, 1:-1])

            # 地面 + 静摩擦：贴地节点的切向位移只留一小部分，逼它真正爬
            low = self.x[:, :, 1] < NODE_R * self.foot[None, :]
            self.x[:, :, 1] = np.where(low, NODE_R * self.foot[None, :], self.x[:, :, 1])
            on_ground = self.x[:, :, 1] <= NODE_R * self.foot[None, :] + 1e-4
            dx = self.x[:, :, 0] - prev[:, :, 0]
            self.x[:, :, 0] = np.where(on_ground,
                                       prev[:, :, 0] + dx * 0.22,
                                       self.x[:, :, 0])

        # 地面摩擦
        low = self.x[:, :, 1] <= NODE_R + 1e-6
        self.v[:, :, 1] = np.where(low, np.minimum(self.v[:, :, 1], 0.0), self.v[:, :, 1])
        self.v = (self.x - prev) / dt
        np.clip(self.v, -6.0, 6.0, out=self.v)
        self.v[:, :, 0] *= 0.965
        self.t += dt

    # ---------- 适应度 ----------
    def fitness(self):
        cx = self.x[:, :, 0].mean(axis=1)
        dist = cx - self.start_x
        # 翻倒（整体重心太低或头部着地过久）不惩罚，只要跑得远
        return dist

    def evolve(self):
        fit = self.fitness()
        order = np.argsort(-fit)
        keep = max(2, self.pop // 4)
        elite = order[:keep]
        new = Genome(self.pop, self.muscles, self.rng)
        # 精英直接复制
        for k in range(min(keep, self.pop)):
            new.amp[k] = self.genome.amp[elite[k]]
            new.phase[k] = self.genome.phase[elite[k]]
            new.freq[k] = self.genome.freq[elite[k]]
        # 其余由精英杂交 + 变异
        for k in range(keep, self.pop):
            a = elite[self.rng.integers(0, keep)]
            b = elite[self.rng.integers(0, keep)]
            mask = self.rng.random(self.muscles) < 0.5
            new.amp[k] = np.where(mask, self.genome.amp[a], self.genome.amp[b])
            new.phase[k] = np.where(mask, self.genome.phase[a], self.genome.phase[b])
            new.freq[k] = self.genome.freq[a] if self.rng.random() < 0.5 else self.genome.freq[b]
            # 变异
            new.amp[k] = np.clip(new.amp[k] + self.rng.normal(0, 0.12, self.muscles), 0, 1)
            new.phase[k] = (new.phase[k] + self.rng.normal(0, 0.35, self.muscles)) % (2 * np.pi)
            if self.rng.random() < 0.4:
                new.freq[k] = np.clip(new.freq[k] + self.rng.normal(0, 0.12), 0.4, 3.0)
        self.genome = new
        self._refresh_muscles()
        self.reset()
        return float(fit.max()), float(fit.mean())


# ==================== 画布 ====================
class SwarmCanvas(QWidget):
    """每个个体一条泳道，像赛跑一样横向对比。"""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(640, 420)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(247, 245, 240))
        app = self.app
        sim = app.sim
        if sim is None:
            p.end()
            return

        w, h = self.width(), self.height()
        top, bottom = 30, h - 16
        lane_h = (bottom - top) / sim.pop
        scale = min(88.0, (w - 90) / 11.0)          # 视野里保留 12 米左右

        lead = float(sim.x[:, :, 0].max())
        cam = max(0.0, lead - 9.0)

        def X(x):
            return (x - cam) * scale + 64

        fit = sim.fitness()
        order = np.argsort(-fit)          # 名次榜：第一名在最上面
        best = int(order[0])

        f = p.font()
        f.setPointSize(7)
        p.setFont(f)

        for rank in range(sim.pop):
            k = int(order[rank])
            base = top + lane_h * (rank + 0.72)
            is_best = (rank == 0)
            # 泳道底
            if is_best:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(196, 84, 58, 22))
                p.drawRect(QRectF(0, base - lane_h * 0.62, w, lane_h * 0.9))
            p.setPen(QPen(QColor(226, 222, 214), 1))
            p.drawLine(QPointF(40, base), QPointF(w - 8, base))
            # 距离刻度
            for m in range(int(cam) + 1, int(cam) + 16):
                xx = X(m)
                if 40 < xx < w - 8:
                    p.setPen(QPen(QColor(236, 232, 224), 1))
                    p.drawLine(QPointF(xx, base), QPointF(xx, base - lane_h * 0.28))
            # 个体
            col = QColor(196, 84, 58) if is_best else QColor(120, 140, 172)
            col.setAlpha(240 if is_best else 110)
            p.setPen(QPen(col, 1.8 if is_best else 1.0))
            pts = [QPointF(X(sim.x[k, i, 0]), base - sim.x[k, i, 1] * scale)
                   for i in range(sim.n)]
            p.drawPolyline(QPolygonF(pts))
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            for q in pts:
                r = 3.0 if is_best else 1.8
                p.drawEllipse(q, r, r)
            # 距离数字
            p.setPen(col)
            p.drawText(QPointF(6, base - 2), f"{fit[k]:+.1f}")

        # 顶部距离标尺
        p.setPen(QColor(150, 145, 138))
        for m in range(int(cam) + 2, int(cam) + 16, 1):
            xx = X(m)
            if 50 < xx < w - 10:
                p.drawText(QPointF(xx - 8, 18), f"{m} m")

        self._draw_chart(p, w)
        p.end()

    def _draw_chart(self, p, w):
        hist = self.app.history
        if len(hist) < 2:
            return
        cw, ch = 200, 74
        x0, y0 = w - cw - 12, 26
        p.setPen(QPen(QColor(220, 216, 208), 1))
        p.setBrush(QColor(252, 251, 248, 235))
        p.drawRect(QRectF(x0, y0, cw, ch))
        best = np.array([b for b, m in hist])
        mean = np.array([m for b, m in hist])
        hi = max(1e-3, best.max(), mean.max())
        n = len(hist)
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        p.setPen(QColor(120, 116, 110))
        p.drawText(QPointF(x0 + 6, y0 + 11), f"{n} 代 · 最远 {best.max():.1f} m")
        for arr, col in ((mean, QColor(150, 165, 190)), (best, QColor(196, 84, 58))):
            pts = []
            for i, v in enumerate(arr):
                xx = x0 + 6 + (cw - 12) * i / max(1, n - 1)
                yy = y0 + ch - 6 - (ch - 24) * min(1.0, max(0.0, v) / hi)
                pts.append(QPointF(xx, yy))
            p.setPen(QPen(col, 1.3))
            p.drawPolyline(QPolygonF(pts))


# ==================== 主窗口 ====================
class SoftEvolutionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("演化软体 · 遗传算法学爬行")
        self.setGeometry(120, 80, 1120, 640)

        self.sim = Swarm(pop=24)
        self.generation = 1
        self.history = []
        self.show_all = True
        self.running = True
        self._gen_steps = 0
        self._gen_total = int(GEN_TIME * FPS)

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / FPS))

    # ---------- 循环 ----------
    def tick(self):
        if not self.running:
            return
        for _ in range(SUBSTEPS):
            self.sim.step()
        self._gen_steps += 1
        if self._gen_steps >= self._gen_total:
            best, mean = self.sim.evolve()
            self.history.append((best, mean))
            self.generation += 1
            self._gen_steps = 0
            self.lbl_gen.setText(f"第 {self.generation} 代")
            self.lbl_best.setText(f"最远 {best:.2f} m")
        self.canvas.update()
        fit = self.sim.fitness()
        self.lbl_now.setText(f"当前 {fit.max():.2f} m")

    # ---------- UI ----------
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

        title = QLabel("演化软体")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("24 条软体虫子并排赛跑\n肌肉收缩由基因决定，跑得远的留下后代")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = SwarmCanvas(self)

        g1 = QGroupBox("世代")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(4)
        v1.setContentsMargins(8, 8, 8, 8)
        self.lbl_gen = QLabel("第 1 代")
        self.lbl_best = QLabel("最远 –")
        self.lbl_now = QLabel("当前 –")
        for wdg in (self.lbl_gen, self.lbl_best, self.lbl_now):
            v1.addWidget(wdg)
        pv.addWidget(g1)

        g2 = QGroupBox("控制")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(6)
        v2.setContentsMargins(8, 8, 8, 8)
        self.btn_run = QPushButton("暂停")
        self.btn_run.clicked.connect(self.toggle_run)
        v2.addWidget(self.btn_run)
        row = QHBoxLayout()
        row.setSpacing(6)
        b_reset = QPushButton("重开种群")
        b_reset.clicked.connect(self.reset)
        row.addWidget(b_reset)
        b_next = QPushButton("跳过本代")
        b_next.clicked.connect(self.skip)
        row.addWidget(b_next)
        v2.addLayout(row)
        self.chk_all = QCheckBox("显示全部个体")
        self.chk_all.setChecked(True)
        self.chk_all.toggled.connect(self._toggle_all)
        v2.addWidget(self.chk_all)
        pv.addWidget(g2)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("每一代 7 秒，取前 25% 杂交变异，生成下一代")
        self.statusBar().addWidget(self.lbl_status, 1)

    def toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("暂停" if self.running else "继续")

    def skip(self):
        best, mean = self.sim.evolve()
        self.history.append((best, mean))
        self.generation += 1
        self._gen_steps = 0
        self.lbl_gen.setText(f"第 {self.generation} 代")
        self.lbl_best.setText(f"最远 {best:.2f} m")
        self.canvas.update()

    def reset(self):
        self.sim = Swarm(pop=24, seed=int(time.time()) % 10000)
        self.generation = 1
        self.history = []
        self._gen_steps = 0
        self.lbl_gen.setText("第 1 代")
        self.lbl_best.setText("最远 –")
        self.canvas.update()

    def _toggle_all(self, on):
        self.show_all = bool(on)
        self.canvas.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SoftEvolutionApp()
    window.show()
    sys.exit(app.exec())
