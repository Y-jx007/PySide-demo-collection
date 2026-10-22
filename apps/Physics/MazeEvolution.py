from custom_import import *
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPolygonF

import numpy as np

# ==================== 迷宫 ====================
CELL = 1.0
SUB = 8                    # 每格 8×8 采样点，用来做射线与碰撞
N_SENSOR = 5
SENSOR_ANGLES = np.deg2rad([-70, -35, 0, 35, 70])
SENSOR_RANGE = 3.6
AGENT_R = 0.16


class Maze:
    def __init__(self, cw=14, ch=10, seed=0):
        self.cw, self.ch = cw, ch
        self.rng = np.random.default_rng(seed)
        self.build(seed)

    def build(self, seed):
        self.rng = np.random.default_rng(seed)
        cw, ch = self.cw, self.ch
        # 每格记录四壁：N, E, S, W
        walls = np.ones((ch, cw, 4), bool)
        visited = np.zeros((ch, cw), bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            i, j = stack[-1]
            nbrs = []
            for d, (di, dj) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                ni, nj = i + di, j + dj
                if 0 <= ni < cw and 0 <= nj < ch and not visited[nj, ni]:
                    nbrs.append((d, ni, nj))
            if not nbrs:
                stack.pop()
                continue
            d, ni, nj = nbrs[self.rng.integers(len(nbrs))]
            walls[j, i, d] = False
            walls[nj, ni, (d + 2) % 4] = False
            visited[nj, ni] = True
            stack.append((ni, nj))
        self.walls = walls

        # 采样网格
        self.H = ch * SUB
        self.W = cw * SUB
        occ = np.zeros((self.H, self.W), bool)
        occ[0, :] = occ[-1, :] = True
        occ[:, 0] = occ[:, -1] = True
        for j in range(ch):
            for i in range(cw):
                y0, x0 = j * SUB, i * SUB
                if walls[j, i, 0]:
                    occ[y0, x0:x0 + SUB] = True
                if walls[j, i, 2]:
                    occ[y0 + SUB - 1, x0:x0 + SUB] = True
                if walls[j, i, 3]:
                    occ[y0:y0 + SUB, x0] = True
                if walls[j, i, 1]:
                    occ[y0:y0 + SUB, x0 + SUB - 1] = True
        self.occ = occ
        self.goal = np.array([cw - 0.5, ch - 0.5], np.float32)
        self.start = np.array([0.5, 0.5], np.float32)
        self.goal_cell = (cw - 1, ch - 1)

    def blocked(self, x, y):
        ix = np.clip((x * SUB).astype(np.int32), 0, self.W - 1)
        iy = np.clip((y * SUB).astype(np.int32), 0, self.H - 1)
        return self.occ[iy, ix]

    def blocked_circle(self, x, y, r=AGENT_R):
        """圆与墙是否相交：圆周八点采样，比四点更不容易漏掉薄墙。"""
        d = r * 0.7071
        hit = self.blocked(x + r, y) | self.blocked(x - r, y)
        hit |= self.blocked(x, y + r) | self.blocked(x, y - r)
        hit |= self.blocked(x + d, y + d) | self.blocked(x + d, y - d)
        hit |= self.blocked(x - d, y + d) | self.blocked(x - d, y - d)
        return hit

    def ray(self, x, y, ang, maxr=SENSOR_RANGE, steps=24):
        """沿角度方向找最近的墙，返回归一化距离（0=贴墙，1=开阔）。"""
        n = x.shape[0]
        best = np.full(n, maxr, np.float32)
        for k in range(1, steps + 1):
            d = maxr * k / steps
            px = x + np.cos(ang) * d
            py = y + np.sin(ang) * d
            hit = self.blocked(px, py) & (best == maxr)
            best = np.where(hit, d, best)
        return best / maxr


# ==================== 群体 ====================
class Swarm:
    def __init__(self, pop=80, maze=None, seed=1):
        self.pop = pop
        self.maze = maze or Maze(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.w1 = (self.rng.normal(0, 0.4, (pop, N_SENSOR, 8))).astype(np.float32)
        self.b1 = np.zeros((pop, 8), np.float32)
        self.w2 = (self.rng.normal(0, 0.4, (pop, 8, 2))).astype(np.float32)
        self.b2 = np.zeros((pop, 2), np.float32)
        self.reset_run()

    # ---------- 一代开始 ----------
    def reset_run(self):
        self.pos = np.tile(self.maze.start, (self.pop, 1)).astype(np.float32)
        self.pos += self.rng.normal(0, 0.05, self.pos.shape).astype(np.float32)
        self.ang = np.full(self.pop, 0.0, np.float32)
        self.alive = np.ones(self.pop, bool)
        self.reached = np.zeros(self.pop, bool)
        self.t = 0.0
        self.trail = [[] for _ in range(self.pop)]
        self.best_path = None

    # ---------- 单步 ----------
    def step(self, dt=1.0 / 30):
        m = self.maze
        sens = np.stack([m.ray(self.pos[:, 0], self.pos[:, 1], self.ang + a)
                         for a in SENSOR_ANGLES], axis=1)
        h = np.tanh(np.einsum('pi,pih->ph', sens, self.w1) + self.b1)
        out = np.tanh(np.einsum('ph,pho->po', h, self.w2) + self.b2)

        turn = out[:, 0] * 2.2
        speed = 0.75 + 0.60 * (out[:, 1] + 1) * 0.5
        self.ang = self.ang + turn * dt
        nx = self.pos[:, 0] + np.cos(self.ang) * speed * dt
        ny = self.pos[:, 1] + np.sin(self.ang) * speed * dt

        # 优先级：完整位移 > 只走 x > 只走 y > 原地
        ox, oy = self.pos[:, 0], self.pos[:, 1]
        ok_b = ~m.blocked_circle(nx, ny)
        ok_x = ~m.blocked_circle(nx, oy)
        ok_y = ~m.blocked_circle(ox, ny)
        nx2 = np.where(ok_b, nx, np.where(ok_x, nx, ox))
        ny2 = np.where(ok_b, ny, np.where(ok_x, oy, np.where(ok_y, ny, oy)))
        self.pos = np.stack([nx2, ny2], axis=1)

        d = np.linalg.norm(self.pos - m.goal, axis=1)
        self.reached |= d < 0.55
        self.t += dt

    # ---------- 适应度 ----------
    def fitness(self):
        d0 = np.linalg.norm(self.maze.start - self.maze.goal)
        d = np.linalg.norm(self.pos - self.maze.goal, axis=1)
        prog = d0 - d
        return prog + self.reached * 12.0

    # ---------- 繁殖 ----------
    def evolve(self):
        fit = self.fitness()
        order = np.argsort(-fit)
        keep = max(4, self.pop // 5)
        elite = order[:keep]
        nw1 = self.w1[elite].copy()
        nb1 = self.b1[elite].copy()
        nw2 = self.w2[elite].copy()
        nb2 = self.b2[elite].copy()

        while len(nw1) < self.pop:
            a = elite[self.rng.integers(0, keep)]
            b = elite[self.rng.integers(0, keep)]
            mask1 = self.rng.random(self.w1.shape[1:]) < 0.5
            mask2 = self.rng.random(self.w2.shape[1:]) < 0.5
            w1 = np.where(mask1, self.w1[a], self.w1[b])
            w2 = np.where(mask2, self.w2[a], self.w2[b])
            b1 = self.b1[a].copy()
            b2 = self.b2[a].copy()
            w1 = w1 + self.rng.normal(0, 0.18, w1.shape).astype(np.float32)
            w2 = w2 + self.rng.normal(0, 0.18, w2.shape).astype(np.float32)
            b1 = b1 + self.rng.normal(0, 0.12, b1.shape).astype(np.float32)
            b2 = b2 + self.rng.normal(0, 0.12, b2.shape).astype(np.float32)
            nw1 = np.concatenate([nw1, w1[None]])
            nb1 = np.concatenate([nb1, b1[None]])
            nw2 = np.concatenate([nw2, w2[None]])
            nb2 = np.concatenate([nb2, b2[None]])

        self.w1, self.b1, self.w2, self.b2 = nw1[:self.pop], nb1[:self.pop], nw2[:self.pop], nb2[:self.pop]
        best_path = self.best_path
        self.reset_run()
        return float(fit.max()), float(fit[order[0]] if False else fit.mean()), best_path


# ==================== 画布 ====================
class MazeCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMinimumSize(620, 460)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 249, 246))
        app = self.app
        sw = app.swarm
        if sw is None:
            p.end()
            return
        m = sw.maze
        w, h = self.width(), self.height()
        scale = min((w - 20) / m.cw, (h - 20) / m.ch)
        ox = (w - m.cw * scale) / 2
        oy = (h - m.ch * scale) / 2

        def X(v):
            return ox + v * scale

        def Y(v):
            return oy + v * scale

        # 墙：用圆头粗线画，转角自然衔接
        p.setPen(QPen(QColor(52, 54, 60), max(2.0, 0.16 * scale),
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for j in range(m.ch):
            for i in range(m.cw):
                x0 = i * CELL
                y0 = j * CELL
                if m.walls[j, i, 0]:
                    p.drawLine(QPointF(X(x0), Y(y0)), QPointF(X(x0 + CELL), Y(y0)))
                if m.walls[j, i, 3]:
                    p.drawLine(QPointF(X(x0), Y(y0)), QPointF(X(x0), Y(y0 + CELL)))
                if i == m.cw - 1 and m.walls[j, i, 1]:
                    p.drawLine(QPointF(X(x0 + CELL), Y(y0)), QPointF(X(x0 + CELL), Y(y0 + CELL)))
                if j == m.ch - 1 and m.walls[j, i, 2]:
                    p.drawLine(QPointF(X(x0), Y(y0 + CELL)), QPointF(X(x0 + CELL), Y(y0 + CELL)))

        # 起点与终点
        sx, sy = m.start
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(190, 186, 178), 1.6, Qt.DashLine))
        p.drawEllipse(QPointF(X(sx), Y(sy)), 0.30 * scale, 0.30 * scale)
        gx, gy = m.goal
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(96, 176, 120, 60))
        p.drawEllipse(QPointF(X(gx), Y(gy)), 0.42 * scale, 0.42 * scale)
        p.setBrush(QColor(96, 176, 120))
        p.drawEllipse(QPointF(X(gx), Y(gy)), 0.26 * scale, 0.26 * scale)

        # 最优个体的轨迹
        if app.best_trail:
            p.setPen(QPen(QColor(196, 84, 58, 120), 1.6))
            for k in range(len(app.best_trail) - 1):
                a = app.best_trail[k]
                b = app.best_trail[k + 1]
                p.drawLine(QPointF(X(a[0]), Y(a[1])), QPointF(X(b[0]), Y(b[1])))

        # 个体
        fit = sw.fitness()
        best = int(np.argmax(fit))
        p.setPen(Qt.NoPen)
        for i in range(sw.pop):
            if not app.show_all and i != best:
                continue
            if i == best:
                col = QColor(196, 84, 58)
                r = 0.16 * scale
            else:
                col = QColor(110, 130, 165, 110)
                r = 0.11 * scale
            p.setBrush(col)
            p.drawEllipse(QPointF(X(sw.pos[i, 0]), Y(sw.pos[i, 1])), r, r)
            if i == best:
                p.setPen(QPen(QColor(196, 84, 58), 1.6))
                p.drawLine(QPointF(X(sw.pos[i, 0]), Y(sw.pos[i, 1])),
                           QPointF(X(sw.pos[i, 0] + np.cos(sw.ang[i]) * 0.35),
                                   Y(sw.pos[i, 1] + np.sin(sw.ang[i]) * 0.35)))
                p.setPen(Qt.NoPen)
        p.end()


# ==================== 主窗口 ====================
class MazeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("进化神经网络走迷宫")
        self.setGeometry(110, 70, 1120, 760)

        self.maze_seed = 7
        self.swarm = Swarm(pop=80, maze=Maze(10, 8, seed=self.maze_seed))
        self.generation = 1
        self.running = True
        self.show_all = True
        self.episode_t = 0.0
        self.episode_len = 35.0
        self.best_trail = []
        self.best_fit = 0.0
        self.reached = 0

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---------- 主循环 ----------
    def tick(self):
        if not self.running:
            return
        sw = self.swarm
        for _ in range(4):                      # 每帧推进 4 个仿真步
            sw.step()
        self.episode_t += 4 / 30.0
        fit = sw.fitness()
        best = int(np.argmax(fit))
        self.best_trail.append(tuple(sw.pos[best]))
        if len(self.best_trail) > 900:
            self.best_trail.pop(0)
        if self.episode_t >= self.episode_len:
            self.next_generation()
        self.canvas.update()
        self.update_stats()

    def next_generation(self):
        sw = self.swarm
        fit = sw.fitness()
        best = float(fit.max())
        reached = int(sw.reached.sum())
        self.best_fit = max(self.best_fit, best)
        self.reached = reached
        sw.evolve()
        self.generation += 1
        self.episode_t = 0.0
        self.best_trail = []
        self.lbl_gen.setText(f"第 {self.generation} 代")
        self.lbl_best.setText(f"本代最佳 {best:.1f}")
        self.lbl_reach.setText(f"到达终点 {reached} 个")

    # ---------- UI ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        panel = QWidget()
        panel.setFixedWidth(266)
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)
        pv.setContentsMargins(0, 0, 0, 0)

        title = QLabel("进化神经网络走迷宫")
        tf = title.font()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        pv.addWidget(title)
        sub = QLabel("每条虫有 5 条射线感知 + 一个 5-8-2 小网络\n"
                     "撞墙就停，几代后能自己绕出迷宫")
        sub.setStyleSheet("color: #7a7a80;")
        sf = sub.font()
        sf.setPointSize(8)
        sub.setFont(sf)
        pv.addWidget(sub)

        self.canvas = MazeCanvas(self)

        g1 = QGroupBox("进度")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(4)
        v1.setContentsMargins(8, 8, 8, 8)
        self.lbl_gen = QLabel("第 1 代")
        self.lbl_best = QLabel("本代最佳 –")
        self.lbl_reach = QLabel("到达终点 0 个")
        for wdg in (self.lbl_gen, self.lbl_best, self.lbl_reach):
            v1.addWidget(wdg)
        pv.addWidget(g1)

        g2 = QGroupBox("控制")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(6)
        v2.setContentsMargins(8, 8, 8, 8)
        self.btn = QPushButton("暂停")
        self.btn.clicked.connect(self.toggle)
        v2.addWidget(self.btn)
        row = QHBoxLayout()
        row.setSpacing(6)
        b_new = QPushButton("新迷宫")
        b_new.clicked.connect(self.new_maze)
        row.addWidget(b_new)
        b_next = QPushButton("跳过本代")
        b_next.clicked.connect(self.next_generation)
        row.addWidget(b_next)
        v2.addLayout(row)
        chk = QCheckBox("显示全部个体")
        chk.setChecked(True)
        chk.toggled.connect(self._toggle_all)
        v2.addWidget(chk)
        pv.addWidget(g2)
        pv.addStretch()

        root.addWidget(panel)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(0)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        root.addWidget(right, 1)

        self.lbl_status = QLabel("64 个个体同时跑，每代 30 秒；红点是最强个体")
        self.statusBar().addWidget(self.lbl_status, 1)

    def toggle(self):
        self.running = not self.running
        self.btn.setText("暂停" if self.running else "继续")

    def new_maze(self):
        self.maze_seed += 1
        self.swarm = Swarm(pop=80, maze=Maze(10, 8, seed=self.maze_seed))
        self.generation = 1
        self.episode_t = 0.0
        self.best_trail = []
        self.best_fit = 0.0
        self.lbl_gen.setText("第 1 代")
        self.lbl_best.setText("本代最佳 –")
        self.lbl_reach.setText("到达终点 0 个")
        self.canvas.update()

    def _toggle_all(self, on):
        self.show_all = bool(on)
        self.canvas.update()

    def update_stats(self):
        fit = self.swarm.fitness()
        self.lbl_status.setText(
            f"第 {self.generation} 代 · {self.episode_t:.1f}/{self.episode_len:.0f} 秒 · "
            f"当前最佳 {fit.max():.1f}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MazeApp()
    window.show()
    sys.exit(app.exec())
