# PySide 绘图工具集 — 数学与物理可视化

一个基于 **PySide6 + Taichi + OpenGL** 的交互式数学/物理可视化工具集，包含 28 个可独立运行的应用。

## 界面预览

<img width="1312" height="660" alt="image" src="https://github.com/user-attachments/assets/ff9e75e9-4eb0-4258-8a69-0d75faad1c13" />

## 功能列表

### Math

| 应用 | 说明 |
|------|------|
| **Fourier** | 绘制图形并查看其二维傅里叶变换 |
| **Arnold** | 图像猫变换（Arnold's cat map） |
| **Lloyd** | 用 Voronoi 图的 Lloyd 松弛模拟蜂巢结构 |
| **Cplot** | 复变函数幅角图，基于 [cplot](https://github.com/nschloe/cplot) |
| **Traveler** | 遗传算法求解旅行商问题 |
| **Lorenz** | 三维洛伦兹吸引子可视化 |

### Physics

| 应用 | 说明 |
|------|------|
| **Schrödinger** | Runge-Kutta 模拟二维量子波包演化 |
| **Turing** | Gray-Scott 等反应扩散系统 |
| **Ising** | 二维伊辛模型，参考 [Ising.js](https://mattbierbaum.github.io/ising.js/) |
| **Maxwell** | 二维麦克斯韦-玻尔兹曼速率分布 |
| **Navier-Stokes** | 2D Q9 格子 Boltzmann 模拟卡门涡街，参考 [LBM_Taichi](https://github.com/hietwll/LBM_Taichi) |

### Fractals

| 应用 | 说明 |
|------|------|
| **Quaternion Julia** | 三维四元数 Julia 集 |
| **Mandelbrot-Julia** | 双画布 Mandelbrot / Julia 集，支持自定义函数 |
| **Newton** | Newton / Nova 分形 |
| **Lindenmayer** | L-System 分形 |
| **IFS** | 迭代函数系统分形 |
| **Logistic** | Lyapunov 分形与分叉图 |

### Cellular Automata

| 应用 | 说明 |
|------|------|
| **Lenia** | 连续元胞自动机，参考 [Lenia](https://chakazul.github.io/Lenia/JavaScript/Lenia.html) |
| **Conway** | 生命游戏 |
| **Distill** | 神经网络元胞自动机，参考 [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)，需要 PyTorch |

### 实验（未完全打磨）

| 应用 | 说明 |
|------|------|
| **2D Boids** | Boids 群体行为 |
| **2D Noise** | 麦克风实时噪声频谱（需要 sounddevice） |
| **2D Qusicrystal** | 二维准晶衍射图样 |
| **Mass Spring** | 质点-弹簧系统 |
| **Particle Lenia** | 粒子化 Lenia |
| **Caustic** | 焦散/光线折射模拟 |
| **Deformation** | 网格形变可视化 |
| **KAM** | KAM 环面与标准映射 |

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

Distill 需要额外的 PyTorch：

```bash
pip install -r requirements-distill.txt
```

### 运行

启动主界面：

```bash
python main.py
```

主界面会为每个应用启动**独立子进程**，因此可以同时打开多个应用，互不影响 Taichi / OpenGL / PyTorch 运行时；关闭主界面时会一并关闭子进程。

单独运行某个应用：

```bash
python app_runner.py "apps/Fractals/Mandelbrot.py"
```

`app_runner.py` 会配置好 `apps/` 下的导入路径，并把工作目录切换到项目根目录，避免资源路径依赖启动位置。

## 技术栈

- **GUI**：PySide6 (Qt for Python)
- **GPU 计算**：Taichi（`taichi-forge` fork）
- **JIT**：Numba
- **3D 渲染**：OpenGL（PyOpenGL）
- **数值计算**：NumPy、SciPy、mpmath
- **深度学习**：PyTorch（仅 Distill）
- **图像**：Pillow、matplotlib

## 项目结构

```text
├── main.py                  # 启动器：画廊 + 子进程管理
├── app_runner.py            # 单应用运行入口（配置 sys.path / cwd）
├── apps_manifest.py         # 应用清单：标题、描述、分类、依赖
├── apps/
│   ├── custom_import.py     # 公共导入中心
│   ├── custom_function.py   # Taichi 复数运算与自定义表达式
│   ├── utils.py             # OrbitCamera, ColorButton
│   ├── dynamic_fractal.py   # 分形画布基类
│   ├── reaction_diffusion.py# 反应扩散基类
│   ├── Math/ Physics/ Fractals/ Cellular Automata/
│   └── 2D *.py, Mass Spring.py
├── models/                  # Distill 预训练模型
├── images/                  # 卡片缩略图
├── tests/                   # 导入 / manifest 冒烟测试
├── style.qss                # 统一 QSS（Arnold 使用）
├── requirements.txt
├── requirements-distill.txt
└── pyproject.toml
```

## 主要特性

- **进程隔离**：每个应用一个子进程，Taichi/OpenGL/PyTorch 不会互相重置；单个应用崩溃不影响启动器
- **双精度 GPU 分形**：Mandelbrot/Julia 使用 Taichi `f64` 复数内核
- **交互式探索**：鼠标拖拽平移、滚轮缩放、实时参数调节
- **抗锯齿**：1x / 2x / 4x 超采样
- **颜色自定义**：ColorButton 原生颜色选择器
- **PNG 导出**：Fourier、Arnold、IFS、Lindenmayer、Qusicrystal 支持
- **JIT 加速**：Numba 编译热点路径，NumPy 批量渲染

## 开发与测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 公共模块可独立导入、无循环导入
- `apps_manifest.py` 中登记的应用文件都存在
- 应用路径/标题唯一、分类合法
- `app_runner.py` 存在

CI 见 `.github/workflows/ci.yml`：静态编译 + manifest 测试，以及在 Windows 上安装依赖后的导入冒烟测试。

### Lenia 预设自动搜索

`tools/search_lenia_presets.py` 默认使用**随机参数 + 模拟画笔随机作画**：每次独立采样参数，随机画点/直线/圆弧/随机游走/十字，再演化筛选；不会从现有预设做小扰动（需要变体时用 `--mode mutate`）。

```bash
# 只看看结果，不写文件
python tools/search_lenia_presets.py --trials 300 --dry-run

# 找到并保存（自动备份到 apps/preset_backups/）
python tools/search_lenia_presets.py --trials 300 --save
```

默认做 1200 步长程复核；本轮随机搜索另做了 3000 步复核，保存了 7 个“随机…”预设。当前预设库共 331 个。

### 经典 Lenia 形态

283 个“经典-…”预设全部解码自 Bert Chan 的 [Chakazul/Lenia](https://github.com/Chakazul/Lenia)（MIT）的 `Lenia-LifeForms.js`，覆盖文件里 13 个科中的 283/315 个形态：

| 科 | 数量 | 科 | 数量 |
|---|---|---|---|
| P 有翼科 Pterifera | 63 | R 放射科 Radidae | 22 |
| K 冠形科 Kronidae | 32 | D 齒形科 Dentidae | 21 |
| C 環形科 Circidae | 29 | F 瓣形科 Folidae | 17 |
| S 盾形科 Scutidae | 28 | A 星形科 Astridae | 15 |
| H 卷形科 Helicidae | 28 | G 雙子科 Geminidae | 12 |
| O 球形科 Orbidae | 9 | Q 方形科 Quadridae | 4 |
| V 輪形科 Volvidae | 3 | | |

筛选标准：350 步粗筛 + 3000 步复核（`mass>5`、`area<0.15`、`bbox<160`），失败的形态会再按 `omega`×`dt` 网格重调后重新复核。剩余 32 个形态在高斯近似下始终膨胀/死亡，未收录，例如 `V2 乙輪虫`、`S14/S24 大盾形`、`H8 辛卷虫`、`K16 庚冠虫`、`4F13 十三瓣虫` 等。

注：`quad4` 核与本项目的 Gaussian 核形状不同，部分形态只保留了“稳定有界”，形状会退化（例如旋鋸雙子虫变成圆环、大翼虫长成树枝状）。要完全还原原版形态需要给应用加 `kernel type` 选项。

第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已知限制

- 实验分类中的应用没有缩略图，界面与稳定性未完全打磨。
- 分形渲染仍在 GUI 线程同步计算；4x 抗锯齿 + 高迭代次数时缩放会短暂卡顿。
- `eval` / `exec` 用于用户自定义表达式，仅应输入本地可信内容，不构成安全沙箱。

## License

MIT，见 [LICENSE](LICENSE)。
