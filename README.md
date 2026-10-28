# PySide 绘图工具集 — 数学与物理可视化

一个基于 **PySide6 + Taichi + OpenGL** 的交互式数学/物理可视化工具集，包含 34 个可独立运行的应用。

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
| **Stigmergy** | 间接通信（stigmergy）两种模型：**黏菌**用三向嗅探 + 信息素沉积自发形成静脉状输运网络；**蚁群**有巢穴与食物，蚂蚁在觅食/返程两态间切换，沿途留下两种信息素，最终收敛出通往食物的蚁道。左键拖拽投放食物、右键擦除，Shift+左键设置蚁巢 |

### Physics

| 应用 | 说明 |
|------|------|
| **Schrödinger** | Runge-Kutta 模拟二维量子波包演化 |
| **Turing** | Gray-Scott 等反应扩散系统 |
| **Beam** | 横梁承重：欧拉-伯努利梁有限元，应力超材料强度即断裂，可扫描承重与梁高/梁长的关系 |
| **Ising** | 二维伊辛模型，参考 [Ising.js](https://mattbierbaum.github.io/ising.js/) |
| **Maxwell** | 二维麦克斯韦-玻尔兹曼速率分布 |
| **Navier-Stokes** | 2D Q9 格子 Boltzmann 模拟卡门涡街，参考 [LBM_Taichi](https://github.com/hietwll/LBM_Taichi) |
| **Bremsstrahlung** | 变速圆周电子：真实推迟 Liénard-Wiechert 辐射场，按加速度平行/垂直分量显示轫致（红）与同步（蓝），远场 FFT 频谱加色叠加 |

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
| **Conway** | 生命游戏，14 个经典预设（滑翔机枪、脉冲星等）、残影轨迹、滚轮缩放 |
| **Greenberg-Hastings** | 激发介质元胞自动机：静息/激发/不应期三态，支持螺旋波、靶波、随机激发和点击刺激；正方形格点的 Moore 邻域会形成方形波前 |
| **Distill** | 神经网络元胞自动机，参考 [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)，需要 PyTorch |
| **Snowflake** | Reiter 雪花生长模型，六重对称枝晶，参考 [Reiter 2005](http://www.patarnott.com/pdf/SnowCrystalGrowth.pdf) |

### Unpolished（未打磨）

| 应用 | 说明 |
|------|------|
| **Circulation** | 全球环流：合成模式用解析气压 + 地转风 + ETOPO1 地形；真实模式用 NCEP/NCAR 2015–2024 月平均 10 m 风，可看月均、月际 Δ 与十年趋势 |
| **Erosion** | 小尺度平面泥地；两层土壤水（入渗/径流/下渗/蒸散）+ Voronoi 多边形 cohesive 网络干裂，黏粒/气温/湿度/风速/韧性可调 |
| **Caustic** | 焦散/光线折射模拟 |
| **MPM2D** | 2D MLS-MPM 材料演示：固体（弹性）/ 雪（塑性）/ 液体（J 驱动压力），本构与参数照 taichi_elements 的 mpm_solver；鼠标可倾倒材料 |
| **Noise** | 环境噪声：A 计权声级、L10/L50/L90、1/f 斜率、小时/长期历史（需要 sounddevice + matplotlib） |
| **2D Qusicrystal** | 二维准晶衍射图样 |
| **Mass Spring** | 质点-弹簧系统 |
| **Particle Lenia** | 粒子化 Lenia |
| **KAM** | KAM 环面与标准映射 |

## Noise

Noise 用麦克风实时统计环境噪声：

- A 计权声级、10 s Leq、L10/L50/L90、峰值/波峰因数、主频；
- 50–2000 Hz 频谱对数斜率 α（0 白、1 粉红/1/f、2 红/棕），并叠加拟合线与 1/f 参考线；
- 时域波形、A 计权频谱、声级历史、声级分布、1/f–能量密度图；
- 1/f–能量密度图：横轴是线性的 1/f（ms），纵轴是**未计权**能量密度（FS²/Hz），默认取 30 s 时间平均（可选 5/10/30/60 s）；对 S(f)=C/f，令 x=1/f 得到 S=Cx，应呈过原点的直线。上一版误用了 A 计权谱，低频被衰减，才会看起来随 1/f 下降；
- 每秒把一条聚合记录（含 α）写入 runtime/logs/noise_history.sqlite，并显示最近 7 天的小时均值与 7 d α；
- 校准滑杆 -30…+30 dB，默认满幅按 94 dB 估算；
- 演示模式默认只勾选 1/f；1/f^0.5、1/f、1/f^1.5、1/f²、白噪声、180 Hz 瞬态都是独立可选项，可任意组合；全都不勾选或无输入信号时清空实时统计，不再把数值 epsilon 底噪显示成能量；演示数据不写入历史。无输入设备或设置 NOISE_DEMO=1 时自动进入演示模式。

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

主界面支持直接拖动卡片调整顺序和分类：跨分类拖动时，应用文件会一起挪到 `apps/<新分类>/`，
并同步改写 `apps_manifest.py` 里的 `path` / `category`（应用正在运行或目标处有同名文件时会拒绝这次拖动）。
卡片顺序与所在分类保存在 `runtime/launcher_layout.json`（已加入 `.gitignore`）。

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
├── common/                  # 共享导入与基础模块
│   ├── paths.py             # 目录约定：assets/ runtime/ 的唯一入口
│   ├── qt_base.py           # Qt + 标准库（再导出 widgets / app / workers）
│   ├── numpy_base.py        # NumPy + Pillow（再导出 imaging 助手）
│   ├── widgets.py           # LabeledSlider、分组框/表单、分隔线、正方形容器
│   ├── imaging.py           # ndarray ↔ QImage、ImageLabel、打开/保存对话框
│   ├── field_canvas.py      # ArrayCanvas：等比绘制的 numpy 图像画布
│   ├── mpl_canvas.py        # MplCanvas：matplotlib 画布 + Gcf 登记 + 工具栏
│   ├── app.py               # QApplication / QSS / QSurfaceFormat / run_app
│   ├── workers.py           # StoppableThread：可协作停止的后台线程
│   ├── utils.py             # OrbitCamera, ColorButton, FPSCounter
│   ├── taichi_base.py       # Taichi
│   ├── numba_base.py        # Numba（缺依赖时自动降级为纯 Python）
│   ├── opengl_base.py       # OpenGL
│   ├── mp_base.py           # mpmath
│   ├── custom_base.py       # custom_function / dynamic_fractal
│   ├── custom_function.py   # Taichi 复数运算与自定义表达式
│   ├── dynamic_fractal.py   # 分形画布基类 + 视图坐标变换
│   └── reaction_diffusion.py# 反应扩散基类
├── apps/                    # 只放应用入口（清单里的每个条目一个 .py）
│   ├── Math/                # Arnold / Cplot / Fourier / Lloyd / Lorenz / Stigmergy / Traveler
│   ├── Physics/             # Beam / Bremsstrahlung / Ising / KAM / Maxwell / Navier-Stokes /
│   │                        #   Noise / Schrödinger / Turing
│   ├── Fractals/            # IFS / Lindenmayer / Logistic / Mandelbrot / Newton /
│   │                        #   Quaternion Julia / Snowflake
│   ├── Cellular Automata/   # Conway / Distill / Greenberg-Hastings / Lenia / Particle Lenia
│   └── Unpolished/          # 2D Qusicrystal / Caustic / Circulation / Erosion / MPM2D /
│                            #   Greenberg-Hastings / Mass Spring
├── support/                 # 应用专用支撑模块（按分类分目录，供 app 与测试导入）
│   ├── physics/             # circulation_data / lienard_wiechert / lw_taichi /
│   │                        #   noise_dsp / noise_history / soil_water
│   ├── math/                # cplot_parallel / cplot_fast_contour
│   ├── fractals/            # fractal_common（双画布窗口、轨道叠加）/ fractal_kernels
│   └── cellular_automata/   # greenberg_hastings_model
├── assets/                  # 资源目录（应用启动时读取；Lenia 保存预设会回写 data/）
│   ├── data/                # worldelev / wind_real / worldmask / czm_* / lenia_presets
│   ├── presets/             # turing/fk_map.png、snowflake_alpha/、snowflake_terrain/
│   ├── models/              # Distill 预训练 .pth
│   ├── thumbnails/          # 启动器卡片缩略图
│   └── style.qss            # 统一 QSS（可选：run_app(stylesheet=True) 时自动加载）
├── runtime/                 # 运行期产物（不入库）：logs/、launcher_layout.json、imgui.ini
├── tools/                   # 数据重建、Lenia 预设搜索、CZM bank
├── tests/                   # 导入 / manifest / 数据冒烟测试
├── requirements.txt
├── requirements-distill.txt
└── pyproject.toml
```

## 公共组件（common/）

各 app 原先都在自己文件里重写同一批样板代码，现已抽到 common/ 下。
Math 7 个、Physics 13 个、Fractals 6 个、Cellular Automata 6 个应用都已改用统一的
`run_app` 入口（`Mass Spring.py` 用的是 Taichi 自带 GUI，不涉及 PySide）。
画布层也统一了：matplotlib 画布（`MplCanvas`）覆盖 Fourier / Cplot / Noise / Lenia 核预览，
numpy 图像画布（`ArrayCanvas`）覆盖 Erosion / MPM2D / Circulation / Conway / Snowflake /
Greenberg-Hastings。其余替换按需推进（`ndarray_to_qimage` 覆盖 Erosion / MPM2D /
Bremsstrahlung / Caustic / Circulation；`LabeledSlider` 覆盖 Beam / Erosion / Schrödinger /
MPM2D / Bremsstrahlung / Navier-Stokes / KAM / Noise / Circulation / Quaternion Julia /
Conway / Snowflake / Greenberg-Hastings / Particle Lenia / Distill，共 60 余个滑块；
`FPSCounter` 覆盖 Erosion / Schrödinger / Conway，`StoppableThread` 覆盖 KAM / Distill）：

| 组件 | 位置 | 替代掉的重复代码 |
| --- | --- | --- |
| `run_app` / `setup_gl_format` | `common/app.py` | `__main__` 里的 QApplication + QSS + QSurfaceFormat 样板 |
| `LabeledSlider` | `common/widgets.py` | `_create_slider_pair` / `_add_slider` 等手写滑块行 |
| `spinbox` / `double_spinbox` | `common/widgets.py` | 每处都写的 setRange / setValue / setSingleStep |
| `group_box` / `form_group` / `hbox` / `vbox` / `caption` / `separator` | `common/widgets.py` | QGroupBox + layout + addWidget 的样板 |
| `SquareContainer` | `common/widgets.py` | Lorenz 里的 SquareGLContainer |
| `ImageLabel` / `ndarray_to_qimage` / `ndarray_to_qpixmap` | `common/imaging.py` | Arnold 的 display_image + resizeEvent 重绘 |
| `ask_open_image` / `ask_save_path` / `save_ndarray` | `common/imaging.py` | 文件对话框 + 补后缀 + Pillow 落盘 |
| `StoppableThread` | `common/workers.py` | CatMapWorker / TSPWorker 各自维护的 stop 标志与收尾 |
| `FPSCounter` | `common/utils.py` | Stigmergy 手写的帧率统计 |
| `MplCanvas` | `common/mpl_canvas.py` | Fourier / Cplot / Noise / Lenia 各自的 Figure + FigureCanvas（Cplot 的 Gcf 登记） |
| `ArrayCanvas` | `common/field_canvas.py` | Erosion / MPM2D / Circulation / Conway / Snowflake / Greenberg-Hastings 的居中绘制与鼠标坐标换算 |

新的 app 只需要两行入口：

```python
if __name__ == "__main__":
    raise SystemExit(run_app(MainWindow, gl=True))   # gl=True 时自动设置 QSurfaceFormat
```

## 主要特性

- **进程隔离**：每个应用一个子进程，Taichi/OpenGL/PyTorch 不会互相重置；单个应用崩溃不影响启动器
- **双精度 GPU 分形**：Mandelbrot/Julia 使用 Taichi `f64` 复数内核
- **交互式探索**：鼠标拖拽平移、滚轮缩放、实时参数调节
- **地形 / 海拔**：ETOPO1 1° 起伏用于底图阴影、海拔配色，以及合成风场的减速与绕流
- **合成气压场**：解析式海平面气压 + 地转风；地形用于底图阴影、减速与绕流。合成场是教学/对照模型，南极陆地保持旧版的低压形态
- **真实气候数据**：NCEP/NCAR Reanalysis 1 的 2015–2024 月平均 10 m 风，支持逐月播放、同月 Δ 与十年趋势
- **土壤水与断裂**：Erosion 用两层水量平衡（Green-Ampt 入渗、Penman-Monteith 蒸散、径流/下渗）驱动湿润；CZM 裂缝由 Voronoi 多边形 cohesive 网络离线计算，按土壤收缩率选帧
- **抗锯齿**：1x / 2x / 4x 超采样
- **颜色自定义**：ColorButton 原生颜色选择器
- **PNG 导出**：Fourier、Arnold、IFS、Lindenmayer、Qusicrystal 支持
- **JIT 加速**：Numba 编译热点路径，NumPy 批量渲染

### Circulation 的地形与真实风场数据

- **地形**：`assets/data/worldelev.npz` 是 ETOPO1（NOAA NGDC）的 1° 抽样，用于底图海拔配色、坡向阴影，以及合成模式的减速/绕流修正。界面里的“地形影响”可在 0–200% 调节；真实风场本身已包含地形效应，因此该滑杆只作用于合成模式。
- **海岸线**：`worldmask.npz` 为 Natural Earth 720×360 海陆掩膜。
- **说明**：合成场是旧版解析气压（不含温度），南极陆地是低压；真实月平均风场请用“真实模式”，它来自完整原始方程 GCM 的再分析产品。
- **真实风场**：`assets/data/wind_real.npz` 来自 NCEP/NCAR Reanalysis 1（观测同化的再分析产品）的月平均 10 m 风，2015–2024 共 120 个月，插值到 2.5° 网格。界面可选年份和月份，逐月播放；“月际变化 Δ”是相对 2015–2024 同月平均的 Δ 风；“十年趋势”是年均值的线性拟合（m/s per decade）。10 个年均值上的趋势只用于展示，不用于气候归因。
- **重建数据**：`python tools/build_circulation_data.py`。数据源为 NOAA ERDDAP（ETOPO1）和 NOAA PSL NetCDF Subset Service（NCEP/NCAR），无需 API key。来源与引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### Erosion 的 Voronoi CZM 场

assets/data/czm_fields.npz 保存 4 组独立 Voronoi 多边形 cohesive 网络的 9 级裂缝场（4×9×320×320）。Erosion 的“CZM 裂缝（离线）”开关按当前土壤水平均收缩率在帧间插值；干燥步仍由 SoilWater 驱动。

    python tools/build_czm_bank.py

只保留运行所需文件：

- tools/cohesive_voronoi_desiccation.py：Voronoi 多边形 cohesive 求解器；
- tools/build_czm_bank.py：生成多组裂缝场；
- apps/Physics/czm_fields.npz：预计算裂缝场；
- support/physics/soil_water.py：两层土壤水。

原先的 1D/2D patch test、单缺口、多单元缺口、Q4 网络等验证脚本已删除；它们只用于开发期验证，不参与应用启动。

### CZM 图案随机化与降雨滞后

Erosion 启动时从 4 组图案中随机选一组；裂缝完全闭合后开启新一轮干燥时再换一组，避免每次干裂同一张图。

CZM 帧选择带一阶滞后：张开 τ≈240 min、闭合 τ≈900 min；降雨时裂缝逐渐变浅，不会一帧消失。

### 启动与导入优化

- 启动器卡片缩略图改为首次绘制时用 QImageReader 加载，去掉每卡 QGraphicsDropShadowEffect；
- 每个 app 的 wildcard 导入已删除，改为按需导入：
  - qt_base：Qt + 标准库 + utils；
  - numpy_base：NumPy + Pillow；
  - taichi_base：Taichi；numba_base：Numba；opengl_base：OpenGL；mp_base：mpmath；custom_base：custom_function / dynamic_fractal。
- 纯 Qt 应用不再导入 Taichi/OpenGL；所有 app 的公共冒烟测试仍通过。
- 所有 ti.init 已启用 offline_cache；Cplot 的 cplot/matplotlib 改为首次绘图时导入。

## 开发与测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 公共模块可独立导入、无循环导入
- `apps_manifest.py` 中登记的应用文件都存在
- 应用路径/标题唯一、分类合法
- `app_runner.py` 存在
- Circulation 的 ETOPO1 海拔、2015–2024 逐月真实风场，以及旧合成气压场的南极低压回归
- Erosion 的两层土壤水质量平衡、Voronoi CZM 场与降雨闭合滞后

CI 见 `.github/workflows/ci.yml`：静态编译 + manifest 测试，以及在 Windows 上安装依赖后的导入冒烟测试。

### Lenia 预设自动搜索

`tools/search_lenia_presets.py` 默认使用**随机参数 + 模拟画笔随机作画**：每次独立采样参数，随机画点/直线/圆弧/随机游走/十字，再演化筛选；不会从现有预设做小扰动（需要变体时用 `--mode mutate`）。

```bash
# 只看看结果，不写文件
python tools/search_lenia_presets.py --trials 300 --dry-run

# 找到并保存（自动备份到 assets/data/preset_backups/）
python tools/search_lenia_presets.py --trials 300 --save
```

默认做 1200 步长程复核；本轮随机搜索另做了 3000 步复核，保存了 7 个“随机…”预设。当前预设库共 340 个。

预设画廊分两个标签页：**自定义**（自己保存的和随机搜索得到的）与**经典**（从原版移植的形态）。自定义页按环数分组，经典页按 13 个科分组并显示各科数量；每行最多 6 个 100px 方格，缩略图按内容做正方形裁切。关闭时会记住标签页、筛选条件、滚动位置和选中项（`QSettings`，键名 `PySideTools/LeniaPresets`），下次打开回到原处。

### 核函数与生长函数（原版兼容）

`Lenia.py` 现在实现了 Bert Chan 原版的核与生长函数，可在界面里选择：

| 类型 | 选项 | 公式 |
|---|---|---|
| 生长 | `gaus` | G(u) = 2·exp(-(u-μ)²/(2σ²)) - 1 |
| 生长 | `quad4` | G(u) = 2·(1-(u-μ)²/(9σ²))⁴ - 1，|u-μ|>3σ 时 -1 |
| 生长 | `trap` / `stpz` | 梯形 / 阶跃 |
| 核 | `bump4` | exp(α - α/(4r(1-r)))，α=4 |
| 核 | `quad4` | (4r(1-r))⁴ |
| 核 | `stpz1/4` / `trap1/5` / `life` | 阶跃 / 梯形 / Life 型 |
| 核 | `bell(旧)` | 本项目早期的高斯环 β·exp(-(frac-ρ)²/(2ω²))，保留用于兼容旧预设 |

多环形态按原版的 `KernelFunc` 叠加：`K(r) = CoreFunc(frac)·β[ring]`。预设格式扩展为 16 个参数（末尾追加 `growth_type`、`kernel_type`），旧的 14 参数预设加载时自动补 0（即高斯环 + gaus，行为不变）。

### 经典 Lenia 形态

292 个“经典-…”预设全部解码自 Bert Chan 的 [Chakazul/Lenia](https://github.com/Chakazul/Lenia)（MIT）的 `Lenia-LifeForms.js`（已归档在 `tools/`）：

| 科 | 数量 | 科 | 数量 |
|---|---|---|---|
| P 有翼科 Pterifera | 65 | R 放射科 Radidae | 22 |
| K 冠形科 Kronidae | 35 | D 齒形科 Dentidae | 21 |
| C 環形科 Circidae | 29 | F 瓣形科 Folidae | 18 |
| H 卷形科 Helicidae | 29 | A 星形科 Astridae | 15 |
| S 盾形科 Scutidae | 29 | G 雙子科 Geminidae | 12 |
| O 球形科 Orbidae | 9 | Q 方形科 Quadridae | 4 |
| V 輪形科 Volvidae | 4 | | |

其中 **266 个用原版 `quad4`/`bump4` 核跑通 3000 步**（形状与原版一致）。判定口径：质量在 1500→3000 步之间保持稳定（比值 0.9–1.1）、面积 < 6%、且最大连通分量占 60% 以上——早期用 `bbox` 判据会把跨环形边界的形态误判为爆炸。

剩余 26 个在原版核下仍会膨胀/死亡（`S14/S24 大盾形`、`P24 二十四翼虫`、`HN+/PN+ 链状`、`H8 辛卷虫`、`4F13 十三瓣虫` 等），保留此前用高斯环核验证过的稳定版本作为兜底；界面上切换核类型即可看到差别。

第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已知限制

- Unpolished 分类中的应用默认不配缩略图（Noise 例外，保留界面预览），界面与稳定性未完全打磨。
- 分形渲染仍在 GUI 线程同步计算；4x 抗锯齿 + 高迭代次数时缩放会短暂卡顿。
- `eval` / `exec` 用于用户自定义表达式，仅应输入本地可信内容，不构成安全沙箱。

## License

MIT，见 [LICENSE](LICENSE)。