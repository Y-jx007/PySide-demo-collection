# PySide 绘图工具集 — 数学与物理可视化

一个基于 PySide6 + Taichi + OpenGL 的交互式数学/物理可视化工具集
## 界面预览

![界面预览](<img width="1312" height="660" alt="image" src="https://github.com/user-attachments/assets/fb616f96-f595-4a8b-a39e-0ad719691986" />
)

## 功能列表

### 分形与复数
| 应用 | 说明 |
|------|------|
| **Mandelbrot / Julia 集** |  |
| **Newton / Nova 分形** | 牛顿分形 |
| **2D IFS** | 迭代函数系统分形 |
| **L-System** | 林氏系统分形 |

### 反应扩散与物理模拟
| 应用 | 说明 |
|------|------|
| **Ising 模型** | 2D 伊辛模型，参考项目[Ising.js by matthbierbaum](https://mattbierbaum.github.io/ising.js/)|
| **Belousov-Zhabotinsky** | 化学波 |
| **Turing 斑图** | 图灵斑 |
| **Navier-Stokes** | 2D 不可压流体，参考项目[LBM_Taichi by hietwll](https://github.com/hietwll/LBM_Taichi) |
| **Schrodinger** | 2D 量子力学波包演化，参考项目[Schrodinger equation by davidar](https://www.shadertoy.com/view/lsKGRW) |
| **Maxwell** | 2D 麦克斯韦-玻尔兹曼分布 |

### 细胞自动机与群体
| 应用 | 说明 |
|------|------|
| **Conway Game of Life** | 生命游戏 |
| **Lenia** | 连续细胞自动机，参考项目[Lenia by Chakazul](https://chakazul.github.io/Lenia/JavaScript/Lenia.html)|
| **Distill** | 神经元胞自动机，参考项目[Growing Neural Cellular Automata by Distill](https://distill.pub/2020/growing-ca/) |

### 其他
| 应用 | 说明 |
|------|------|
| **Lloyd 松弛** | |
| **3D Lorenz** | 3D吸引子可视化|
| **Cplot** | 复变函数幅角图，使用工具[Cplot by nschloe](https://github.com/nschloe/cplot) |
| **Fourier 合成** | 2D 傅里叶变换 |
| **Arnold** | 猫变换 |

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行

启动主界面（列出所有应用）：

```bash
python main.py
```

或直接运行单个应用：

```bash
python "apps/2D Mandelbrot.py"
python "apps/2D IFS.py"
```

## 技术栈

- **GUI 框架**: PySide6 (Qt for Python)
- **GPU 计算**: Taichi (taichi-forge)
- **JIT 加速**: Numba
- **3D 渲染**: OpenGL (PyOpenGL + GLUT)
- **数值计算**: NumPy, SciPy, mpmath
- **深度学习**: PyTorch（仅 Distill 应用需要）
- **图像**: Pillow, matplotlib

## 项目结构

```
├── main.py                  # 主启动入口
├── apps/
│   ├── custom_import.py     # 公共导入中心
│   ├── custom_function.py   # Taichi 复数运算函数
│   ├── utils.py             # OrbitCamera, ColorButton 等
│   ├── reaction_diffusion.py # 反应扩散基类
│   ├── dynamic_fractal.py   # 分形基类
│   ├── 2D Mandelbrot.py     # Mandelbrot/Julia 集
│   ├── 2D Newton.py         # Newton/Nova 分形
│   ├── 2D IFS.py            # 迭代函数系统
│   ├── ...                  # 其他 20+ 应用
├── models/                  # 预训练模型
├── images/                  # 截图
├── style.qss               # Qt 样式表
├── requirements.txt         # 依赖清单
└── tests/
    └── test_imports.py      # 冒烟测试
```

## 主要特性

- **双精度 GPU 加速**: 分形应用使用 Taichi ti.f64 双精度复数计算
- **交互式探索**: 鼠标拖拽平移、滚轮缩放、实时参数调节
- **功能热切换**: 预设函数 / 自定义 Taichi 函数即时切换
- **抗锯齿渲染**: 支持 1x/2x/4x 超采样抗锯齿
- **颜色自定义**: ColorButton 原生颜色选择器
- **图像保存**: 所有应用支持 PNG 导出
- **性能优化**: Numba JIT 编译热点路径，numpy 批量渲染替代逐点绘制

## 开发说明

- 所有 app 文件通过 from custom_import import * 获取公共依赖
- 内部模块独立导入，不存在循环依赖
- 分形应用继承 BaseFractalWidget，反应扩散应用继承 SimulationBase
- Taichi 内核使用 @ti.kernel + ti.ndrange 并行计算

## License

MIT
