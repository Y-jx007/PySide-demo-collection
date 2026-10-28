"""Single source of truth for the launcher gallery.

Each entry describes one standalone app.  The launcher runs every app in its own
process (see app_runner.py), so this module must stay import-safe: it only uses
the standard library.
"""

from __future__ import annotations

# Category display order.  Unknown categories are appended alphabetically.
CATEGORY_ORDER = ["Math", "Physics", "Fractals", "Cellular Automata", "Unpolished"]

# Optional runtime dependencies: module -> install hint.
INSTALL_HINTS = {
    "torch": "pip install -r requirements-distill.txt",
    "cplot": "pip install cplot",
    "sounddevice": "pip install sounddevice",
}

APPS = [
    # ---------------- Math ----------------
    {
        "title": "Fourier",
        "description": "绘制图形并查看其傅里叶变换",
        "path": "Math/Fourier.py",
        "friendly": "二维傅里叶变换",
        "category": "Math",
    },
    {
        "title": "Arnold",
        "description": "可视化图像猫变换",
        "path": "Math/Arnold.py",
        "friendly": "猫变换",
        "category": "Math",
    },
    {
        "title": "Lloyd",
        "description": "用 Voronoi 图的 Lloyd 松弛过程模拟蜂巢结构的形成",
        "path": "Math/Lloyd.py",
        "friendly": "Lloyd松弛",
        "category": "Math",
    },
    {
        "title": "幅角图",
        "description": "在复平面上可视化复变函数",
        "path": "Math/Cplot.py",
        "friendly": "辐角图",
        "category": "Math",
        "requires": ["cplot"],
    },
    {
        "title": "旅商问题",
        "description": "使用遗传算法解决旅行商问题",
        "path": "Math/Traveler.py",
        "friendly": "旅行商问题遗传算法",
        "category": "Math",
    },
    {
        "title": "Lorenz",
        "description": "三维奇异吸引子可视化",
        "path": "Math/Lorenz.py",
        "friendly": "洛伦兹吸引子",
        "category": "Math",
    },
    {
        "title": "Stigmergy",
        "description": "黏菌/蚁群的信息素寻路：个体留下痕迹，痕迹反过来引导个体",
        "path": "Math/Stigmergy.py",
        "friendly": "黏菌网络",
        "category": "Math",
    },
    # ---------------- Physics ----------------
    {
        "title": "Schrödinger",
        "description": "Runge-Kutta 模拟二维量子波包演化过程",
        "path": "Physics/Schrödinger.py",
        "friendly": "薛定谔方程",
        "category": "Physics",
    },
    {
        "title": "Turing",
        "description": "反应扩散系统模拟",
        "path": "Physics/Turing.py",
        "friendly": "反应扩散模型",
        "category": "Physics",
    },
    {
        "title": "Noise",
        "description": "环境噪声实时频谱、A 计权统计、1/f 斜率与长期规律",
        "path": "Physics/Noise.py",
        "friendly": "环境噪声分析",
        "category": "Unpolished",
        "requires": ["sounddevice", "matplotlib"],
    },
    {
        "title": "Snowflake",
        "description": "Reiter 雪花生长模型，六重对称枝晶",
        "path": "Fractals/Snowflake.py",
        "friendly": "雪花生长",
        "category": "Fractals",
    },
    {
        "title": "Beam",
        "description": "横梁承重：有限元算弯曲应力，超强度即断裂",
        "path": "Physics/Beam.py",
        "friendly": "横梁承重",
        "category": "Physics",
    },
    {
        "title": "Bremsstrahlung",
        "description": "推迟 Liénard-Wiechert 场：轫致分量（红）与同步分量（蓝）",
        "path": "Physics/Bremsstrahlung.py",
        "friendly": "轫致/同步辐射",
        "category": "Physics",
    },
    {
        "title": "Ising",
        "description": "模拟二维伊辛模型",
        "path": "Physics/Ising.py",
        "friendly": "二维伊辛模型",
        "category": "Physics",
    },
    {
        "title": "Maxwell",
        "description": "模拟气体分子速率分布",
        "path": "Physics/Maxwell.py",
        "friendly": "麦克斯韦分布律",
        "category": "Physics",
    },
    {
        "title": "Navier-Stokes",
        "description": "2D Q9 格子 Boltzmann 模拟卡门涡街",
        "path": "Physics/Navier-Stokes.py",
        "friendly": "Navier-Stokes方程",
        "category": "Physics",
    },
    # ---------------- Fractals ----------------
    {
        "title": "Quaternion Julia",
        "description": "三维 Julia 集可视化",
        "path": "Fractals/Quaternion Julia.py",
        "friendly": "三维Julia集",
        "category": "Fractals",
    },
    {
        "title": "Mandelbrot-Julia",
        "description": "复动力系统生成分形",
        "path": "Fractals/Mandelbrot.py",
        "friendly": "复动力系统",
        "category": "Fractals",
    },
    {
        "title": "Newton",
        "description": "牛顿迭代法生成分形",
        "path": "Fractals/Newton.py",
        "friendly": "牛顿迭代法分形",
        "category": "Fractals",
    },
    {
        "title": "Lindenmayer",
        "description": "林氏系统分形",
        "path": "Fractals/Lindenmayer.py",
        "friendly": "林氏系统",
        "category": "Fractals",
    },
    {
        "title": "IFS分形",
        "description": "迭代函数系统生成分形图案",
        "path": "Fractals/IFS.py",
        "friendly": "IFS分形",
        "category": "Fractals",
    },
    {
        "title": "Logistic分形",
        "description": "逻辑映射生成分形图案",
        "path": "Fractals/Logistic.py",
        "friendly": "逻辑映射分形",
        "category": "Fractals",
    },
    # ---------------- Cellular Automata ----------------
    {
        "title": "Lenia",
        "description": "连续化元胞自动机",
        "path": "Cellular Automata/Lenia.py",
        "friendly": "连续生命游戏",
        "category": "Cellular Automata",
    },
    {
        "title": "Conway",
        "description": "生命游戏，含 14 个经典预设与残影轨迹",
        "path": "Cellular Automata/Conway.py",
        "friendly": "元胞自动机",
        "category": "Cellular Automata",
    },
    {
        "title": "Greenberg-Hastings",
        "description": "激发介质元胞自动机：螺旋波、靶波与不应期",
        "path": "Unpolished/Greenberg-Hastings.py",
        "friendly": "激发介质",
        "category": "Unpolished",
    },
    {
        "title": "Distill",
        "description": "神经网络元胞自动机",
        "path": "Cellular Automata/Distill.py",
        "friendly": "神经网络元胞自动机",
        "category": "Cellular Automata",
        "requires": ["torch"],
    },
    # ---------------- Unpolished ----------------
    {
        "title": "Circulation",
        "description": "全球环流：解析气压 + 地形；真实模式看 2015–2024 月均风",
        "path": "Unpolished/Circulation.py",
        "friendly": "全球环流",
        "category": "Unpolished",
    },
    {
        "title": "Erosion",
        "description": "降雨侵蚀 + 两层土壤水/蒸散 + K_I/K_IC 龟裂",
        "path": "Unpolished/Erosion.py",
        "friendly": "水力侵蚀",
        "category": "Unpolished",
    },
    {
        "title": "Caustic",
        "description": "焦散/光线折射模拟",
        "path": "Unpolished/Caustic.py",
        "friendly": "焦散模拟",
        "category": "Unpolished",
    },
    {
        "title": "MPM2D",
        "description": "2D MPM 材料：固体 / 雪 / 液体三种本构，照 taichi_elements 实现",
        "path": "Unpolished/MPM2D.py",
        "friendly": "MPM 材料",
        "category": "Unpolished",
    },
    {
        "title": "2D Qusicrystal",
        "description": "二维准晶衍射图样",
        "path": "Unpolished/2D Qusicrystal.py",
        "friendly": "二维准晶",
        "category": "Unpolished",
    },
    {
        "title": "Mass Spring",
        "description": "质点-弹簧系统",
        "path": "Unpolished/Mass Spring.py",
        "friendly": "质点弹簧系统",
        "category": "Unpolished",
    },
    {
        "title": "Particle Lenia",
        "description": "粒子化 Lenia",
        "path": "Cellular Automata/Particle Lenia.py",
        "friendly": "粒子化Lenia",
        "category": "Unpolished",
    },
    {
        "title": "KAM",
        "description": "KAM 环面与标准映射",
        "path": "Physics/KAM.py",
        "friendly": "KAM环面",
        "category": "Unpolished",
    },
]
