# 项目约定

## UI 规范

新增或重构应用必须遵循 [docs/UI_GUIDELINES.md](docs/UI_GUIDELINES.md)，样板为：

- `apps/Math/Quasicrystal.py`
- `apps/Physics/Mass-Spring.py`
- `apps/Cellular Automata/Particle Lenia.py`
- 双视图样板：`apps/Cellular Automata/Distill.py`

硬性要点：

- 标准应用：窗口默认 960×640；左侧面板 330；画布固定 600×600。
- 双视图应用（两张并列图像，如 Distill）：标记 `UI_LAYOUT = "two_view"`，控制区放顶部，使用宽幅双画布；不套 330 左栏和 600 方画布，推荐 1040×740、最小 1000×700；两个预览各加 2px 中性色边框；像素网格用最近邻放大（SmoothPixmapTransform=False）。
- 左侧不放大标题、副标题、logo 或 emoji 图标；直接从 QGroupBox 开始。
- 选项开关用默认样式 `QCheckBox`，不要用长条形可勾选按钮（`QPushButton` + `setCheckable(True)`）。
- 画布纯白、无边框、无内缩、无左上角 HUD；图像铺满画布。
- 封面 520×520，只截画布图像/有效内容，不包含窗口或面板空白。
- 新增应用同步更新 `apps_manifest.py`、`runtime/launcher_layout.json`、README 和缩略图。
- 提交前运行 `python tools/check_ui_style.py <app路径>` 和 `python -m unittest discover -s tests`。
