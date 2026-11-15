# UI 规范（新应用 / 重构应用必须遵循）

本规范归纳当前统一风格，样板为：

- `apps/Math/Quasicrystal.py`
- `apps/Physics/Mass-Spring.py`
- `apps/Cellular Automata/Particle Lenia.py`
- 双视图样板：`apps/Cellular Automata/Distill.py`（顶部控制区 + 宽幅双画布）
- 参考同类简洁画布：`apps/Physics/Bremsstrahlung.py`

旧应用允许暂时保留历史样式；**新增或重构应用必须满足本规范**。

## 0. 双视图应用例外（two_view）

当应用的主内容必须是**两张并列图像**时，不套用“330 左栏 + 600 正方形画布”：

- 源码标记：`UI_LAYOUT = "two_view"`，检查器会自动切换到双视图规则。
- 控制区放在**窗口顶部**，使用横向 `QGroupBox`；图像双视图放在下方。
- 不使用 `setFixedWidth(330)`，也不使用 `setFixedSize(600, 600)`。
- 推荐窗口 **1040 × 740**，最小 **1000 × 700**；两个预览保持等大正方形、间隔 6px，并且每个预览要有 **2px 中性色边框**。
- 神经 CA 等像素网格必须最近邻放大：设置 `QPainter.SmoothPixmapTransform, False`，不要用双线性平滑。
- 仍要求纯白背景、无标题/副标题、无 emoji 图标、状态栏、520×520 封面。
- 样板：`apps/Cellular Automata/Distill.py`。

## 1. 窗口与整体布局

- 初始窗口固定为 **960 × 640**：
  - `setMinimumSize(960, 640)`
  - `resize(960, 640)`
  - 不再使用 1180 × 780 之类的默认大窗口。
- 窗口背景纯白：`background-color: #ffffff`。
- 水平布局：
  - 左侧参数面板固定 **330 px**：`panel.setFixedWidth(330)`
  - 右侧图像画布固定 **600 × 600**：`canvas.setFixedSize(600, 600)`
  - 外边距 8，面板与画布间距 8，画布居中。
- 最终窗口应刚好容纳：左侧 330 + 画布 600 + 边距/状态栏。
- 状态栏：
  - 左侧显示运行状态（运行中 / 暂停 / 已保存）；
  - 右侧显示关键读数，如 `N 900 · 41 fps`。

## 2. 左侧参数面板

- 顶部**不放标题、不放副标题、不放 logo / icon**。面板直接从第一个 `QGroupBox` 开始。
- 禁止使用 `title = QLabel(...)`、`hint = QLabel(...)` 作为页头。
- 禁止 emoji / 符号图标按钮，例如 `▶ 动画`、`⏸ 暂停`；使用纯文字 `动画 / 暂停`、`保存`、`重置`。
- 使用 `QGroupBox` 分组，组内用 `QVBoxLayout` / `QHBoxLayout`：
  - 组内边距 6，控件间距 5；
  - 数值参数用 `LabeledSlider`；
  - 下拉选择用 `QComboBox`；
  - 按钮使用默认 `QPushButton`，不要重 QSS 皮肤；
  - 开关类选项直接用默认样式的 `QCheckBox`，不要做成按钮。
- **选项开关不要用按钮**：设置类开关（固定基点 / 初始向量 / 网格 / 自动旋转 …）一律用默认样式 `QCheckBox`，
  不要用 `QPushButton(...).setCheckable(True)` 做成横贯整行的长条按钮来切换选项。
  按钮只承担动作（重置 / 保存 / 动画 / 暂停 / 运行）；动作按钮要表示状态时用文字切换即可，不必勾选。
  需要多选一用 `QComboBox`，一行按钮只放同类动作。
- 推荐分组顺序：
  1. 物理 / 结构 / 模型参数；
  2. 显示或系统参数；
  3. 控制（暂停、重置、保存）。
- 标签宽度要足够，不能把“衍射缩放”截成“衍射缩”；中文标签建议 56–64 px。
- 颜色只用项目默认控件色 + 必要状态色；不要给单个页面另造整套配色。

## 3. 右侧画布与图像

- 画布背景纯白；**不要深色画布 + 白色窗口的割裂感**。
- 画布固定 600 × 600；图像/内容必须**铺满整个画布**。
- 不要内缩边距，不要画外框，不要额外留白：
  - 不要 `side = min(width, height) - 64` 之类的内缩；
  - 不要 `drawRect(...)` 画边框；
  - 不要在画布左上角画标题、参数或 HUD 文字。
- 图像绘制直接使用：
  - `painter.fillRect(self.rect(), QColor(255, 255, 255))`
  - `painter.drawImage(self.rect(), image)`
- 若内容本身较稀疏，可在封面阶段裁到内容；但应用内画布仍保持 600 × 600 的完整画幅。
- 白色背景上的点、线、粒子要使用足够深的颜色（深蓝、深灰、红棕等），保证对比度。
- 如果某个科学可视化必须使用深色场（历史特例），必须在 PR/说明中写清理由；新页面默认白色。

## 4. 封面缩略图

- 尺寸固定 **520 × 520**。
- **只能截取画布本身的图像内容**，不能包含：
  - 应用窗口；
  - 左侧参数面板；
  - 窗口外空白；
  - 画布外的白边/黑边。
- 稀疏图像（如弹簧桥、粒子团）应裁到有效内容，不要让缩略图大部分是空白。
- 文件名必须是应用模块文件名去掉 `.py` 后加 `.png`：
  - `Quasicrystal.py` → `assets/thumbnails/Quasicrystal.png`
  - `Mass-Spring.py` → `assets/thumbnails/Mass-Spring.png`
  - `Particle Lenia.py` → `assets/thumbnails/Particle Lenia.png`
- 启动器在启动时加载封面；更新封面后需要重启启动器。
- **实验田（`Unpolished`）** 里的应用可以不配封面；装了封面就按上面的规则截。

## 5. 实验田（Unpolished）

`Unpolished` 是一个**分类意义上的实验田**，不是“没写完”的垃圾桶。放进去的条件是下面任意一条：

- 依赖额外数据或权重（Circulation 的 ETOPO1/风场、Erosion 的地形数据）；
- 依赖外部设备或可选依赖（Noise 需要 `sounddevice`）；
- 接口、参数或界面仍在改（KAM、MPM2D、Caustic）。

实验田的约定：

- 分类标记与说明只写在 `apps_manifest.CATEGORY_NOTES` 一处，启动器分类条、`docs/APP_MATRIX.md`、README 共用；
- 封面可省略；**但窗口结构、`check_ui_style.py`、测试仍然必须通过**，不能拿“实验田”当不守规范的借口；
- 需要额外依赖时，必须在清单条目里写 `requires`，启动器据此提示安装；
- 打磨到“不依赖外部数据、接口稳定”之后，把它挪进 Math / Physics / 对应分类，并从 `CATEGORY_NOTES` 的语境里移出。

## 6. 交互与状态

- 参数滑块改动后立即生效，不要求用户点“应用”。
- 必须有：
  - 暂停 / 继续；
  - 重置；
  - 保存当前画布为 PNG。
- 运行类应用用 `QTimer`，关闭窗口时停止定时器。
- 状态栏右侧持续显示规模与帧率（或等价读数）。
- 错误只在保存失败等必要场景弹窗；不要用弹窗代替状态栏。

## 7. 代码组织

用 `python tools/new_app.py <分类>/<名称> --title <标题> --description <说明>` 生成骨架：
它会写出合规的 app 文件、插入 `apps_manifest.py` 条目、重新生成 `docs/APP_MATRIX.md`，
并离屏渲染 520×520 封面；生成的文件当场就能通过 `tools/check_ui_style.py`。

- 新应用使用 `common.lab_window` 的 `LabWindow` / `SquareCanvas` 基类，统一白底、960×640 窗口和 600×600 画布。
- 应用入口放在 `apps/<Category>/<Name>.py`，一个入口一个文件。
- 用 `raise SystemExit(run_app(MainWindow))` 启动。
- 物理/模型代码放 `support/`，应用文件只负责 UI、定时器和交互。
- Taichi 模块：
  - 使用 `common.taichi_base.ensure_taichi()`；
  - 不要在多个模块里重复 `ti.init()`；
  - 内核模块不要写 `from __future__ import annotations`。
- 新应用同步更新：
  - `apps_manifest.py`；
  - `runtime/launcher_layout.json`；
  - `README.md`；
  - `assets/thumbnails/`；
  - 对应测试。

## 8. 提交前检查清单

- [ ] 窗口 960 × 640，无大默认窗口。
- [ ] 左栏 330，右画布 600 × 600。
- [ ] 左栏无标题、副标题、logo、emoji 图标。
- [ ] 画布纯白、无边框、无内缩、无左上角文字。
- [ ] 图像/内容铺满画布。
- [ ] 状态栏有状态 + 规模/fps。
- [ ] 暂停 / 重置 / 保存齐全。
- [ ] 开关类选项用默认 `QCheckBox`，没有整行的可勾选按钮。
- [ ] 封面 520 × 520，只含图像内容。
- [ ] manifest / layout / README / 缩略图已更新。
- [ ] `python -m unittest discover -s tests` 通过。
- [ ] `python tools/check_ui_style.py <app路径>` 无报错。

## 9. 自动检查

样板应用会由 `tests/test_ui_guidelines.py` 检查以下硬性结构：

- 左栏 330；
- 画布 600 × 600；
- 窗口 960 × 640；
- 白色背景；
- `QGroupBox` / `LabeledSlider`；
- 无左栏标题/副标题/符号图标；
- 封面存在且为 520 × 520。

新应用可用：

```powershell
python tools/check_ui_style.py apps/Math/Quasicrystal.py
```

自检。
