# 性能记录

本文件记录已实施的性能改动和实测结果。环境：Windows、Python 3.14、
PySide6 6.12、Taichi Forge 0.7、NVIDIA RTX 4060 Laptop。

## 启动与 CPU

| 应用/场景 | 改动 | 结果 |
|---|---|---:|
| Noise | `_clear_live` 幂等，重绘降到 1 Hz | 单核 98% → 32% |
| Diffusion | 模型加载移到后台线程 | 窗口 6.5 s → 0.11 s；引擎约 4.5 s 后台就绪 |
| Lenia | torch/scipy 懒加载 + 后台预热 | 启动 6.8 s → 3.5 s；256→1024 切换 2.7 s → 0.74 s |
| Maxwell | Numba 缓存按入口脚本分域 | 测试/冒烟后仍能启动 |

## Nonlinear

- `NLSE` / `TemporalNLSE` 增加 torch 后端；状态常驻 GPU。
- 空间模式 640²，与画布 1:1；GPU 调色板（分位 + LUT）。
- 启动时预热 cuFFT，运行期不切换后端。
- 实测：`space.step` 约 2.3–2.9 ms（NumPy 约 12.8 ms）；640² 完整 tick 约 14 ms。
- 无 torch 时回退 NumPy + 192²。

## MPM2D

- `MAX_SUBSTEPS` 40 → 32：同场景 30 帧平均 35.1 → 27.4 ms（约 1.28×），无 NaN。
- 砂粒 `p2g` 跳过无用的 SVD（等价，但无额外提速）。
- `draw()` 改为单层 GPU 软 splat + 逐粒子色差；600² 渲染约 1.5–2 ms，只影响显示。
- ndarray + `ti.graph` 原型：22.7 / 24.5 ms/帧，慢于原 field 实现（10.8–11.0 ms），未采用。
- kernel 合并在 Navier 上实测 6.45 → 10.44 ms，未采用。

## 结论

- 当前 Taichi Forge 下，MPM/Navier 的主要成本是 kernel 提交/同步；
  把 field 改成 ndarray 会拖慢 GPU 内核，graph 收益被抵消。
- 保持全速运行，不做 FPS 限制或低功耗降级。
- 更激进的优化需要改写 CUDA/native 内核或改变数值格式，属于另一量级改造。

## 验证

- `python -m unittest discover -s tests -v`：397 tests。
- `RUN_WINDOW_SMOKE=1`：41/41 应用窗口。
- CI 使用 Python 3.14；无 GPU runner 使用 `RUN_WINDOW_SMOKE=cpu`。
