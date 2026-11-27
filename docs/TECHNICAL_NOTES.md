# 技术札记

README 只保留项目级信息，这里放各应用的实现细节和数据来源。

## Kerr–Newman 黑洞光线追踪

`apps/Unpolished/Lensing.py` 对画布上的每个像素反向积分一条 Kerr–Newman
零测地线，而不是贴图。取 G = c = M = 1，自旋 a = J/M、电荷 Q/M 由滑杆调节，
要求 a² + Q² < 1；a = Q = 0 退化为史瓦西黑洞。

GPU 内核在 Kerr–Schild 笛卡尔坐标中用 RK4 积分位置和动量，避免 Boyer–Lindquist
极轴上的 1/sin²θ 奇点。相机使用零角动量观测者（ZAMO）局部标架；吸积盘是赤道面
内的中性顺行圆轨道薄盘，多普勒增亮、引力红移和参考系拖曳同时生效，颜色取
gT_emit 的 Planck 黑体色。

CPU 参考解 `trace_kerr` 仍使用 BL 分离方程；GPU 与 CPU 结果用于回归测试。
界面可调自旋、电荷、质量、倾角、视场、盘内外缘、温度、不透明度、曝光和星密。

## Circulation 的地形与风场

- `assets/data/worldelev.npz`：ETOPO1（NOAA NGDC）1° 抽样，用于海拔配色、坡向阴影和合成风场的减速/绕流。
- `assets/data/worldmask.npz`：Natural Earth 720×360 海陆掩膜。
- `assets/data/wind_real.npz`：NCEP/NCAR Reanalysis 1 的 2015–2024 月平均 10 m 风，插值到 2.5° 网格；支持逐月播放、同月 Δ 和十年趋势。
- 重建脚本：`python tools/build_circulation_data.py`（NOAA ERDDAP / PSL，无需 API key）。
- 合成气压场是教学/对照模型，不含温度；真实风场来自再分析产品。

来源和引用见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## Erosion 的土壤水与 CZM 干裂

- `support/physics/soil_water.py` 实现两层水量平衡：Green–Ampt 入渗、
  Penman–Monteith 蒸散、径流与下渗。
- `assets/data/czm_fields.npz` 保存 4 组 Voronoi 多边形 cohesive 网络的
  9 级裂缝场（4×9×320×320），由 `tools/build_czm_bank.py` 生成。
- 运行时按土壤平均收缩率在帧间插值；张开 τ≈240 min、闭合 τ≈900 min，
  降雨时裂缝逐渐变浅，不会一帧消失。
- 每次启动随机选一组图案；裂缝完全闭合后开启新一轮干燥时换组。

## Lenia 预设

- 预设画廊分“自定义”和“经典”两页，按环数/科分组。
- 当前预设库 340 个，其中 292 个经典形态解码自
  [Chakazul/Lenia](https://github.com/Chakazul/Lenia)（MIT）的 `Lenia-LifeForms.js`。
- 核函数与生长函数实现 Bert Chan 原版：`gaus`、`quad4`、`trap`、`stpz`、
  `bump4`、`life`、`bell(旧)`。
- 预设格式扩展为 16 个参数（末尾追加 `growth_type` / `kernel_type`）；
  旧的 14 参数预设自动补 0，行为不变。
- 自动搜索默认随机参数 + 随机画笔作画，不从小扰动开始；支持 `--mode mutate`。

## 启动与导入优化

- 每个应用只导入自己需要的层：`qt_base`、`numpy_base`、`taichi_base`、
  `numba_base`、`opengl_base`、`mp_base`。
- 缩略图首次绘制时用 `QImageReader` 懒加载。
- Numba 磁盘缓存放在 `runtime/numba_cache/<入口脚本>/`，避免不同入口的
  编译缓存串味；只读安装回退到系统临时目录。
- 子进程就绪标记 + 日志轮转，避免固定的 6 秒等待和崩溃后无日志。
- 分形拖动先出 1x 预览，停手 180 ms 后按所选 AA 精修。
