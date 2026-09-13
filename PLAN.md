# 实验设计：TSFM 频率偏置的成因

本文件是**实验设计契约**（范围、结论目标、必须遵守的控制变量、度量定义）。

- 运行日志 / 结果 / 结论 → `experiments/README.md`
- 每个实验的意图、配置、通过标准 → `experiments/<id>/README.md`

## 0. 范围

| | |
|---|---|
| **问题** | 为什么 next-patch 预测模型表现出频率偏置？ |
| **模型** | `SimTFM`：input-space next-patch、**无 RevIN**、**无 SIGReg**、RoPE、T+patch_size 训练窗口 |
| **数据** | 合成语料（受控频谱） |
| **产出** | 机制解释 + 合成证据；**不做修正**，不追求真实数据 forecasting 收益 |
| **v1 目标** | 在一个干净 testbed 上**复现 lab 的结论**，不是提新主张 |

v1 之所以先复现：`SimTFM` 相比 lab 的 `MiniLTFM` 少了 RevIN 和 SIGReg，而这两个都是 journal 里被证明**承重**的部件。不先确认基本现象还在，后面任何新主张都无法解释。

## 1. 起点：lab 已确立的结论

来自 `projects/temporal-batch-comp-sigreg`。v1 就是在 `SimTFM` 上重建这张表。

| ID | 结论 | lab 证据 | 复现 |
| :-- | :-- | :-- | :-- |
| C1 | 偏置在**预测侧**，不是分辨率损失 | It.27-28：高频段在 `z` 上可复现 r²=**0.9995**，模型只给 r²≈0 / **7%** 幅度；低频斜率保留 ~54% | `1_pred_side` |
| C2 | **patching 编码频率**，不是 transformer | It.15：`k=1` 时 f=16/32/64 几何完全相同，plane overlap **1.000** | `4_patch_granularity` |
| C3 | shrinkage 是**学到的、位置无关的先验** | It.14：跨预测位置 `p=2..14` 比值平坦（f=32 **0.537**，f=64 **0.524**） | `2_pos_invariance` |
| C4 | 确定性语料上 shrinkage **消失**；排序由 `Δφ` 决定 | It.29-30：15k 步后 finals **0.988–0.998**，`Δφ=π` 的那个 f 永远最慢 | `3_deterministic_dphi` |
| C5 | **覆盖度**：训练集缺失的 f 即使确定也被阻尼 | It.31：不均匀 rich 语料收敛在 0.865，最差 cluster 0.715 | `6_coverage` |
| C6 | 相位环几何是**纯架构性**的 | It.9：无 z-regularizer 的 direct 模型环依然存在；It.26：`dim1+2 ≥ 0.94` | `5_ring_geometry` |

**这是概念复现，不是数值复刻。** `SimTFM` 没有 RevIN，journal 的 E3 显示 no-revin 会让环变偏心、trend tilt 塌陷；没有 SIGReg，It.49-51 显示 SIGReg 会把环拉成椭圆。所以 C6 的几何量预期会偏离 lab，这属于**已记录的差异**而不是失败。通过标准一律写成方向性判据。

**决策规则**：lab 的全部主证据都是在**带 RevIN** 的模型上采集的（E3 的 no-revin 只是消融）。所以若 `1_pred_side`（C1）在本仓库的无 RevIN 模型上**不复现**，不能直接判定 lab 结论错——必须先跑一个 RevIN twin 才能把「架构差异」和「结论错误」分开。这是 v1 最大的解释风险。

## 2. 先厘清：两种性质不同的 bias

Plan 初稿把两者混在了一个指标里，这是最需要修的：

- **暂态（优化顺序）**：低频先学会、高频后学会。由 `Δφ = 2π·f·k/ctx (mod 2π)` 决定，**不是由 f 决定**——It.29-30 在纯确定性语料上证明它是暂态，收敛后消失。
- **不动点（收敛后的阻尼）**：由训练边缘分布里的**条件可预测性**决定，`Δφ` 解释不了。It.31/It.14 都是它。

**推论**：`(高频误差 − 低频误差) / (高频误差 + 低频误差)` 这个"频谱偏置指数"不能用。误差是 `Δφ mod 2π` 的周期函数，高低频一相减就是在对周期函数做平均，换 `k` 或换 `ctx` 就能翻转符号。两者必须分开量，且频率一律在 `cycles/patch` 坐标上报。

## 3. 实验契约（每个实验都必须遵守）

这些是 lab 花了几十个迭代买来的 confound，不是建议：

1. **horizon 固定在 patch 数上**。It.35/36：「小 patch 更好」最后被证明是 horizon 不一致造成的假象。跨配置比较时预测的 patch 数必须相同。
2. **报告收敛状态，同时记录暂态**。It.21：同一配置从 10k 步增加，`k=1` 的 retention 从 0.574 涨到 0.843——一半的"偏置"其实是欠训练。
3. **探针语料的频率成分均匀计数**。It.45/49：不均匀 cluster 密度造成的"椭圆环"结论被归因错了，真正成因是训练时长。
4. **探针频率间隔 ≥ `ctx/k`（即 1 cycle/patch）**。It.34：正交阈值恰好是 `d = k`（`Δf=1` → 重叠 0.985，`Δf=8` → 0.29，`Δf=16` → **0.04**）。间隔更小则 DFT bin 能量泄漏，逐频 retention 直接失去意义。
5. **每个实验都有一把 oracle 上界**。It.46：冻结 `z` 上的手工 detect→rotate→recombine 管线 MSE **0.043–0.070**，transformer head **0.44–1.55**，差 10–35×。没有 oracle 就分不清"没学会"和"不可能"。
6. **单位**：频率以 `cycles/patch` 报，同时报 `Δφ`。
7. **每个配置至少 3 个 seed**，报离散度；单次运行的差异不作为结论。

## 4. 变量与度量定义

频率记法：`f` 以 cycles/window 记（窗长 `ctx`）。于是

```
cycles/patch = f · k / ctx        Δφ = 2π · f · k / ctx (mod 2π)
```

| 度量 | 定义 |
| :-- | :-- |
| **retention** `r_f` | 某个**固定预测位置**上，预测 patch 与真实 patch 在 bin `f` 的复幅度之比 `\|P̂_f\| / \|P_f\|`。这是"半径收缩"的频域形式 |
| **相位误差** | `angle(P̂_f) − angle(P_f)`（unwrap 后） |
| **ring** `dim1+2` | 对每个 `f`，收集相位环上的 latents，PCA 前两维的方差占比 |
| **plane overlap** | 两个频率各自 top-2 子空间的 Grassmann `mean(s²)`。1 = 同一平面，0 = 正交，0.5 = 共享一维 |
| **暂态指标** | `r_f(t)` 曲线，以及到达阈值 `ε` 所需步数 |

`r_f` 在 oracle 上定义为 `‖P̂_f^oracle‖ / ‖P_f‖`，用来把"表示里有没有"和"head 用不用"分开。

## 5. v1 实验列表

按基础设施依赖排序。`0_init` 已在仓库中建好。

### `0_init` — 基础设施 + 冒烟

把 harness 跑通：`data.py` 语料构造、`cli/train.py` 训练循环（T+patch_size 窗口、input-space shifted MSE、SGD）、`probes.py` 度量。用最小配置跑一次，确认能训到 loss 下降、能出 retention 数值。
**通过标准**：端到端跑通，产出一个可读的 sidecar JSON。

### `1_pred_side` — 偏置在预测侧（C1，主实验）

混合语料含低频 + 高频两个成分，间隔 ≥ 1 cycle/patch 保证 bin 可分。训练后测两件事：冻结 `z` 上的 oracle 能否重建下一 patch 的高频段；模型自己的预测保留了多少。
**通过标准**：oracle 在高频段 r² ≥ 0.9，且 model retention ≤ 0.3，且两者差距 ≥ 3×。
**副产品**：本仓库没有 SIGReg，所以这里若复现出 readout gap，就直接给"SIGReg 是成因"这个假设脱罪（journal backlog #1 的一支）。

### `2_pos_invariance` — shrinkage 位置无关（C3）

固定 `f`，扫预测位置。排除"上下文长度瓶颈"这类解释。
**通过标准**：`r_f` 跨位置的相对变化 < 20%。

### `3_deterministic_dphi` — 确定性语料 + Δφ 排序（C4）

纯音 / 双音确定性语料，多个 `f`，训练中每 N 步探针。
**通过标准**：收敛后所有 `f` 的 retention ≥ 0.9；收敛步数排序与 `Δφ mod 2π` 相关，`Δφ≈π` 最慢。

### `4_patch_granularity` — patching 编码频率（C2）

`k=1` vs `k>1`，同一组 `f`。
**通过标准**：`k=1` 时跨 `f` 的 plane overlap ≈ 1.000；`k>1` 时显著低于 1。

### `5_ring_geometry` — 相位环（C6）

**通过标准**：中频段 `dim1+2 ≥ 0.90`。无 RevIN 导致的偏心/塌陷按 §1 的说明记录为差异。

### `6_coverage` — 覆盖度（C5）

训练集完全缺失某些 `f`，在这些 `f` 上用纯音探针。
**通过标准**：off-support 的 retention 显著低于 on-support，且在确定性语料上依然如此。

## 6. 基础设施需求

| 文件 | 内容 |
| :-- | :-- |
| `src/fbias/data.py` | 语料构造器：纯音、多音混合、rich clusters、off-support 变体 |
| `src/fbias/probes.py` | §4 的度量 + oracle（冻结 `z` 上的 ridge / MLP） |
| `src/fbias/cli/train.py` | 训练循环；窗口必须是 `context_size + patch_size` |
| `experiments/<id>/scripts/` | 该实验的驱动脚本 |
| `experiments/<id>/runs/` | state_dict + sidecar JSON |
| `experiments/README.md` | 运行日志 |

**默认配置**：沿用 lab canonical 的 direct 配置——`ctx 1024 / k 64 / hidden 64 / 2L / 4 heads / SGD 1e-2 / batch 64`，便于与 journal 对照。冒烟用 `ctx 512 / k 32 / hidden 32`。

## 7. v2 候选（新主张，v1 完成后再定）

- **幅度轴**：固定 `f` 与可预测性，只动幅度分布（lognormal）。这是对 Fredformer「过度关注高能量频率」假设的直接检验。journal 只在 encoder 侧测过（It.5），readout 侧没测过。
- **梯度探针**：量"哪些频率在降低 loss"，作为"可预测性先验"说法的机制证据。journal 完全没做。
- **readout gap 的机制**：`z` 里有频率信息但 head 不用——为什么。v2 里这是主攻方向。
- **频率密度**：均匀 / 对数均匀 / 双峰。

## 8. 文献定位

| 工作 | 主张 | 与本设计的关系 |
| :-- | :-- | :-- |
| FreIE (ICDM 2025) | 频谱偏置源于**自相关** | 控制覆盖度与噪声，可分离"自相关"与"数据可预测性"的贡献 |
| Basri et al. (ICML 2020) | NTK 理论：偏置与**输入密度**有关 | `6_coverage` 是对该理论的直接实证检验 |
| Fredformer (KDD 2024) | 偏置源于**过度关注高能量频率** | v2 的幅度轴直接检验 |
| Maddix et al. (arXiv 2510.19236) | patching 引入的 temporal bias | `4_patch_granularity` |
| Yu et al. (arXiv 2510.03358) | TS transformer 的 rank 结构 | `5_ring_geometry` 的低维性 |
