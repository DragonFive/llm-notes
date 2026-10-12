---
title: Fusion / Tiling / Layout 三元组
aliases:
  - 'fusion_tiling_layout'
  - 'Fusion / Tiling / Layout 三元组'
date: 2026-06-23
updated: 2026-06-23
tags:
  - type/concept
  - domain/llm-inference
  - topic/compiler
  - topic/gpu-kernel
  - topic/optimization
source_count: 1
status: draft
---

# Fusion / Tiling / Layout 三元组

> Fusion、Tiling、Layout 是 AI 编译器中决定 GPU 执行效率的三大核心优化维度。它们不是独立的"开关"，而是相互约束、相互增强的优化体系——局部最优的组合往往不是全局最优。

## 核心要点

- **Fusion** = 数据流边界的重构（消除中间结果的 HBM 读写）
- **Tiling** = 片上资源的空间分配（tile 大小决定数据在存储层次间的分布）
- **Layout** = 访存模式的全局优化（数据排布决定带宽利用率）
- 三者形成协同效应：Tiling 为 Fusion 提供空间，Layout 影响 Tiling 的有效性，Fusion 改变 Layout 的约束

## Fusion：数据流边界的智能重构

### 表面理解 vs 系统本质

| 维度 | 表面理解 | 系统本质 |
|---|---|---|
| 做什么 | 合并多个算子为一个 kernel | **重画运行时的数据流边界和执行边界** |
| 核心收益 | 减少 kernel launch overhead | **消除中间 tensor 的 materialization** |
| 本质改变 | 少 launch 几次 | 中间结果是否落盘 HBM、调度粒度是否改变 |

为什么"系统本质"比"合并算子"更关键？下面展开四个核心问题：

### 中间结果是否 materialize（落盘）

**Materialize** 的字面意思是"物化"、"实体化"——Fusion 决策实际上决定的是：一个算子的输出是否必须**写回到 HBM**，才能作为下一个算子的输入被读回来。

没有 Fusion 的情况：
```
算子 A: input → 计算 → 写回 HBM（产生 tensor X）
算子 B: 从 HBM 读 tensor X → 计算 → 写回 HBM（产生 tensor Y）
算子 C: 从 HBM 读 tensor Y → 计算 → 写回 HBM（最终结果）
```
**三个步骤分别三个 kernel，中间结果 X 和 Y 都 materialize 到了 HBM。**

Fusion 后的情况：
```
单一 kernel: input → 算 A → 算 B → 算 C → 最终结果写回 HBM
```
中间结果 X 和 Y**不会落盘**——它们直接以**寄存器值**或**Shared Memory 数据**的形式在算子边界之间传递，根本不出现在 HBM 上。

这就是文章中说的「Fusion 真正改变的是中间结果是否 materialize」。它节省的不是浮点运算本身，而是**HBM 的来回搬运**。

> Materialize 是编译器/数据库领域的通用术语。在数据库里叫"物化视图"（Materialized View），指把查询结果预先写出来存盘；在 GPU 编译器里含义完全相同：把一个计算过程的结果显式写到 HBM 上。

### 数据是否回写 HBM

这是上一个问题的硬件视角。"回写 HBM"= materialize 的硬件动作。Fusion 改变了什么？

| 维度 | 多 kernel（不 fusion） | 单一 fused kernel |
|---|---|---|
| 算子 A→B 之间 | 数据路径：Register → **HBM** → Register | 数据路径：Register → **Shared Memory** → Register（或更短） |
| 数据生命周期 | 退出 register 后"假死"在 HBM，等待下一个 kernel resurrection | 数据一直在片上流转，保持"活着" |
| 带宽瓶颈 | 受 HBM 带宽（GB/s）限制 | 受 L2/Shared Memory 带宽（TB/s）限制 |

Fusion 前，每过一个算子就有一次 HBM write；Fusion 后，所有中间步骤共享同一条片上流水线。

### 不同 kernel 之间的边界是否保留

这是从**执行单元调度**视角看 Fusion 的本质改变。

**Boundary 保留**（不 fusion）：
```
kernel_A() { 算 A }  ← GPU 执行完毕，CPU 发射下一个
kernel_B() { 算 B }  ← GPU 执行完毕，CPU 发射下一个
kernel_C() { 算 C }
```
每两个算子之间都有**kernel boundary**。这意味着：
- 算子 A 的所有 thread 必须**全部完成**
- GPU 必须**全部排空流水线**
- 然后 CPU 才能**发射**算子 B
- 算子 B 执行中，算子 A 的中间结果被视为**只读输入**

**Boundary 溶解**（fusion 后）：
```
fused_kernel() { 算 A → 算 B → 算 C }
```
算子之间没有 kernel boundary，只有**代码边界**（switch-case 或内联函数调用）。这意味着：
- 数据可以在 register/shared memory 中**活着传递**
- 不需要 CPU 再参与调度
- GPU thread 的**生命周期**贯穿整个计算链（而非每个算子都生死一次）

> 所以"不同 kernel 之间的边界是否保留"不是说"代码有没有分割线"，而是说 GPU 的**执行单元调度粒度**——是"每算子调度一次"还是"整条链调度一次"。

### 调度粒度是否改变

Fusion 前后，GPU 调度的**原子单位**发生了变化：

| 场景 | 调度粒度 | 调度器（谁决定下一个跑什么） |
|---|---|---|
| 不 Fusion | 单个 kernel | CPU + GPU hardware scheduler |
| Fusion 后 | 整个算子链 | **编译器静态决定**（kernel 内部顺序是确定的） |

**调度粒度改变意味着什么？**

1. **Kernel launch 开销**：每多一个 kernel boundary，就多一次 CPU→GPU 的 launch 开销（几 us 到几十 us）。Fusion 把多个 launch 合并为一次。
2. **流水线利用**：不 fusion 时，kernel 之间可能有 idle（等上一个完成）。Fusion 后，算子 A 的 tile N 还没算完，算子 B 的 tile N-1 已经可以在同一个 SM 上启动——**时间维度的流水线重叠**。
3. **并行性粒度**：不 fusion 时，并行粒度由每个算子自己的并行策略决定。Fusion 后，并行粒度由编译器重新设计——可能原来 A 和 B 的 thread grid 形状不同，Fusion 后编译器统一了并行粒度。

> 这就是为什么文章说 Fusion 真正改变的不是"少 launch 几次"，而是**系统级的调度语义、数据生命周期、存储层次使用方式**的全局重组。减少 launch overhead 只是最表面的副产品。

### Fusion 与 Occupancy 的关系

文章提到"更大的计算块使得调度器能够更稳定地维持高 occupancy"——这需要理解 occupancy 的真正含义。

**Occupancy = SM 上活跃 warp 数 ÷ SM 最大可容纳 warp 数**

GPU 不靠 cache 命中隐藏延迟，而是靠**切换其他 warp**来隐藏延迟（latency hiding）。Occupancy 高 ≠ 每个 thread 都在算，而是 **SM 调度器随时都能找到可运行的 warp**，不出现"无活可干"的空转。

**未融合时 occupancy 为什么波动？**

每个小算子（如 `Bias Add`、`ReLU`）launch 一个独立的 kernel，但它们的 thread grid 很小：

```
算子 A：MatMul   → 大 kernel，1024 个 block，SM 满载
    ↓ kernel boundary
算子 B：Bias Add → 小 kernel，128 个 block，大量 SM 空闲
    ↓ kernel boundary
算子 C：ReLU     → 小 kernel，128 个 block，大量 SM 空闲
```

Bias Add 和 ReLU 计算量太小——它们的 grid 不够大，很多 SM 根本没活干。Occupancy 在 kernel 切换的瞬间骤降。

**融合后为什么稳定？**

融合后三个算子变成一个 fused kernel，thread grid 是 MatMul 级别的（大），Bias Add + ReLU 的工作量被"附挂"在每个 thread 上——不增加新的 grid launch，也不会出现 SM 空载的 gap。

```
Fused kernel：MatMul + Bias Add + ReLU
    → thread grid = MatMul 的大小
    → 每个 thread 在算完 GEMM 的一块后，紧接着算 Bias + ReLU
    → SM 始终满载，无调度 gap
```

### Epilogue Fusion：GEMM 收尾阶段的「免费」计算

文章提到"将 element-wise 操作融入矩阵乘法的 epilogue 阶段，利用计算单元的空闲周期'免费'完成"——这是 Fusion 最精妙的模式之一。

#### GEMM 内部的三阶段

```
1. Main Loop（主循环）：
   沿 K 维度分 block，循环做 MMA（矩阵乘加）
   → 计算密度极高，Tensor Core 全力运转
   → compute bound

2. Reduction（归约）：
   把各 K-block 的部分和加起来
   → 轻量计算

3. Epilogue（收尾阶段）：
   把累加器（Accumulator，在寄存器里）的结果写回 HBM
   → Tensor Core 已经闲置
   → memory bound（等 HBM 写回）
```

**关键洞察**：Epilogue 阶段的 Tensor Core 是闲着的——它在等 Accumulator 数据写回 HBM。这段时间是**硬件空闲周期（free cycles）**。

#### Epilogue Fusion 的原理

把 Bias Add 和 ReLU 放进 Epilogue 阶段：

```
正常流程：
  [Tensor Core 算 GEMM] → [写 C 到 HBM] → kernel 结束
                              ↑
                        Tensor Core 在这里闲着

Epilogue Fusion 后：
  [Tensor Core 算 GEMM] → [读 Accumulator] → [加 Bias] → [算 ReLU] → [写结果到 HBM]
                                               ↑             ↑
                          这两步用 CUDA Core（普通计算单元），
                          藏在 HBM 写回的 memory latency shadow 里
```

**为什么"免费"？**

1. Bias Add 和 ReLU 用的是 **CUDA Core**，不是 Tensor Core
2. 在 GEMM 的 Epilogue 阶段，CUDA Core 本来就有空闲时间
3. 整个 kernel 的**墙钟时间基本不变**——Epilogue 的延迟由 HBM 写回主导，Bias Add + ReLU 在这段时间内被"藏进去"了

不是真的零成本，而是**隐藏在 memory latency 的 shadow 里**——这段时间本来就会花，额外的计算是"赠品"。

#### Transformer 中的具体案例

Transformer Block 中每个 Linear 本质是一个 GEMM，每个 GEMM 后面都跟着 element-wise 操作：

```
Q/K/V Projection:
  GEMM → + Bias Add（proj_bias）        ← ✅ Epilogue Fusion
       → + Rotary Embedding（RoPE）      ← ✅ element-wise sin/cos

FFN FC1:
  GEMM → + Bias Add（fc1_bias）          ← ✅ Epilogue Fusion
       → + GELU / SwiGLU                 ← ✅ 经典 epilogue fusion 案例

FFN FC2:
  GEMM → + Bias Add（fc2_bias）          ← ✅ Epilogue Fusion
       → + Residual Add（x = x + out）   ← ✅ element-wise
```

**最具代表性的案例：FC1 + GELU**

以 LLaMA-3 70B（d_ffn=28672）为例：

```
不 Fusion：
  FC1 写 [B, S, 28672] → HBM     (HBM write: B×S×28672 × 2 bytes)
  GELU 从 HBM 读 → 算 → 写回 HBM  (又一次 HBM 来回)
  → 两次 HBM IO，d_ffn=28672 极大

Epilogue Fusion 后：
  FC1 算完 → Accumulator 在寄存器 → 直接算 GELU → 写结果到 HBM
                                     ↑
                 中间结果 [B, S, 28672] 从未出现在 HBM 上
                 节省了 2×(B×S×28672×2 bytes) 的 HBM 流量
```

对于每个 token，仅 FC1+GELU 的 Epilogue Fusion 就节省约 28672×2×2 ≈ 100KB 的 HBM IO——这是 prefill 阶段的主要加速来源之一。

### 关键 trade-off

| Fusion 类型 | 典型场景 | 效率增益来源 | 复杂度挑战 |
|---|---|---|---|
| Element-wise Fusion | 激活函数链 | 消除中间 tensor | 寄存器分配 |
| Reduction Fusion | LayerNorm、Softmax | 减少全局同步 | 并行 reduction 设计 |
| Epilogue Fusion | GEMM + Bias + ReLU | 利用计算空闲周期 | 与主计算流水线协调 |
| Attention Fusion | FlashAttention | 重构数据流拓扑 | 算法级重新设计 |

Fuse 得越激进 → 数据局部性越好，但 kernel 复杂度上升 → register 压力增大 → 可能降低 occupancy。

Fuse 得太保守 → 中间结果频繁落盘 HBM → 带宽成为瓶颈。

#### Element-wise Fusion（元素级融合）

**核心特征**：每个输入元素独立运算，不需要看别的元素。融合没有任何同步依赖——两条 Python 语句放一个 kernel 循环里就完成了。

```python
# 融合前：两个 kernel
def kernel1(x, bias):
    y = x + bias            # HBM: 读 x+bias，写 y
def kernel2(y):
    return relu(y)           # HBM: 读 y，写结果

# 融合后（一个 kernel）
fused: y = relu(x + bias)   # 中间结果 y 从未出现在 HBM 上
```

**为什么几乎"免费"**？因为每个元素独立 → 不需要 barrier、不需要 shared memory、不需要 warp 间通信。融合代价为零，收益是省一次 HBM round-trip。

#### Reduction Fusion（归约融合）

**核心特征：融合对象中包含需要"看所有元素才能算出结果"的算子（sum、max、mean）。每个元素不独立。**

以 LayerNorm 为例：

```python
# 不融合：需要两个 kernel
# Kernel 1：算 sum-of-squares 得到 mean/var → Reduction
mean = sum(x) / N
var = (x - mean)^2
# 两个 kernel 之间有隐式的全局同步——等所有元素算完

# Kernel 2：归一化 + 缩放 → Element-wise
x_norm = (x - mean) / sqrt(var + eps)
y = gamma * x_norm + beta

# 融合后：一个 kernel（含 parallel reduction 步骤）
# 在 shared memory 里做树形归约、多次 __syncthreads() barrier
```

**与 Element-wise 的本质区别**：Element-wise 的融合不需要线程通信，reduction 的跨元素运算（sum、max）需要 warp shuffle + barrier + shared memory。代价是更高的同步开销，收益是消除 reduction 中间结果的 HBM round-trip。

**可组合性**：Reduction + Element-wise 经常组合。RMSNorm + GELU = one kernel，前半段是 parallel reduction（同步复杂），后半段 GELU 是 element-wise（无额外开销）。

#### Epilogue Fusion（收尾融合）

**核心特征**：把 element-wise 操作 "塞" 进 GEMM kernel 的 Epilogue 阶段。

```
GEMM 内部三阶段：
  Main Loop → Tensor Core 满载进行矩阵乘加（compute bound）
  Reduction  → CUDA Core 归约，轻计算
  Epilogue   → 数据搬回 HBM（memory bound）— 此时 Tensor Core 空闲、CUDA Core 空闲

Epilogue Fusion 的原理：
  在 Epilogue 阶段把 Bias Add / ReLU / GELU / Residual Add 插进去
  → 利用这段"本就在等待 HBM 写回"的时间——结果几乎没有额外开销
```

**代价极低**：这些轻量运算正常需要它们自己的 kernel launch + 过一次 HBM round-trip。塞进 Epilogue 后，这些成本全部消失。

#### Attention Fusion（注意力融合）

**核心特征**：不是省 HBM 流量，而是改写整个计算的**数据流拓扑**。详见 FlashAttention。

具体来说：Q·K^T 是一个 GEMM，Output 也是计算量大的 MatMul。但中间的 Softmax 是 Reduction——按普通做法，这会 materialize S = Q·K^T 这个大矩阵（O(N²)，Attention 的瓶颈所在）。**FlashAttention 用 tiling + online softmax + selective recomputation 把它全链融合成一个 kernel。**

tiling 要求计算流程完全重新设计——当 Q、K、V 被切块后，Softmax 的 sum-of-exponentials 必须分多次迭代（online softmax），不能再一次性 materialize。

**局限性**：目前只适合 Self-Attention（Q、K、V 来自同源），通用 cross-attention 的融合仍需研究。

#### 这四种的分类维度

四种不是"四个互斥的池子"（即不是说什么可以和什么合并的意思），而是按**融合的内在复杂度来源**分类：

| Fusion 类型 | 复杂度来源 | 融合的本质 |
|---|---|---|
| **Element-wise** | 无（每个元素独立） | 两条 Python 语句合并 |
| **Reduction** | 同步（barrier + shared memory + warp shuffle） | 跨元素汇聚后的 element-wise 写回 |
| **Epilogue** | 微架构理解（时机精确控制） | 把轻量运算藏进重量 kernel 的硬件空闲期 |
| **Attention** | 算法重写（online softmax + tiling） | 重排计算拓扑，消除 O(N²) 中间矩阵 |

**能组合的典型组合**：
- Reduction + Element-wise：RMSNorm + GELU（reduction 阶段耗 barrier，element-wise 阶段几乎免费）
- Epilogue + Element-wise：GEMM + Bias + GELU
- Attention + Element-wise：FlashAttention 本身含三个 matmul + softmax + dropout

**不会融合的组合**：
- Reduction + Reduction：两个 reduction 不能共用同步点（各有各的 barrier）
- 两个都很大的算子：如两个 GEMM 融合后 register pressure 爆炸，反而降低 occupancy

### 硬件代际差异：Fusion 收益随 ops:byte 比率变化

**ops:byte 比率** = FLOPs / Byte/s——每从 HBM 搬 1 byte 数据，GPU 能做多少次浮点运算。这个比率决定了一个 kernel 是 compute-bound 还是 memory-bound，进而决定 fusion 的收益大小。

#### A100 vs H100：谁更需要 fusion？

| 指标 | A100 SXM4 | H100 SXM5 | 增幅 |
|---|---|---|---|
| 算力 (FP16 dense) | 312 TFLOPS | 989.5 TFLOPS | **3.17x** |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s | **1.68x** |
| **ops:byte 比率** | **156** | **295** | **~1.89x** |

**文章的一个错误**："高带宽架构（如 H100 HBM3）fusion 的收益相对降低"——**这与数据矛盾**。

H100 的 ops:byte 比率从 156**涨**到 295，算力增长远超带宽增长：

```
A100 上：每 byte 数据可做 156 次运算 → compute-bound 的多
H100 上：每 byte 数据可做 295 次运算 → memory-bound 的更多 → fusion 收益更大
```

类比：A100 是"一口小锅，小火慢炖"，H100 是"一口大锅，大火猛炒但水龙头只大了 1.7 倍"——水龙头（带宽）跟不上锅（算力）的增长，**水龙头反而成了更明显的瓶颈**，消除 IO 的 fusion 收益**不降反升**。

**TMA（Tensor Memory Accelerator）不能替代 fusion**：

| 路径 | 做什么 | 是否消除 IO？ |
|---|---|---|
| TMA | 数据搬运与计算重叠（overlap） | ❌ 中间结果仍然 materialize 到 HBM |
| Fusion | 数据流边界重构 | ✅ 中间结果根本不碰 HBM |

TMA 是"降本"，Fusion 是"消除"。两者正交——TMA 优化必须搬的数据搬运效率，Fusion 消除不必要的搬运。

#### 一般性原理：ops:byte 如何影响 fusion 策略

```
ops:byte 高（H100: 295）→ 对带宽极度敏感 → fusion 收益大 → 编译器偏向激进 fusion
ops:byte 低（假设未来卡: ~99）→ 带宽富余 → fusion 消除 IO 的边际收益缩水 → 动力减弱
```

> [!warning] 但这不意味着富带宽无需 fusion。compile-time fusion 的不可替代优势是**跨算子全局 tiling/layout 优化**——编译器看到整条算子链的 shape，可以做联合 tiling/layout 决策。runtime persistent kernel 在运行时才知道 shape，来不及全局搜索最优解。这是编译器的护城河。

#### compile-time fusion vs runtime persistent kernel

带宽富余后还有一个关键问题：编译器还走 compile-time 静态 fusion，还是更多交给 runtime persistent kernel？

| 维度 | compile-time fusion | runtime persistent kernel |
|---|---|---|
| IO 处理 | ✅ 中间结果完全消除 | ❌ 中间结果仍走 HBM |
| 全局 tiling/layout | ✅ 跨算子联合优化 | ❌ 运行时才知 shape，来不及搜索 |
| 灵活性 | ❌ shape 变要重新编译 | ✅ 适配动态 shape |
| register pressure | ❌ 复杂 kernel 可能爆 | ✅ 逻辑可分散 |
| 跨"逻辑算子"共享 | ❌ 每个 kernel 独立执行 | ✅ 同一 kernel内 shared memory 可跨算子复用 |

**两条路的本质张力**：compile-time 做全局优化但缺灵活性，runtime 做灵活适配但无法做全局搜索。**ops:byte 从高变低 = "消除 IO"收益缩水 → runtime 的灵活性优势相对上升 → 编译器倾向于降低融合激进程度。**

**两条路并不互斥**。未来的最优解很可能是：**轻度编译时 fusion（处理已知 pattern）+ runtime 做动态编排。** 编译器的护城河——全局 tiling/layout 优化——仍然不可替代。

## Tiling：片上资源的精细化分配

### 超越数学分块的系统视角

传统理解：tiling = 矩阵分块的数学操作。
系统理解：tiling = **有限片上资源约束下寻找全局最优分配方案**。

这和传统 OS 管理页缓存、调度 I/O 队列、分配文件缓冲区的本质逻辑高度同构——区别仅在于管理对象从 page/cache line/file buffer 变成了 tile/register block/shared memory slice/tensor fragment。

### 三维分析

**计算效率维度**：
- Tile 大小 ↔ Occupancy 的 trade-off（大 tile → 单 block 高效但并行度受限）
- Tile 形状决定 warp 协作、访存 coalescing、bank conflict
- Tensor Core 场景下 tile 必须对齐 MMA 原生分块要求（如 16×16×16）

**内存开销维度**：
- HBM ↔ Shared Memory 带宽差约 10x，Shared Memory ↔ Register 差数倍
- 合理 tiling 使数据在片上复用 O(tile_size) 次，而非每次从 HBM 重载

**硬件适配维度**：
- A100：192KB 共享内存/SM，需要更大 tile
- H100：TMA 支持异步 tile 预取，需考虑 prefetch 时机
- 最优 tile 尺寸往往需要 auto-tuning

## Layout：访存模式的全局优化

### Layout 的"传染性"

Layout 的影响不是局部的——一旦某个 tensor 选定了特定 layout，与之交互的所有算子都需要适配，否则就插入代价高昂的 layout 转换（transpose）。这种传染性意味着 layout 决策的影响范围远超单个算子。

### Layout 与系统其他部分的耦合

| 系统组件 | 与 Layout 的耦合点 | 影响范围 |
|---|---|---|
| 通信层 | Collective 操作的切分方式 | 跨卡带宽利用率 |
| 内存分配器 | Tensor 对齐与 padding | 显存碎片率 |
| 图编译器 | 算子 fusion 的可行性 | 全图优化空间 |
| Auto-tuning | 搜索空间维度 | 编译时间与最优解质量 |

## 三大技术的协同效应

### 互相约束与互相使能

| 技术维度 | 对其他技术的约束 | 对其他技术的使能 |
|---|---|---|
| Tiling | 限制可 fusion 算子的大小 | 为 fusion 提供片上空间 |
| Fusion | 改变 tiling 的资源预算 | 消除 layout 转换需求 |
| Layout | 影响 tiling 的访存效率 | 使 fusion 的数据流更高效 |

### 核心公式

```
Tiling 为 Fusion 提供可能性：
  → tile 够小 → 中间结果驻留片上 → fusion 收益实现
  → tile 太大 → 片上溢出 → spill 到 HBM → fusion 收益被抵消

Layout 影响 Tiling 的有效性：
  → layout 匹配访问模式 → coalesced → 带宽高效
  → layout 不匹配 → non-coalesced → 带宽浪费

Fusion 改变 Layout 的约束：
  → 融合后生产者消费者在同一 kernel → layout 转换可能不必要
  → 融合后新访问模式 → 可能引入新的 layout 要求
```

## 计算范式的根本迁移：CPU 时代 vs AI 时代

### 一等对象的全面对比

AI 系统的核心矛盾已从"进程如何使用 CPU 时间"转向"张量程序如何高效利用异构算力与多层内存"。这导致系统真正操作的对象发生了根本性改变：

| 维度 | CPU 时代一等对象 | AI 时代一等对象 | 本质差异 |
|---|---|---|---|
| 执行单元 | Process / Thread | Kernel / Stream / Graph | 从"时间片竞争"到"空间分配" |
| 内存对象 | Page / Buffer | Tensor / KV Cache / Activation | 从"虚拟内存页"到"张量块" |
| 调度粒度 | 时间片 | Token / Request / Batch | 从"公平调度"到"吞吐优先" |
| 通信抽象 | Socket / Pipe | Collective / AllReduce | 从"点对点"到"集体通信" |
| 状态管理 | File Descriptor | Session / Context | 从"文件句柄"到"会话上下文" |

**核心洞察**：Linux Kernel 的四大抽象（Process、Virtual Memory、File、Device Model）仍然存在，但关键路径上的决策已不在其"热路径"中。真正决定性能的事情，正在被 runtime、compiler、kernel DSL、serving system 接管。

### GPU 系统栈的层次结构

现代 GPU 已从"加速器"演变为拥有独立执行模型、内存层次、同步机制的**复杂计算域**。GPU 系统栈形成了完整的层次结构：

```
┌─────────────────────────────────────────────────────────┐
│ 应用层：Agent / Workflow / Tool                          │
│ 提供高层语义，屏蔽底层优化细节                              │
├─────────────────────────────────────────────────────────┤
│ 运行时层：Serving / Session / Memory                     │
│ Fusion 策略的运行时决策、KV Cache 管理                    │
├─────────────────────────────────────────────────────────┤
│ 编译器层：Compiler + Kernel DSL                          │
│ Fusion / Tiling / Layout 的核心战场                      │
│ 决定 tensor program 如何映射到 GPU 并行层次与内存层次     │
├─────────────────────────────────────────────────────────┤
│ 执行内核层：Driver + Runtime                             │
│ Context、地址空间映射、命令提交、同步原语                  │
│ Stream、Event、Memory Allocator、Graph Execution         │
├─────────────────────────────────────────────────────────┤
│ 底座层：Linux Kernel                                     │
│ 提供基础隔离与资源管理                                    │
└─────────────────────────────────────────────────────────┘
```

**关键分层职责**：

| 层次 | 核心职责 | 三大技术的作用点 |
|---|---|---|
| Driver（驱动层） | Context、地址空间、DMA、中断 | 提供硬件访问接口 |
| Runtime（运行时层） | Stream、Event、Allocator、Graph | 暴露执行原语 |
| Compiler（编译器层） | Fusion、Tiling、Layout、Codegen | **核心优化战场** |
| Kernel DSL（内核 DSL） | 并行结构、数据布局、memory movement | 显式表达硬件约束 |
| Serving System（服务系统） | KV Cache、Request、Batch、Session | 运行时决策 |

**编译器层的特殊性**：

在传统应用开发中，编译器只是"翻译工具"。但在 AI 系统中，编译器成为决定性能的核心系统组件：

- **不再是**：源码 → 二进制的简单翻译
- **而是**：tensor program → GPU 并行层次 + 内存层次的映射决策

这正是 Fusion、Tiling、Layout 发挥作用的核心战场。

## 与 2026-06-23_lowering 的关系

三大技术在 2026-06-23_lowering 流水线的不同阶段发挥作用：

- **Graph Lowering 阶段** → Fusion 决策（哪些算子合并）
- **Operator Lowering 阶段** → Tiling + Layout 决策（tile 大小、数据排布）
- **Instruction Lowering 阶段** → 将优化决策落实为指令

> [!warning] 容易混淆的点
> Fusion 和 Tiling 看似在不同阶段，但它们通过"片上资源约束"深度耦合。Graph Lowering 做融合决策时，必须"向前看"——考虑 tiling 阶段是否放得下。这不是简单的 pipeline，而是需要跨层联合优化。

## 经典案例：FlashAttention

FlashAttention 是三大技术协同的典范：

- **Tiling**：Q/K/V 切分为适配片上的小块，online softmax 避免全量 materialization
- **Fusion**：Scale + Mask + Softmax + MatMul 融合为一个 kernel，消除所有中间 HBM 访问
- **Layout**：专门的数据排布使分块计算时访存高度 coalesced

结果：attention 显存从 O(N²) → O(N)，同时因更好的访存模式获得速度提升。

## 与传统 OS 资源管理的同构性

| AI 编译器概念 | 传统 OS 概念 | 本质同构 |
|---|---|---|
| Tile | Page | 有限资源的分配单元 |
| Shared Memory 分配 | Buffer Cache | 热数据的暂存区 |
| Fusion | I/O 合并 | 减少不必要的数据搬运 |
| Layout | 磁盘块排布 | 顺序访问 vs 随机访问 |
| Auto-tuning | 自适应调度 | 运行时参数搜索 |

## 未解问题

- 联合优化的搜索空间爆炸：三维联合搜索 vs 逐阶段优化的 trade-off
- 编译时 vs 运行时：哪些决策可以静态做，哪些必须 JIT？
- 跨硬件平台：NVIDIA/AMD/Ascend 的"甜点"差异极大，如何统一抽象？

## 参考来源

1. 知乎文章《从 Fusion、Tiling、Layout 来看 AI 时代的计算范式变革》— 提出三元组协同优化框架，CPU vs AI 时代一等对象对比，GPU 系统栈层次结构
2. 与 Maxwell 的讨论 (2026-06-23) — 追问 lowering 的定义、Triton Block-Level 编程模型、occupancy 与 fusion 的关系、epilogue fusion 的"免费"原理及 Transformer 具体案例、ops:byte 比率与硬件代际对比、compile-time vs runtime 路径的权衡、四种 Fusion 类型详解及组合规则
