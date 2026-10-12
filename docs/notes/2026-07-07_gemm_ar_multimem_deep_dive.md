---
title: GEMM+AR 融合与 multimem 通信深潜
aliases:
  - 'gemm_ar_multimem_deep_dive'
  - 'GEMM+AR 融合与 multimem 通信深潜'
date: 2026-07-07
updated: 2026-07-07
tags:
  - type/lecture
  - domain/llm-inference
  - topic/gpu-kernel
  - topic/cuda
  - topic/gemm
  - topic/communication
source_count: 2
status: stable
difficulty: advanced
---

# GEMM+AR 融合与 multimem 通信深潜

> 完整讲解：从通信粒度选择、UC/MC 地址体系、multimem 指令集、NVLink SHARP 在网计算，到 two-shot 同步协议、任务划分与三层同步的完整链路。

## 场景引入

在 Tensor Parallelism（TP）的 RowParallelLinear 中，每个 GPU 算出输出矩阵的一个部分（f_i @ B_i = Y_i），需要 AllReduce 得到完整结果 Y = sum(Y_i)。传统做法是 GEMM kernel 算完后，再走 NCCL ring AllReduce——但这开启了两个问题：

1. **调度不确定性**：GEMM kernel 占满所有 SM，AR kernel 排队等 SM 释放。
2. **重叠窗口碎片化**：GEMM 尾部释放 SM 是渐进的，AR 能抢到的 SM 数量不可控。

答案是把 AllReduce 做进 GEMM kernel 内部——GEMM+AR 融合。底层硬件能力是 multimem 指令集 + NVLink SHARP。

## 前置概念解释

### 通信粒度：tile grain vs chunk grain

| 维度 | Tile Grain | Chunk Grain |
|------|-----------|-------------|
| 粒度 | 单个 GEMM CTA tile（例如 128×128） | 更大数据块（多个 tiles、KV block） |
| 通信时机 | 计算自然产出边界 | 阶段边界（输入 gather、输出散射） |
| 通信方式 | 嵌入 GEMM kernel | 单独通信 kernel / Copy Engine |
| 优点 | 天然 overlap、隐藏延迟好 | 工程简单、可用 CE 卸载 |
| 缺点 | kernel 复杂 | 通信尾巴长、暴露空闲 |
| 典型场景 | GEMM+AR (TP RowParallel) | All-Gather+GEMM、Ring Attention |

GEMM+AR 适合 tile grain，因为 RowParallelLinear 的输出按 tile 产出——每个 CTA tile 算完后数据就在 register/shared/global 路径上，立刻发起通信天然契合。

### CTA 内 warp 分工

一个 CTA（线程块）最多 1024 线程 = 32 个 warp。程序员自由分配 warp 到不同"岗位"：

- GEMM warp group：做 MMA 计算
- EPILOGUE warp group：做 epilogue（TMA store、flag arrive）
- AR warp group：做 AllReduce 通信

所有 warp 在同一个 SM 上驻留，4 个 warp scheduler 轮转调度。分工后各 group 可以流水线并行：GEMM 算 tile_{i+1}，epilogue 写 tile_i，AR reduce tile_{i-1}。

### TMEM：Tensor Memory

Blackwell（B200）引入的新硬件层次，专门存 MMA 指令的 accumulator（累加器）。Hopper/Ampere 中 accumulator 占通用寄存器（GPR），导致 register pressure 大、occupancy 低。TMEM 把 accumulator 从 GPR 分离出来，**释放 GPR 空间给额外的 AR warp group**。这就是"同一个 CTA 里多放一组 AR warp"的前提。

### UC 地址与 MC 地址

| 类型 | 含义 | 访问方式 | 目标范围 |
|------|------|---------|---------|
| UC address | 普通 CUDA 物理地址 | `ld.global` / `st.global` | 本 GPU 的物理内存 |
| MC address | 多播虚拟地址 | `multimem.*` 指令 | 所有 rank 的对应物理内存 |

同一块物理内存在 GM 上。UC 和 MC 是两条不同访问路径。UC 走本地 load/store，MC 走 NVLink → NVSwitch 在网计算。

## 核心讲解

### 1. NVLink SHARP 与 NVSwitch

NVSwitch 是 GPU 之间的**交换芯片**，本体功能是路由：把数据从端口 A 搬到端口 B。NVLink SHARP（NVLS）是 NVSwitch 内部集成的**在网计算引擎**，可以在数据流经 switch 时截住数据做 reduce/broadcast，不需要把一个 rank 的数据搬进另一个 rank 的 SM 上算。

SHARP 最早来自 Mellanox InfiniBand 交换机，NVIDIA 收购后搬到 NVSwitch 上。**首次出现于 Hopper 架构（H100，SM90）。Ampere 有 NVSwitch 但没有 SHARP 能力。**

### 2. 三条 multimem 指令

| 指令 | 作用 | AR 中的用法 |
|------|------|-----------|
| `multimem.ld_reduce` | 从多个 rank 的 MC 地址读数据，NVSwitch 做 reduce，结果返回发起者 | Tile AllReduce 的核心：Gather + Reduce |
| `multimem.st` | 发起者写数据到 MC 地址，NVSwitch broadcast 到所有 rank | 把 reduce 结果广播回所有 rank |
| `multimem.red` | 对多个 rank 的 MC 地址做原子 reduce（如 add） | 做 tile flag 计数器、跨 rank barrier |

### 3. two-shot 同步协议

多 rank 协作中，AR warp 必须知道"所有 rank 的 tile 都写好了"才能发起 ld_reduce。two-shot 协议用 tile 粒度的 MC flag 做生产者-消费者同步：

**生产者（epilogue warp group）**：
1. `cp_async_bulk_wait_group(0)` — 等 TMA store 硬件完成
2. `fence_acq_rel_gpu()` — 确保数据先于 flag 被其他 rank 看到
3. `elect_one()` + `multimem.red.add(flag, 1)` — 一个线程对 flag +1
4. `fence_proxy("alias")` — 确保 MC 路径可见 UC 路径刚写的数据

**消费者（AR warp group）**：
1. `elect_one()` + `spin_wait(flag == num_ranks)` — 一个线程等 flag
2. `arrive_and_wait()` — CTA 内 barrier，召集 AR warp 所有线程
3. `multimem.ld_reduce` + `multimem.st` — 开始 AllReduce

对比：one-shot 假设 GEMM 进度在所有 rank 严格同步，不做 flag 检查，适用于小数据量。two-shot 多一步 flag 开销，换确定性正确。

### 4. 任务划分（Task Partition）

核心思想：不要把完整的 M×N tile 交给每个 rank 做重复 reduce。

将 M 维度切成 num_ranks 份（切 M 不是切 N，因为行主序下切 M 保证数据连续访问，TMA 和 multimem 的 DMA 效率高）：

```python
m_local_rank = cta_mma_tile_m / num_ranks  # 每个 rank 负责 M 行的 1/4

# CuTe DSL 切分 tensor
tCgC_mc_slice_partitioned = cute.zipped_divide(tCgC_mc_slice, (m_local_rank, N))
tCgC_mc_local_rank = cute.slice_(tCgC_mc_slice_partitioned, ((None, None), (rank_id, 0)))
```

每个 rank 只负责自己的 1/4 → ld_reduce 只处理这部分 → st broadcast 自己的 1/4 → 最后每张卡都有完整 tile。reduce 工作被摊开，NVSwitch 带宽充分利用。

128-bit 处理粒度由 multimem PTX 指令的硬件限制决定——每次 ld_reduce / st 操作数就是 128 bits = 8×BF16 = 4×FP32 = 16×FP8。

### 5. 三层同步

#### 第一层：Tile pre-sync

每个 tile 的 epilogue 写完后，通过 MC flag `multimem.red.add` 通知其他 rank。AR warp spin 等 flag == num_ranks。scope="gpu"（本 GPU 内协调），order="relaxed"（数据完整性已有其他 fence 保证）。

#### 第二层：Tile 内数据并行

每个线程只处理自己的 128-bit 数据，无跨线程依赖。不需要元素间额外同步。

#### 第三层：Kernel 尾部 final sync

所有 SM 的 AR 工作完成后，确保所有 `multimem.st` broadcast 对其他 GPU 可见。使用 Per-SM flag（`barrier_flag + last_tile_id + sm_id_linear`），scope="sys"（跨所有设备），order="release"（写入方）/"acquire"（等待方），确保跨 GPU memory ordering。

Scope 语义：
- `scope="gpu"`：本 GPU 所有 SM、copy engine 可见。用于 tile ready flag——AR warp 在本 GPU 读本 GM 的 flag 副本（NVSwitch 已同步）。
- `scope="sys"`：所有 GPU + CPU 可见。用于 final sync——后继 kernel 在其他 GPU 上运行，必须看到前序 kernel 的 broadcast 结果已落地。

## 多角度追问

> [!question] 一个 CTA 里有多组 warp，为什么是"多放一个 AR warp group"而不是"多放一个 AR warp"？
> 一个 warp 只有 32 线程，处理 tile 的 128-bit 数据流水线需要多线程并行。AR 工作需要足够线程覆盖 N 方向，通常需要多个 warp 组成 AR warp group。

> [!question] multimem.ld_reduce 的数据从哪来？
> 从各 rank 的 GM（通过 MC 地址）。之前 epilogue 通过 TMA store 把 tile 从 shared memory 写到了 UC 地址（local GM）。multimem 碰的是 GM，不碰 shared memory 和 register。

> [!question] 为什么 two-shot 有两个 PR 展示不同角色？
> FlashInfer PR #1695 展示 epilogue 侧的 arrive 代码（生产者），SGLang PR #15103 展示 AR warp 侧的 spin wait 代码（消费者）。它们是同一个同步协议的互补双方。

> [!question] NCCL 还有用吗？
> 有。NCCL 适合 chunk grain 的大数据量 AllReduce（梯度同步、AllGather KV），启动开销可摊薄。SHARP 适合 tile grain 的零启动开销即时 reduce，不可替代。

## 边界与极端情况

- **one-shot vs two-shot 切换**：数据量小时 one-shot 够了（tile 生产同步）；数据量大时 TMA store 延迟不均，必须 two-shot 的 flag 检查。
- **final sync 的 race condition**：如果所有 SM 共用一个全局 final flag，SM 0 先完成 +1，但 SM 5 还在 broadcast 老 tile——后继 kernel 读到半成品。Per-SM flag 排除此窗口。
- **scope 混用风险**：tile flag 用 `scope="sys"` 会引入不必要的跨设备开销；final sync 用 `scope="gpu"` 会导致后继 kernel 在 GPU 1 上看不到 GPU 0 的 flag 更新——正确性 bug。

## 参考文献 & 延展学习

| 来源 | 类型 | 难度 | 链接 |
|------|------|------|------|
| 杨轶凡《GEMM+AR 融合》系列 | 知乎文章 | advanced | https://zhuanlan.zhihu.com/p/2035845795082133865 |
| SGLang PR #15103 | PR (two-shot tile sync) | advanced | github.com/sgl-project/sglang/pull/15103 |
| FlashInfer PR #1695 | PR (CuTe DSL two-shot) | advanced | github.com/flashinfer-ai/flashinfer/pull/1695 |
| FlashInfer PR #2171 | PR (barrier fix) | advanced | github.com/flashinfer-ai/flashinfer/pull/2171 |
| CUTLASS Distributed README | 官方文档 | intermediate | github.com/NVIDIA/cutlass |
| NVIDIA NVLink SHARP 白皮书 | 官方文档 | advanced | nvidia.com |
