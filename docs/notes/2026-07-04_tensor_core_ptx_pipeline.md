---
title: Tensor Core PTX 指令管线全拆解
aliases:
  - 'tensor_core_ptx_pipeline'
  - 'Tensor Core PTX 指令管线全拆解'
date: 2026-07-04
tags:
  - type/lecture
  - domain/llm-inference
  - topic/gpu-kernel
  - topic/cuda
  - topic/gemm
source_count: 5
status: stable
difficulty: intermediate
---

# Tensor Core PTX 指令管线全拆解

> 从 shared memory 到 Tensor Core 再到写回：ldmatrix、mma.sync、stmatrix、wgmma 的全链路推演。覆盖跨线程数据搬移、bank conflict 解决、架构代际跃迁。

## 场景引入

想象一个快递分拣中心。卡车上卸下来的货（global memory）先放到分拣台（shared memory），然后分拣员把货装到 32 个小推车上（32 个线程的寄存器），再送进 4 个加工机器（4 个 Tensor Core），加工完的结果放到小推车上，最后装车发走。

Pre-Hopper 的分拣员叫 `ldmatrix`，卸货的叫 `cp.async`。Hopper 把分拣员和加工机器合并成了一个全能工 `wgmma`，卸货也升级成了自动化传送带 `TMA`。

这个比喻贯穿整篇文章。

## 前置概念解释

在往下读之前，确保你已经理解：

| 概念 | 一句话 | 详情 |
|------|--------|------|
| **Warp** | 32 个线程的调度单元 | SIMT 锁步执行 |
| **Shared Memory Bank** | 32 个 4-byte 宽的并行存储体 | 同周期同 bank = conflict |
| **Crossbar** | SM 内部的全互联网络 | warp shuffle、ld/st matrix 都用它 |
| **Tensor Core** | 矩阵乘法专用硬件 | 一个 cycle 算 M×N×K 的乘加 |
| **MMA** | Matrix Multiply Accumulate | PTX 指令，`mma.sync`（SM_70+）或 `wgmma`（SM_90+） |
| **LSU** | Load Store Unit | 执行内存指令的硬件单元 |
| **TMA** | Tensor Memory Accelerator | Hopper 的硬件异步搬移单元 |

## 核心讲解：Pre-Hopper vs Hopper 两条管线

### 结论先行：两条管线的本质区别

**Pre-Hopper（SM_75-80）**：每步都是显式指令。shared memory 搬到寄存器（ldmatrix）、寄存器进 Tensor Core（mma.sync）、结果写回 shared memory（手动 shuffle + store），分三步走。

**Hopper（SM_90）**：`wgmma` 一条指令 = 读 shared + 算，`stmatrix` 把写回也硬化了。`TMA` 异步搬 global→shared。全链路异步、解耦。

```
SM_75-80 (Turing/Ampere)

  global mem
      │ cp.async（显式指令，程序员写循环地址计算）
      ▼
  shared mem ────── swizzle 布局（见下文 bank conflict 讨论）
      │ ldmatrix（显式指令，8→32 fanout）
      ▼
  register (per-thread)
      │ mma.sync（显式指令）
      ▼
  register (accumulator)
      │ __shfl_sync + st.shared（手动！）
      ▼
  shared mem → global mem


SM_90 (Hopper)

  global mem
      │ TMA（硬件自动：地址计算 + 边界检查 + swizzle）
      ▼
  shared mem ────── wgmma descriptor（矩阵描述符，非实际数据）
      │ wgmma.mma_async（一条指令 = 内部读 shared + 计算）
      ▼
  register (accumulator)
      │ stmatrix（硬化）或 TMA store
      ▼
  shared mem → global mem
```

### ldmatrix 拆解

`ldmatrix` 做了三件事：
1. 8 个线程从 shared memory 各读 128-bit 行
2. 通过 crossbar 把数据 fanout 到 32 个线程的寄存器
3. 满足 `mma.sync` 的寄存器分布要求

**最重要的误区**：`ldmatrix.x4` 不是 32 线程同时读。官方确认：x4 = 4 笔串行 transaction，每笔 8 线程：

```
Tx1: T0-T7 读 matrix0（top-left 8×8）→ fanout 到 32 线程
Tx2: T8-T15 读 matrix1（bottom-left 8×8）→ fanout
Tx3: T16-T23 读 matrix2（top-right 8×8）→ fanout
Tx4: T24-T31 读 matrix3（bottom-right 8×8）→ fanout
```

每笔 8 线程 × 16 bytes = 128 bytes = 32 banks × 4 bytes，**恰好填满 banks。**

> [!insight] 关键推论
> 因为每笔 transaction 只有 8 个线程参与，所以 **8 × 4 banks = 32 = bank 总数 → 可以做到零 conflict**。bank conflict 不是必然的！

### Bank Conflict 数学证明

#### 连续 16×16 FP16 布局：有冲突

每行 stride = 32 bytes = 8 banks：

```
r0: bank 0   r1: bank 8   r2: bank 16  r3: bank 24
r4: bank 0   r5: bank 8   r6: bank 16  r7: bank 24
```

8 行 → 4 个唯一起始 bank → 2-way conflict。
ncu 实测：`data_bank_conflicts.sum = 4`（4 笔 × 1 次冲突）。

#### Padding 后：零冲突

`half[16][16]` → `half[16][16+8]`，stride = 48 bytes = 12 banks：

```
r0: bank 0   r1: bank 12  r2: bank 24  r3: bank 4
r4: bank 16  r5: bank 28  r6: bank 8   r7: bank 20
```

**8 个起始 bank 全部唯一！** ncu 实测：`sum = 0`。

#### 数学本质

需要找到 stride S 使 `{(i × S) mod 32 | i = 0..7}` 这 8 个起始 bank 互不重复。连续布局 S=8 → 重复周期 4。Padding 使 S=12 → 8 和 32 互质 → 周期 = 8 → 全部唯一。

### stmatrix 拆解

`stmatrix` 是 `ldmatrix` 的镜像，做 32→8 gather + assemble + store：

1. 8 个目的线程从 32 个源线程各收 2 个 16-bit 元素（通过 crossbar）
2. 拼成完整 128-bit 行
3. 用自己的地址寄存器写入 shared memory

**为什么 SM_90 才引入**（比 ldmatrix 的 SM_75 晚一代）：
- Fanout（scatter）硬件简单：源线程有完整数据
- Gather + assemble：需要多 lane 汇聚再拼装 → 更强的 crossbar
- Hopper 的 DSMEM 架构天然适合这个新的 crossbar 通路

### wgmma：Hopper 的质变

wgmma 一条指令 = 读 shared + 计算。程序员给的是 **matrix descriptor**（描述数据在 shared memory 的地址和 layout），不是实际数据寄存器。Tensor Core 内部自己从 shared memory 读。

**wgmma 不用 stmatrix**。wgmma 的输出是寄存器。epilogue 直接从寄存器写 global memory（TMA store）。CUTLASS 3.x 的标准 GEMM 不经过 shared memory 中转 accumulator。

| 对比 | mma.sync（pre-Hopper） | wgmma（Hopper） |
|------|----------------------|----------------|
| 执行单元 | 1 warp（32 线程） | 1 warp group（128 线程） |
| 输入 | 寄存器（ldmatrix 加载的） | shared memory descriptor |
| 输出 | 寄存器 | 寄存器 |
| epilogue 写回 | 手动 shuffle + st.shared | stmatrix 或 TMA store 直接写 global |
| 异步模型 | 同步（需 fence） | 异步（commit + wait） |

## 多角度追问

> [!question] 追问 1：32 个线程 × 4 banks = 128 > 32，bank conflict 必然吗？
> **不必然**。前提就是错的——不是 32 线程同时读，是 4 笔串行 transaction，每笔只有 8 线程。8 × 4 = 32 = bank 总数。只要 8 个起始 bank 全不同 → 零 conflict。

> [!question] 追问 2：ldmatrix 加载时，前 8 个线程已经把 32 个 bank 占光了，后面 24 个线程不管选哪个 bank 都冲突吧？
> 这是混淆了跨 transaction 和跨 cycle。bank conflict = 同 cycle 内。4 笔 transaction 不同 cycle → 不冲突。这跟「一行只需要 8 个线程提供地址，但需要 32 个线程各提供自己的地址」是两个概念——32 个线程都提供地址，但分 4 波发出。

> [!question] 追问 3：B 矩阵在 shared memory 里存在 row-major 还是 column-major？
> 都可以，取决于你用哪种 ldmatrix：
> - shared memory 存 row-major + `ldmatrix.trans`（转置搬入寄存器）
> - shared memory 存 column-major + 普通 `ldmatrix`（把 16×8 col-major 当成 8×16 row-major 处理）
> 工程上通常选 column-major，因为 B 沿 K 方向连续 → global→shared 搬移 coalesced → 快。

> [!question] 追问 4：为什么 CUTLASS 3.x 的 wgmma GEMM 不用 stmatrix？
> 因为 wgmma 的结果在寄存器里，epilogue 用 TMA store 直接从寄存器写到 global memory。不需要 shared memory 中转 accumulator。stmatrix 在 Hopper 上主要用于 mma.sync 管线（向后兼容）和需要 cross-warp reduction 的 epilogue。

> [!question] 追问 5：「行地址可以不连续」到底有什么用？
> 不是用来躲 bank conflict 的（bank conflict 用 padding/swizzle 解决）。它是用来在 swizzle 之后正确指到每一行的——swizzle 后同一 tile 的 8 行物理上不再连续，ldmatrix 的独立地址设计恰好匹配这个需求。

## 架构代际对比

| | SM_75-80 (Turing/Ampere) | SM_90 (Hopper) |
|---|---|---|
| 数据搬到 shared | cp.async（手动循环） | TMA（硬件自动） |
| shared→register | ldmatrix（显式指令） | wgmma 内部隐含 |
| 矩阵计算 | mma.sync | wgmma.mma_async |
| register→shared（epilogue） | __shfl_sync + st.shared（手动） | stmatrix（硬化指令） |
| shared→global（epilogue） | st.global | TMA store |
| 编程模型 | 同步 | 异步（producer-consumer warp specialization） |

## 总结

> [!insight] 三个最关键的认识

1. **ldmatrix.x4 ≠ 32 线程同时读**。是 4 笔串行 transaction，每笔 8 线程。这是理解 bank conflict 的前提。8×4=32=banks，可以零冲突。

2. **跨线程数据搬移靠 crossbar**。ldmatrix 做 8→32 fanout（容易），stmatrix 做 32→8 gather+assemble（难，SM_90 才硬化）。

3. **Hopper 的 wgmma 是一条包含「读+算」的指令**，程序员只给 shared memory descriptor。ldmatrix 不再显式出现，stmatrix 也只在特定场景用到。全链路异步化。

## 延展学习

| 来源 | 类型 | 说明 |
|------|------|------|
| [stmatrix 与 ldmatrix](https://zhuanlan.zhihu.com/p/1892538766796780877) | 知乎文章 | 原始文章，PTX 指令语法详解 |
| [ldmatrix 与 swizzle 笔记](https://zhuanlan.zhihu.com/p/696231622) | 知乎文章 | 8-way bank conflict + swizzle 构造原理 |
| [ldmatrix bank conflict ncu 实测](https://zhuanlan.zhihu.com/p/27488669542) | 知乎文章 | padding/swizzle 消除 conflict 的 ncu 数据 |
| [NVIDIA Developer Forum: ldmatrix SMEM](https://forums.developer.nvidia.com/t/278716) | 官方论坛 | 确认 x4 = 4 transactions、wavefront 计数 |
| [CUTLASS WGMMA Tutorial (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | 教程 | Hopper wgmma 完整使用指南 |
| [Nvidia Hopper WGMMA 计算分析](https://zhuanlan.zhihu.com/p/12481535419) | 知乎文章 | wgmma 硬件实现猜测 + cycle 计算 |

## 参考来源

1. 知乎文章 [stmatrix 与 ldmatrix](https://zhuanlan.zhihu.com/p/1892538766796780877) — PTX ldmatrix/stmatrix/mma 语法及完整用法
2. 知乎文章 [ldmatrix bank conflict 问题](https://zhuanlan.zhihu.com/p/27488669542) — ncu bank conflict 实测数据
3. 知乎文章 [ldmatrix 与 swizzle 笔记](https://zhuanlan.zhihu.com/p/696231622) — swizzle 构造原理
4. NVIDIA Developer Forum [ldmatrix shared memory access](https://forums.developer.nvidia.com/t/278716) — 官方确认 transaction 模型 + wavefront 计数
5. CUTLASS WGMMA Tutorial [Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) — wgmma 特性、数据流、epilogue
6. CUTLASS CUTE Hopper 特性分析 [DeclK's Blog](https://declk.github.io/blog/CUDA/CUTLASS%20CUTE%205%20Hopper%20%E7%89%B9%E6%80%A7%EF%BC%9Atma%20%26%20wgmma.html) — TMA、wgmma 在 CUTE 中的使用
