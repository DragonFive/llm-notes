---
title: WGMMA Hopper GEMM 完整讲解
aliases:
  - 'wgmma_hopper_gemm'
  - 'WGMMA Hopper GEMM 完整讲解'
date: 2026-07-02
updated: 2026-07-02
tags:
  - type/lecture
  - domain/llm-inference
  - topic/gpu-kernel
  - topic/cuda
  - topic/gemm
source_count: 1
status: draft
difficulty: advanced
---

# WGMMA Hopper GEMM 完整讲解

> 如何在 Hopper 架构 GPU 上使用 CUTLASS 的 WGMMA 指令实现高效矩阵乘法。

## 场景引入

你拿到一张 H100，要写一个矩阵乘法 kernel。Ampere 时代你用 `mma.sync`，同步阻塞，一个 warp 32 线程就能搞定。但 Hopper 给了你一把新武器——`wgmma.mma_async`，4 个 warp 协同执行，异步发射。怎么用？有哪些坑？这篇讲解自底向上，从硬件原语到 CUTLASS 封装再到同步机制，把整条链路串起来。

## 前置概念解释

| 术语 | 一句话解释 |
|------|----------|
| Warpgroup | 4 个连续 warp = 128 线程，WGMMA 的执行单元 |
| WGMMA | Hopper 的异步 warpgroup 级矩阵乘加指令 |
| MMA Atom | CUTLASS 对一条 WGMMA 指令的封装 |
| TiledMMA | CUTLASS 对多个 MMA Atom 的编排（多少个 warpgroup、怎么切分输出） |
| Major | 内存布局方向：MN-major（M/N 维连续）或 K-major（K 维连续） |
| Swizzle | SMEM 地址变换，避免 bank conflict |
| Core matrix | WGMMA atom tile 中最小的数据单元（8B × 16B） |
| Matrix descriptor | 64bit 寄存器数据，描述 WGMMA 要读取的 SMEM 数据格式 |

## 核心讲解

### 结论先行：WGMMA 的完整调用链

```
构造 TiledMMA（host 端）
  → 构造 SMEM layout（满足 Major + Swizzle 约束）
  → partition_A/B/C（每个线程获取自己的 view）
  → warpgroup_arrive() → cute::gemm() → warpgroup_commit_batch() → warpgroup_wait<0>()
  → 结果在 tCrC 中
```

### 展开解释

#### 1. Warpgroup 基础

一条 `wgmma.mma_async` 需要 128 个线程（4 个连续 warp）**协同执行**。这不是"128 个线程各算各的"，而是 128 个线程共同完成一个 tile 的乘加。

**存储约束**：
- B 必须在 SMEM（硬件限制）
- A 可以在 SMEM 也可以在 RMEM
- C（累加器）必须在 RMEM

#### 2. 构造 TiledMMA

```cpp
// 最简单的单 warpgroup 示例
TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});
```

**MMA Atom 命名拆解**：

```
SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>
 │     │  │  │   │   │   │   │  └── B 在 SMEM
 │     │  │  │   │   │   │   └──── A 在 SMEM
 │     │  │  │   │   │   └──────── C 的类型 = FP16
 │     │  │  │   │   └──────────── B 的类型 = FP16
 │     │  │  │   └──────────────── A 的类型 = FP16
 │     │  │  └──────────────────── K = 16
 │     │  └─────────────────────── N = 64
 │     └────────────────────────── M = 64
 └──── SM90 = Hopper 架构
```

**三套记号映射**（核心易混淆点）：

| A 的 Major | B 的 Major | BLAS 记号 | tnspA | tnspB |
|-----------|-----------|----------|-------|-------|
| MN | MN | NT | 1 | 1 |
| K | K | NN | 0 | 0 |
| MN | K | NR | 1 | 0 |
| K | MN | TN | 0 | 1 |

关键理解：BLAS 的 N/T 是相对"列主默认值"定义的，CUTLASS 用 Major 描述内存布局，两者是同一件事的不同视角。`tnspA/tnspB` 是最终编码进 PTX 指令的立即数。

**多 Warpgroup**：`Layout<Shape<_2,_1,_1>>` 表示沿 M 维度放 2 个 MMA Atom = 2 个 warpgroup。`size(tiled_mma) = 2 × 128 = 256`。

#### 3. SMEML 布局约束

WGMMA 要求 SMEM 中的 A、B 使用特定的 swizzled layout，不能是简单的行主/列主。

**8 种 Layout Atom** = 2 种 Major × 4 种 Swizzle：

```cpp
// MN-major + SW128
auto sA = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<T>{},
    cute::make_shape(bM, bK, bP));
```

Swizzle 大小由连续方向上的数据量决定：FP16 时 64 个元素 = 128 bytes → SW128。

#### 4. Fragment 与 Matrix Descriptor

```cpp
ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);  // SMEM 中的 view
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // matrix descriptor（不是数据拷贝！）
```

**关键**：`tCrA` 和 `tCrB` **不是**把 SMEM 数据拷贝到寄存器，而是一个 64bit 的 **matrix descriptor**，描述 WGMMA 要读取的 SMEM 数据格式。CUTLASS 内部的 `make_gmma_desc` 根据布局自动计算 LBO、SBO、swizzle 等参数。

#### 5. 同步机制

WGMMA 是异步指令，需要四步同步协议：

```cpp
cute::warpgroup_arrive();       // wgmma.fence.sync — 确保 SMEM 写入可见
cute::gemm(tiled_mma, ...);     // wgmma.mma_async — 发射异步计算
cute::warpgroup_commit_batch(); // wgmma.commit_group — 打包成 wgmma-group
cute::warpgroup_wait<0>();      // wgmma.wait_group 0 — 等待计算完成
```

| cute 方法 | PTX 指令 | 作用 |
|-----------|---------|------|
| `warpgroup_arrive()` | `wgmma.fence.sync.aligned` | 确保 warpgroup 内的 RMEM/SMEM 写入对 WGMMA 可见 |
| `warpgroup_commit_batch()` | `wgmma.commit_group.sync.aligned` | 将自上次 commit 以来的所有 mma_async 打包成一个 group |
| `warpgroup_wait<N>()` | `wgmma.wait_group.sync.aligned N` | 等待直到只剩 ≤N 个 wgmma-group 未完成 |

> [!tip] fence.proxy.async 的使用条件
> WGMMA 在 async proxy 中执行。如果你用 `ld.global/st.shared` 加载数据到 SMEM，需要额外发射 `fence.proxy.async` 使 generic proxy 的写入对 async proxy 可见。如果用 TMA load，则不需要——TMA 直接在 async proxy 中操作。

### 深入细节：Core Matrices

WGMMA atom 的输入 tile（wA、wB）被划分为 **core matrices**——最小的硬件数据单元：

- 每个 core matrix：strided 方向 8 bytes × contiguous 方向 16 bytes
- wA = 8×2 个 core matrices
- wB = 2×(N/8) 个 core matrices

Matrix descriptor 的 5 个参数精确描述了 core matrices 在 SMEM 中的排列方式：

| 参数 | 含义 |
|------|------|
| 开始地址 | SMEM 基地址 |
| LBO | K 方向相邻 core matrices 的字节间距 |
| SBO | M 方向相邻 core matrices 的字节间距 |
| Swizzle mode | 地址变换模式 |
| Matrix base offset | 对齐补偿 |

## 多角度追问

> [!question] 为什么 WGMMA 需要 128 线程？
> Tensor Core 的计算单元设计为一次处理一个 64×64×16 的 tile。128 线程各持有 32 个累加器元素，2×2×8 = 32，刚好覆盖 64×64 输出的所有位置。

> [!question] 如果 A 在 RMEM，为什么 B 不能也在 RMEM？
> 硬件限制。WGMMA 指令编码中，B 的操作数只能通过 SMEM 地址（通过 matrix descriptor）引用，没有"从寄存器读取 B"的数据通路。A 有 RMEM 变体（RS）是因为 A 的数据可能已经在寄存器中（如前一轮计算的中间结果）。

> [!question] warpgroup_wait 的参数 N 有什么用？
> N 允许"计算与等待重叠"。N=0 表示等当前 group 完成；N=1 表示允许 1 个 group 还在执行就继续。FlashAttention-3 就用了 GEMM-softmax 重叠策略，利用 N>0 隐藏 softmax 的延迟。

> [!question] Swizzle 的大小怎么选？
> 主要由 bank conflict 避免需求决定。更大的 swizzle 模式能避免更多 bank conflict，但会增加 SMEM 地址计算的复杂度，且可能导致布局不 compact。实际中 SW64 和 SW128 是最常用的选择。

## 边界与极端情况

1. **M=1 的 GEMM**：WGMMA 的 M 固定为 64，M=1 时需要 split-K 或用其他策略
2. **非 16bit 类型**：FP8/INT8 等 <16bit 类型必须使用 K-major layout（硬件限制）
3. **多 warpgroup 的线程映射**：线程 0~127 总是 warpgroup 0，128~255 是 warpgroup 1——CTA 中前 256 个线程固定分配，不是动态调度

## 参考文献 & 延展学习

| 来源 | 类型 | 难度 | 链接 |
|------|------|------|------|
| CUTLASS WGMMA 教程（原文） | 代码 | advanced | [GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu) |
| CUTLASS GEMM Tutorial | 文档 | intermediate | [GitHub](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md) |
| 知乎翻译文章 | blog | intermediate | [链接](https://zhuanlan.zhihu.com/p/28622739849) |
| FlashAttention-3 | 论文 | advanced | Hopper 架构上利用 WGMMA 异步性的 attention 实现 |
| PTX ISA 文档 | 官方文档 | advanced | wgmma.mma_async 指令完整规范 |
