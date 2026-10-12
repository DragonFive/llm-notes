---
title: PDL vs CUDA Graph vs Piecewise CUDA Graph — LLM 推理 kernel launch overhead 优化全景
aliases:
  - 'pdl_vs_cudagraph'
  - 'PDL vs CUDA Graph vs Piecewise CUDA Graph — LLM 推理 kernel launch overhead 优化全景'
date: 2026-06-22
tags:
  - type/lecture
  - domain/llm-inference
  - topic/gpu-kernel
  - topic/optimization
source_count: 2
status: stable
difficulty: intermediate
---

# PDL vs CUDA Graph vs Piecewise CUDA Graph — LLM 推理 kernel launch overhead 优化全景

> 完整讲解：从杨轶凡知乎回答的"PDL 调好了没"开始，到 vLLM 0.6+ 的 Piecewise CUDA Graph，逐层讲清楚 LLM 推理中砍掉 CPU launch overhead 的三代技术。

## 场景引入

想象你在餐厅点菜：

- **传统方式（无优化）**：你点一份炒饭 → 等厨师做完 → 端上来 → 你再点一杯饮料 → 等 → 端上来 → 你再点一份汤。每点一道你（CPU）都要等厨房（GPU）做完才能点下一道。**这就是 Unfused kernel 的 launch overhead**——CPU 在中间干等。
- **PDL 优化**：你跟服务员说"炒饭、饮料、汤按顺序来"，服务员把订单发给厨房，自己去招呼其他客人了。**但你还是要先跟服务员说第一句话**（CPU 启动第一个 kernel）。
- **CUDA Graph 优化**：你把整桌菜都点完（一次性录制），之后每次按 play，**连服务员都不用叫**，厨房自己按顺序出菜。
- **Piecewise CUDA Graph 优化**：菜单里有几道菜"做法固定"（稳定的算子），有几道"看当天食材"（动态的算子）。把固定的菜用 Graph 流程跑（按 play 就行），动态的菜单独点单。

**核心问题**：什么时候用哪种方案？为什么业界最终选了 Piecewise？

---

## 前置概念解释

### kernel 是什么？

GPU 上的一次"操作任务"。一次 LLM 推理 = 几十到几百个 kernel 排队执行。比如一个 Transformer 层 ≈ 12 个 kernel（QKV 投影、Attention 计算、Output 投影、FFN 两层、LayerNorm × 2、Residual × 2...）。

### launch overhead 是什么？

CPU 把任务"发射"给 GPU 的成本。每次发射要等 GPU 报告"上一个任务完成了"，CPU 再发射下一个。这个等待 ≈ 5-10 μs。

**为什么是大问题？**
- 单个 kernel 跑 50-100 μs 时，launch overhead 占比 5-10%
- 推理时延敏感场景（如流式输出），每一步推理都涉及多次 launch
- 1000 个 token 输出 × 10 次 launch/步 × 5 μs = 50 ms 纯浪费

### 什么是"图要重录"？

CUDA Graph 录制时冻结 4 类信息：
1. **kernel launch 参数**（block size、grid size、shared memory）
2. **tensor 形状**（shape、stride、dtype）
3. **kernel 之间的数据依赖**
4. **输入输出 tensor 的内存地址**

**这 4 类信息里任一变，Graph 就要重录。** 变了之后三种处理方式：

| 方式 | 适用 | 代价 |
|------|------|------|
| 整图重录 | 变化频率低 | 几十 μs 到几 ms |
| **Piecewise** | 变化有档位 | 预录 N 个，按需选 |
| **PDL** | 放弃录图 | 只砍 CPU 等待，不砍 launch |

---

## 三代技术对比（核心讲解）

### L1：PDL（Programmatic Dependent Launch）

**核心思想**：让 GPU 自己管 kernel 之间的依赖图，CPU 不用在中间等。

**API**：`cudaLaunchKernelEx`（CUDA 12.3+），通过 `cudaLaunchAttribute` 设置 `cudaLaunchAttributeProgrammaticStreamSerialization`。

**适合场景**：
- 团队没有 profiling 能力
- 业务请求分布太散
- 不想录图但要砍 launch overhead

**局限**：
- CPU 还是要启动第一个 kernel
- 每个 launch 都要配置依赖
- 不砍 kernel 启动，只砍 CPU 等待

---

### L2：CUDA Graph（整图录制）

**核心思想**：把整段推理的 kernel 序列一次性录制成 DAG，整段重放，**CPU 完全不参与**。

**API**：`torch.cuda.graph()` 或 `cudaGraph_t` C API。

**适合场景**：
- 单次推理延迟敏感（≤10ms）
- 高频 batch 集中
- 计算量相对小

**硬约束**（图要重录的 4 个条件）：
- kernel launch 参数变
- tensor shape 变
- 数据依赖变
- 内存地址变

---

### L3：Piecewise CUDA Graph（vLLM 0.6+ 主推）

**核心思想**：把大图按算子切分成子图，**稳定的算子块录进 Graph，动态的算子留在 eager 模式**。

**两种 Piecewise 含义**（容易混淆）：

| 类型 | 含义 | 时代 |
|------|------|------|
| **老 Piecewise** | 按 batch/seq_len 分桶，每桶一个图 + padding | v0 时代 |
| **vLLM compilation Piecewise** | 按算子切分大图为子图，每子图独立录制 | vLLM 0.6+ |

**vLLM 0.6+ 的真实做法**：

```
整段推理计算图
    ↓ vLLM compilation pass（找切分点）
    ↓
    切分 (PIECEWISE 切分点)
    ↓           ↓           ↓
 子图 A      子图 B      子图 C
(Linear+LN)  (Attention) (Residual)
    ↓           ↓           ↓
 CUDA Graph   Eager      CUDA Graph
 (录一次)    (动态跑)    (录一次)
```

**核心 Pass**：
1. **图分割 Pass**：用动态规划找最优切分点
2. **算子融合 Pass**：把多个小算子合成一个（LN + Linear → 一层）
3. **编译后端 Pass**：Triton / Inductor / 自定义

**为什么 Attention 排除在外？**
- Attention grid 与 seq_len² 相关，shape 变化最剧烈
- 录制成本 = 录 N² 个变体
- KV cache 长度每步变

---

## 4 种"动态"维度拆解

LLM 推理里有 4 种"动态"，常被混在一起说：

| 维度 | 例子 | 整图能用？ | 工业方案 |
|------|------|----------|---------|
| **D1 调度动态** | batch size 变化 | ❌ | Piecewise 按 batch 桶 |
| **D2 长度动态** | prefill seq_len 变化 | ❌ | Piecewise 按 seq_len 桶 + Chunked Prefill |
| **D3 路由动态** | MoE 专家选择 | ✅ | 整图（kernel 不看路由结果） |
| **D4 数据依赖动态** | spec dec 接受/拒绝 | ❌ | 分支子图 |

**关键洞察**：
- D3 的"动态"在**数据内容**层，不在**形状**层——kernel 接收的 input shape 不变
- vLLM 实际用 Piecewise 把稳定的算子块录进 Graph，动态的 Attention 留在 eager
- **不是"按 D1/D2 分桶"**，而是"按算子切分点切分"

---

## MoE 推理的 kernel-by-kernel 分析

把 MoE 的 dispatch → FC1 → FC2 → combine 拆到 kernel 级别：

### Kernel 1: Router（门控网络）

```python
# 输入: (batch, hidden_dim)
# 输出: (batch, num_experts)  # 每个 token 对每个专家的分数
```

- 形状稳定性：⚠️ 跟 D1 走
- 能不能录 Graph：⚠️ 需要 Piecewise 按 batch 桶

### Kernel 2: Dispatch（把 token 路由到专家）

```python
# 输出形状: (num_dispatched_tokens, hidden_dim)  ← 这个 shape 每次都不同！
```

- 形状稳定性：❌ 不稳定（每个专家分到的 token 数不同）
- 能不能录 Graph：❌ **不能**

### Kernel 3-4: FC1 / FC2（专家 GEMM）

```python
# 输入: (tokens_for_this_expert, hidden_dim)
```

- 形状稳定性：❌ M 维度不固定
- 能不能录 Graph：❌ **不能**（原始实现）

**Grouped GEMM 解法**：
- 把 num_experts 个不同 M 的 GEMM **打包成一次 launch**
- 输入是 `(M_total, K)`，M_total 固定
- **能被 Graph 录制** ✅

### Kernel 5: Combine

- 形状稳定性：⚠️ 跟 D1 走
- 能不能录 Graph：⚠️ Piecewise 按 batch 桶

### 现代 vLLM MoE 优化栈

```
请求进来
    ↓
Router (CUDA Graph - 按 batch 桶)
    ↓
Top-K 路由 (Eager - shape 真动态)
    ↓
Grouped GEMM (CUDA Graph - 按 M_total 桶) ← gmm 关键
    ↓
Activation (CUDA Graph - shape 固定)
    ↓
Grouped GEMM 第二层 (CUDA Graph)
    ↓
Combine (CUDA Graph - 按 batch 桶)
```

---

## 多角度追问

> [!question] Q1：权重在变（动态加载/LoRA 切换）怎么办？
> 答：分两种情况——
> - 权重 shape 不变，只换内容：CUDA Graph 仍能录。Graph 锁的是地址，不是内容。
> - 权重 shape 变：整图废。要么录多份图，要么重新录制（几 ms 代价）。

> [!question] Q2：Megablocks / ScatterMoE 是什么？
> 答：Grouped GEMM 的"更彻底"版本——把 MoE 的 block-sparse pattern 暴露给硬件，用 block-sparse matmul 替代 dense GEMM。算力利用率更高，但工程复杂。

> [!question] Q3：跨节点 MoE 怎么办？
> 答：专家并行（Expert Parallelism, EP）——把 num_experts 分散到 N 个节点。需要 all-to-all 通信。CUDA Graph 录不进去（跨节点通信是硬约束）。这一段用 NCCL 单独优化。

> [!question] Q4：未来会怎么走？
> 答：三个方向——
> 1. CUDA Graph + 动态专家（gmm 这条路）—— vLLM/SGLang 在走
> 2. Megakernel MoE —— 绕开 Graph 的所有限制（但回到杨轶凡的 ablation 批评）
> 3. 稀疏化 + 量化 —— 减少算力需求，让 Graph 录制成本可接受

> [!question] Q5：那 PDL 现在还有用吗？
> 答：**基本没用**。vLLM 0.6+ 已切到 Graph + 切分范式。PDL 是给"自研推理引擎 + 没有 compilation 框架"的团队准备的退路。

---

## 边界与极端情况

| 情况 | 推荐方案 |
|------|---------|
| 单 GPU + 小模型 + 高 QPS | CUDA Graph 整图 |
| 单 GPU + MoE + 动态 batch | Piecewise + Grouped GEMM |
| 多 GPU + MoE + 专家并行 | Piecewise + Grouped GEMM + NCCL（all-to-all 不进 Graph） |
| spec decoding + MoE | 分支子图 + Grouped GEMM（极复杂） |
| 团队无 profiling 能力 | PDL（last resort） |

---

## 总结

**一句话**：

> **PDL 是上一代技术，CUDA Graph 是终极方案，Piecewise CUDA Graph 是 Graph 在 dynamic shape 下的工程妥协。** vLLM 0.6+ 的 compilation 框架（Inductor + 自定义 pass）把这三条路统一了。MoE 的核心解法是 Grouped GEMM 把动态 M 维度对齐成 M_total，让 Graph 可录制。

**核心修正**（来自本次与 Maxwell 的讨论）：

1. ~~Piecewise = bucket + padding~~ → **Piecewise = 按算子切分大图为子图**（vLLM 0.6+ 含义）
2. ~~vLLM 不用 PDL 因为有 Graph~~ → **vLLM 用 compilation 框架 + Piecewise 切分 + 算子融合 + 多后端编译**
3. ~~D3 整图可以录~~ → **更准确：D3 不影响 Graph 录制，但 vLLM 还是用 Piecewise 切分，区别是切分粒度更细**

---

## 参考文献 & 延展学习

| 来源 | 类型 | 难度 | 链接 |
|------|------|------|------|
| vLLM Piecewise CUDA Graph 知乎文章 | blog | intermediate | https://zhuanlan.zhihu.com/p/1955402895890560120 |
| 杨轶凡 PDL 中文实战 | blog | intermediate | https://zhuanlan.zhihu.com/p/1905474307014370232 |
| vLLM 官方 compilation 文档 | doc | intermediate | https://docs.vllm.ai/en/latest/features/computation_graph_compilation.html |
| vLLM 论文 (SOSP 2023) | paper | advanced | 详细介绍 Continuous Batching + PagedAttention |
| SGLang 论文 (NeurIPS 2024) | paper | advanced | 详细介绍 Piecewise CUDA Graph 在生产环境的应用 |
| MegaBlocks (Gale et al., 2023) | paper | advanced | block-sparse matmul for MoE |
| ScatterMoE (NVIDIA, 2024) | paper | advanced | 改进版 grouped GEMM |
| FlashAttention (NeurIPS 2022) | paper | advanced | 解决 attention 的 padding 问题 |
| PyTorch Inductor CUDA Graph 实现 | code | advanced | `pytorch/torch/_inductor/` |

---

## 关联页面

- 2026-06-22_pdl — 上一代技术，本讲解的前置概念
- 2026-06-22_megakernel — 与本讲解形成对比（绕开 Graph 限制的尝试）
- 2026-06-22_kernel_tuning — L1 硬件层优化
- 2026-06-22_ablation_study — 为什么 Megakernel 论文要被批评
- 2026-06-22_cuda_graph_piecewise — 详细概念卡（待创建）
- 2026-06-22_grouped_gemm — MoE 的关键解法（待创建）
