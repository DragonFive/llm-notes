---
title: Megakernel vs Unfused — 为什么 GPU 推理优化不能迷信"大融合"
aliases:
  - 'megakernel_vs_unfused'
  - 'Megakernel vs Unfused — 为什么 GPU 推理优化不能迷信"大融合"'
date: 2026-06-22
tags:
  - type/lecture
  - domain/llm-inference
  - topic/gpu-kernel
source_count: 1
status: stable
difficulty: beginner
---

# Megakernel vs Unfused — 为什么 GPU 推理优化不能迷信"大融合"

> 完整讲解，由陆铭视角回答 + 小白易懂的比喻展开。

## 场景引入

想象你在指挥一个交响乐团。有两个方案：

**方案 A（Unfused kernel）**：你请四个指挥分别负责木管组、铜管组、弦乐组、打击乐组。每个指挥按自己的节奏行进，你（CPU）在中间协调："木管下去之后，铜管才能上"，然后等一会儿，确认铜管下去，再叫弦乐。

**方案 B（Megakernel）**：你把整个乐团合成一个巨大的超级指挥，他一个人同时控制所有乐器。

方案 B 听起来很酷——一个人干四个人的活，省了中间交接时间。有人测出来 B 比 A 快 30%，写了篇论文，觉得这是个大创新。

但杨轶凡说：**慢着，你怎么知道你这个超级指挥真的比四个指挥快？可能你那四个指挥根本没排练好。**

---

## 前置概念解释

在往下走之前，先把几个陌生词解释清楚。

### kernel 是什么？

一个 kernel 就是在 GPU 上运行的一段程序。你可以理解为 GPU 上的一次"操作任务"——比如 dispatch（分发数据）、FC1（第一层全连接）、FC2（第二层全连接）、combine（合并结果）。

### Unfused kernel 是什么？

Unfused = 不融合。就是把四个操作分别写成四个独立的 kernel，一个接一个执行。每个 kernel 之间有一个 **launch overhead**——CPU 要等上一个 kernel 完成，再发下一个。

### Megakernel 是什么？

Mega = 巨大。就是把 dispatch、FC1、FC2、combine 四个操作全部融合到一个超大的 kernel 里，一次性端到端执行。这样就没有 launch overhead 了。

### 什么是 launch overhead？

CPU 和 GPU 是分离的——CPU 要把任务"发射"给 GPU。每次发射都要等 GPU 报告"上一个任务完成了"，然后 CPU 再发射下一个。这个等待就是 launch overhead。它可以占到几十微秒到百微秒——听起来小，但在低延迟推理中非常显著。

---

## 核心讲解（金字塔结构）

### 结论先行：Megakernel 可能根本没赢，只是没比过

杨轶凡的完整论点：

> 如果发现你的 Megakernel 比 Unfused 快，先检查三个东西：Baseline 是不是太差了？PDL 是不是调好了？Kernel 是不是一致的？**暂时没有看到任何 Megakernel 工作做好 ablation study 的。**

翻译成大白话：Megakernel 跑出来的数字可能是一个精心设计的错觉。它赢的不是"融合"这件事，而是"你的对比对象没调好"。

### 展开解释：为什么 Baseline 可能太差？

看下面这个对比：

```
❌ 不公平对比：
Megakernel（手写的，精调过的）          60 ms
Unfused（vLLM 的通用实现，没调过）       90 ms
差距：90 - 60 = 30 ms → "Megakernel 快 33%！"

✅ 公平对比（如果 Unfused 也调好了）：
Unfused + PDL + kernel tune              65 ms
Megakernel（手写的）                      60 ms
差距：65 - 60 = 5 ms → "Megakernel 快 7.7%，不如多调调"
```

> [!insight] 关键洞察
> 公平对比下，Megakernel 可能只贡献了 5 ms。前面 25 ms 是 PDL 和 kernel tuning 的功劳，跟 Megakernel 没关系。

### 深入细节：PDL 怎么优化 Unfused？Kernel tuning 怎么调？

#### PDL（Programmatic Dependent Launch）

传统 Unfused 执行：

```
CPU:   launch kernel-A  →  wait A done（空等）  →  launch kernel-B  →  wait B done  → ...
GPU:                          run A                                 run B
```

PDL 优化后的执行：

```
CPU:   launch_all(A, B, C, D) — 一次性提交依赖图
GPU:   按图自动执行，不需要 CPU 再参与
```

这就是一个**调度技巧**，不是新技术，是靠消除 CPU-GPU 同步开销来降延迟。不改变 kernel 本身，只是改变 launch 方式。

#### Kernel Tuning

同一段 kernel 代码，不同的 block size 和 tile size 会跑出截然不同的性能。80% 的 SM 利用率 vs 50%，差别可能在 30%+ 的吞吐量。

不对 kernel 做 tune，就像开一辆不调教引擎的赛车去比赛——你输不是因为车不行，是因为你没调好。

---

## 多角度追问

### 追问 1：PDL 是什么？为什么要有它？

GPU 是异步计算设备——CPU 发完任务就可以去做别的事。但传统的 CUDA 编程里，CPU 要等 GPU 报告"上一个任务完成"才能发下一个。这个等待就是 host-side launch latency。

PDL 解决的就是这个问题：把多个 kernel 的依赖关系写成一张图（A 依赖 B → B 等 C 完成 → ...），一次性提交给 CUDA runtime。GPU 自己按图执行，不需要 CPU 在中间等着通知。

这是**优化 unfused kernel 的基础功夫**，不是新技术，是 CUDA 程序员应该会的基本操作。

### 追问 2：为什么国内外的 MoE 优化团队对 Megakernel 的看法不一样？

> **杨轶凡的解读**：字节和 DeepSeek 看的 problem size 不一样。DeepSeek 的规模更大，两两 fuse（像 UniEP 的 Flux style）就不够用了——combine 时间 > FC2 时间，所以 FC2+combine 仍然是 communication bound。要拉上 FC1 一起来 hide communication latency，所以才需要四个 kernel 合起来。字节两两 fuse 就够了。

所以你看到的"支持 Megakernel"的论文（MegaMoE）是在特定场景下合理的——不是因为 Megakernel 天生更好，而是因为**DeepSeek 的规模太大，不融合没办法**。

### 追问 3：如果没有 ablation study，我凭什么信你？

Ablation study 就是控制变量法——把每个改进点一个一个拿掉，看性能怎么变。

| 实验编号 | 配置                          | 延迟     |
| ---- | --------------------------- | ------ |
| 1    | 纯 Unfused（什么都没调）            | 100 ms |
| 2    | Unfused + PDL               | 80 ms  |
| 3    | Unfused + PDL + kernel tune | 65 ms  |
| 4    | Megakernel（全上）              | 60 ms  |

看 3 → 4：65 → 60 ms，只降了 5 ms。说明 Megakernel 本身的贡献就是 5 ms。前面 35 ms 的改进跟 Megakernel 无关，是 PDL 和 tuning 的功劳。

> [!contradiction] 核心争议
> **如果没有这个实验，你的 Megakernel 论文就不配说"比 Unfused 快"**——你说的是"比调好的 Unfused 快"，但你的 Unfused 根本没调好。

### 追问 4：Megakernel 真的没用吗？

不是没用。是**在特定场景下合理**：

- 你的 Unfused baseline 已经调好了（PDL + tuning 都用上了）
- 你发现即便调好了，combine 时间还是 > FC2 时间，说明 communication 是瓶颈
- 你的 FC2+combine 融合 + FC1 根本没办法不融合了

这时候 Megakernel 有意义——不是因为它**本来更快**，而是因为**你的规模大到不融合就解决不了 communication latency**。

> [!insight] 陆铭视角
> 杨轶凡认为："MegaMoE 是目前最 make sense 的 Megakernel 用例。虽然两两 fusion 看起来能 dispatch+FC1 和 combine+FC2 一起做，但 combine 时间 > FC2 时间，所以 FC2+combine 仍受 communication 限制。必须拉上 FC1 来 hide communication latency。"

这就是 **compute-communication overlap** 的极限优化。不是"merge 好玩"，而是"不 merge 就来不及"。

---

## 边界与极端情况

### 边界条件 1：什么模型规模不需要 Megakernel？

如果你的模型小、或 MoE 层只有 8 个 expert、或 batch size 很小，那么 dispatch / FC1 / FC2 / combine 都不是瓶颈。**调好 Unfused baseline 就行，不用碰 Megakernel。**

### 边界条件 2：什么场景绝对需要 Megakernel？

DeepSeek 的 MegaMoE——expert 数量巨大，batch size 大，通信才是瓶颈。即便如此，也只有 dispatch+FC1+combine+FC2 这一组操作需要融合——不是全模型都融合。

### 极端情况：如果我不调 Unfused，直接用 Megakernel？

> 你的 Megakernel 如果比 Unfused 快，建议看看是不是 baseline 太差了，PDL 是不是 tune 的够好了，选择的 kernel 是不是一致的。

也就是说，如果 Unfused 是 vLLM/SGLang 的通用实现，而 Megakernel 是你的手写 custom kernel——**这个比较数字不能说明任何问题**，因为你的 Kernel 实现水平不一样。

---

## 参考文献 & 延展学习

| 来源 | 类型 | 难度 | 链接 |
|------|------|------|------|
| 杨轶凡知乎回答 | blog | beginner | https://www.zhihu.com/question/2013258505231050695/answer/2051707069477151853 |
| 杨轶凡 PDL 中文版 | blog | intermediate | https://zhuanlan.zhihu.com/p/2052349834762380330 |
| 杨轶凡 Megakernel 深入讨论 | blog | intermediate | https://zhuanlan.zhihu.com/ (/ 作者 Yifan Yang) |
| CUDA Programming Guide | doc | intermediate | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| NVIDIA GPU Architecture | doc | intermediate | https://developer.nvidia.com/gpu-computing |
| bfloat16 & mixed precision | blog | beginner | https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/ |
| ICLR 2025 — Megakernel 系列 | paper | advanced | （关注 2025 年 ICLR 的 MoE inference 论文） |

> [!tip] 延展练习
> 动手试试：用 PDL 优化一个简单的 unfused kernel，对比没有 PDL 的延迟。看看你测出来的差值是多少——那个差值就是 PDL 的贡献，不是 Megakernel 的。

---

## 总结

| 概念             | 一句话                                                   |
| -------------- | ----------------------------------------------------- |
| PDL            | GPU 自己管理 kernel 依赖关系，消除 host-side launch latency      |
| Kernel Tuning  | 调 launch 参数（block size, tile size），不是调模型权重            |
| Ablation Study | 控制变量法，逐个去掉改进点以证明"是你在优化"                               |
| Megakernel     | 在特定规模下合理，但不建议在不调好 Unfused 的情况下比较                      |
| 核心教训           | 调好 Unfused + PDL + kernel tune 可能已经足够，不要梭哈 Megakernel |
