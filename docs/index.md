# LLM Notes

机器学习系统学习笔记的公开策展镜像：GPU 调度、Tensor Core、编译器与指令级实现。

- **源**：笔记在私有 Obsidian vault 中以 Markdown 维护，本仓是其**只读策展导出**（What you see is a curated export, not the working vault）。
- **构建**：MkDocs-Material，GitHub Actions 推送后自动构建发布。
- **原则**：Markdown 是 source of truth，HTML 只是派生的阅读层。

## 笔记索引（15 篇，类内最新在前）

### 推理系统与分布式并行

- [深入解析 linear tail prefix cache](notes/2026-10-13_xllm_linear_tail_prefix_cache_deep_dive.md) `2026-10-13`
- [SfaDcp:decode KV 分条注意力深潜](notes/2026-10-10_xllm_sfadcp_dcp_decode_attention_deep_dive.md) `2026-10-10`

### GPU Kernel 与指令级

- [GEMM+AR 融合与 multimem 通信深潜](notes/2026-07-07_gemm_ar_multimem_deep_dive.md) `2026-07-07`
- [MMA 指令深潜：线程分配、Fragment Layout 与数据流](notes/2026-07-06_mma-instruction-deep-dive.md) `2026-07-06`
- [Tensor Core PTX 指令管线全拆解](notes/2026-07-04_tensor_core_ptx_pipeline.md) `2026-07-04`
- [WGMMA Hopper GEMM 完整讲解](notes/2026-07-02_wgmma_hopper_gemm.md) `2026-07-02`
- [指令调度代际跃迁：从同步锁资源到异步发令枪](notes/2026-06-29_async_instruction_scheduling.md) `2026-06-29`
- [Warp Scheduler 硬件架构全解析](notes/2026-06-25_warp_scheduler_deep_dive.md) `2026-06-25`
- [GPU 调度深潜：TBS、Warp Scheduler 与延迟隐藏的完整拆解](notes/2026-06-24_gpu_scheduling_deep_dive.md) `2026-06-24`
- [PyTorch 编译栈与 Lowering 层次映射](notes/2026-06-23_pytorch_compiler_stack.md) `2026-06-23`
- [Fusion / Tiling / Layout 三元组](notes/2026-06-23_fusion_tiling_layout.md) `2026-06-23`
- [PDL vs CUDA Graph vs Piecewise CUDA Graph — LLM 推理 kernel launch overhead 优化全景](notes/2026-06-22_pdl_vs_cudagraph.md) `2026-06-22`
- [Megakernel vs Unfused — 为什么 GPU 推理优化不能迷信"大融合"](notes/2026-06-22_megakernel_vs_unfused.md) `2026-06-22`

### 编译器与图优化

- [Inductor FX Passes — Lowering 之前的图级优化全景](notes/2026-07-12_inductor_fx_passes.md) `2026-07-12`
- [Inductor IR 设计精髓 — Define-by-Run、闭包组合、V.ops 虚拟化](notes/2026-07-09_inductor_ir_design.md) `2026-07-09`

- **方法**：内容出自我主导的深度评审与教学流水线——勘探定界、分章深潜、冷读验收、勘误裁决，每个数字都经源码核验。

> 持续更新中。本站只收录纯通用技术篇；涉及内部工程细节的笔记不在范围。

<!-- 本文件由 tools/export.py 生成，请勿手改 -->