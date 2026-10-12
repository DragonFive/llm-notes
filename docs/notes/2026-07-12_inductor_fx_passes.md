---
title: Inductor FX Passes — Lowering 之前的图级优化全景
aliases:
  - 'inductor_fx_passes'
  - 'Inductor FX Passes — Lowering 之前的图级优化全景'
date: 2026-07-12
updated: 2026-07-12
tags:
  - type/lecture
  - domain/llm-inference
  - topic/compiler
  - topic/optimization
source_count: 3
status: draft
difficulty: intermediate
---

# Inductor FX Passes — Lowering 之前的图级优化全景

## 场景引入

FX Graph 从 Dynamo 出来时，是对 Python 代码的"忠实记录"——每一个 view、permute、冗余计算都原封不动地留在图里。Lowering 之后，这些高层语义信息会被碾碎成循环级 IR（per-index 的 lambda 表达式），再也看不出"这 4 个节点合起来是一个 Attention"。

所以 FX Passes 是 **Lowering 之前的最后一道关卡**——图里有大量冗余和低效模式，不在这一步清理掉就永远清理不了。

---

## 前置概念

### 模式匹配 vs 算子分解：两条互补路径

- **分解（Decomposition）**：大算子 → 碎片（增加节点数，为后续 Fusion 创造素材）
  - 例：`silu(x)` → `x / (1 + exp(-x))` — 3 个节点变 5 个节点
- **模式匹配（Pattern Matching）**：碎片 → 高效整体（减少节点数，替换为专用 kernel）
  - 例：`matmul + scale + softmax + matmul` → `flash_attention` — 4 个节点变 1 个节点
- **协同**：分解展开后，模式匹配可能识别出"展开后恰好是某个高效模式"
  - 先破（暴露子结构）→ 再立（选最优实现）

### 经典编译器优化对应

| 经典技术 | Inductor FX Pass 对应 |
|---------|---------------------|
| CSE（公共子表达式消除） | 节点重定向 + DCE |
| DCE（死代码消除） | eliminate_dead_code() |
| 常量折叠 | 分解表中常量直接算 |
| 常量传播 | 沿数据流追踪常量 → 折叠更多表达式 |
| 代数化简 | pointless_view / pointless_permute（noop 消除）|
| 循环融合 | Attention 融合 / Conv-BN 融合 / Group Batch 融合 |
| 指令选择 | PatternMatcherPass + Decomposition Table |

### 常量传播 vs 常量折叠

两者经常混淆，但有本质区别：

- **常量折叠** = "两个常量做运算 → 直接算出结果"（局部，一步）
- **常量传播** = "追踪变量值从哪来 → 发现是常量 → 然后折叠"（全局，多步）
- **传播是折叠的前提**

具体例子：

```python
a = 5          # 常量
b = a + 3      # 传播发现 a=5 → 折叠为 b=8
c = b * 2      # 传播发现 b=8 → 折叠为 c=16
d = c + x      # 传播发现 c=16 → 折叠为 d=16+x
               # x 不是常量，到这里停止
```

没有传播，编译器只能折叠 `5+3`（两个字面量）。有了传播，才能沿数据流一路追踪到 `c=16`，把多步运算全部折叠掉。

### noop 消除

**noop** = 执行了但什么都没改变的操作。

常见例子：
- `x.view(x.shape)` — view 操作但形状没变
- `x.permute(0, 1, 2, 3)` — permute 操作但维度顺序没变
- `x.contiguous()` — 本来就连续，调了也白调

对应的 FX Pass：
- **pointless_view** = view 操作但形状没变 → 直接删除
- **pointless_permute** = permute 操作但维度顺序没变 → 直接删除

> 为什么会出现 noop？Dynamo 忠实记录用户代码中的所有操作，包括实际上是 noop 的那些。用户代码里写了 `x.contiguous()`，Dynamo 就录进图里——它不管这个操作有没有实际效果。

### 节点重定向 + DCE

这是 FX Pass 中"替换 + 清理"的标准两步：

**Step 1 — 节点重定向**：

```
原始图：
  node_1: matmul(Q, K^T)
  node_2: scale(node_1, 1/√d)
  node_3: softmax(node_2)
  node_4: matmul(node_3, V)
  node_5: use(node_4)        ← 下游用 node_4 的结果

模式匹配找到 4 个 SDPA 节点 → 创建新节点：
  node_new: flash_attention(Q, K, V)

重定向：所有"用 node_4 结果的" → 改为用 node_new
  node_5: use(node_new)      ← 现在指向新节点
```

**Step 2 — DCE（Dead Code Elimination）**：

```
node_1 用户数=0（没人用了）→ 删除
node_2 用户数=0 → 删除
node_3 用户数=0 → 删除
node_4 用户数=0 → 删除
```

两步配合 = **"替换 + 清理"**。模式匹配只负责创建新节点并重定向引用，DCE 负责清理失去所有用户的旧节点。

---

## 核心讲解

### 第一部分：三阶段 Pass 体系

```
Pre-grad Pass（前处理）—— AOTAutograd 之前，完整计算图
  ├── fuse_fx：sink_cat_after_pointwise + permute_fusion
  ├── normalization_pass：layernorm → native_layer_norm 标准化
  ├── group_batch_fusion：多个小算子合并为批量执行
  └── efficient_conv_bn_eval_pass：conv + BN → 单个 conv

Joint Graph Pass（联合图优化）—— 前向+反向联合图上
  ├── canonicalize_aten_ir：标准化 ATen 表示
  ├── remove_noop_ops：删除空操作
  ├── early_patterns：pointless_view / pointless_permute
  ├── SDPA 模式匹配：28+ 个 attention 变体 → flash_attention
  └── pad_mm + misc_patterns

Post-grad Pass（后处理）—— 前向/反向分离后，精确形状已知
  ├── eliminate_dead_code()
  ├── group_batch_fusion（第二次）
  ├── pass_patterns[0..2]：三轮模式匹配
  └── reinplace_inplaceable_ops：out-of-place → in-place
```

**为什么分三阶段？**

| 阶段 | 时机 | 能看到什么 | 做什么 |
|------|------|-----------|--------|
| Pre-grad | AOTAutograd 之前 | 完整计算图（未分离前向/反向） | 需要全图信息的融合（Conv-BN、Group Batch） |
| Joint | 联合图上 | 前向+反向在同一张图 | 标准化表示、消除 noop、大模式匹配（SDPA） |
| Post-grad | 前向/反向分离后 | 精确的张量形状（FakeTensor 已跑完） | 需要精确形状的模式匹配、in-place 优化 |

关键因果链：Pre-grad 需要完整图（Conv-BN 需要看到 BN 的 running_mean/var）→ Joint 需要标准化表示（SDPA 模式匹配需要统一方言）→ Post-grad 需要精确形状（模式匹配二次验证需要 FakeTensor 形状）。

#### 各 Pass 详细拆解

**Pre-grad 阶段**：

| Pass | 做什么 | 类比 |
|------|--------|------|
| **sink_cat_after_pointwise** | 把 cat 操作往计算链下游推，让 cat 之前的 pointwise 有机会互相融合 | 先把零件各自打磨好，最后再拼接 |
| **permute_fusion** | 合并连续的 permute 为一次（两次维度重排可数学组合为一次） | 两次转弯合并为一次 |
| **normalization_pass** | 统一 LayerNorm 的多种表示为 `native_layer_norm` | 6 种方言统一为普通话 |
| **group_batch_fusion** | 多个独立小 GEMM 合并为 batched_linear | 3 封单独快递合并为 1 个包裹 |
| **efficient_conv_bn_eval_pass** | 推理时把 BN 参数数学折进 Conv 权重 | 两步工序预合并为一步 |

```python
# sink_cat_after_pointwise 演示：
# 优化前：relu(x), relu(y) → cat → sigmoid（cat 挡在中间）
# 优化后：relu(x)→sigmoid, relu(y)→sigmoid → cat（cat 推到最后）
# 收益：relu+sigmoid 可以被 Fusion 合并为一个 kernel

# permute_fusion 演示：
# x.permute(0,2,1).permute(1,0,2) → x.permute(2,0,1)（一步到位）

# normalization_pass 演示：
# aten.layer_norm → aten.native_layer_norm（输出变为 tuple: (output, mean, rstd)）
# 为什么？native_layer_norm 是底层确定形式，后续模式匹配只需认一种

# group_batch_fusion 演示：
# q_proj(x), k_proj(x), v_proj(x) → batched_mm(x, [W_q, W_k, W_v])
# 能 batch 的条件：共享输入 + 独立权重 + 无依赖 + 形状兼容
```

**Joint 阶段**：

| Pass | 做什么 | 为什么在这个位置 |
|------|--------|---------------|
| **canonicalize_aten_ir** | 统一 ATen 算子多种写法（matmul→mm/bmm、add.Tensor→add.default） | 模式匹配前必须统一"方言"，否则同一模式要写多套规则 |
| **remove_noop_ops** | 删除 noop（view 同形 / contiguous 本来就连续 / to 同类型） | 减少图噪声 |
| **early_patterns** | 删除 pointless_view / pointless_permute | 在 SDPA 匹配前清理——否则 view 节点会打断模式匹配 |
| **SDPA 模式匹配** | 识别 28+ 种 Attention 变体 → flash_attention | 最重要的优化，必须在干净图上做 |
| **pad_mm** | 不对齐矩阵 pad 到 8/16/32 倍数（Tensor Core 要求） | GEMM 效率优化 |
| **misc_patterns** | 杂项（mm+add→addmm、cat+split 互消等） | 收尾清理 |

```python
# canonicalize_aten_ir 为什么需要：
# 用户写 x @ y → Dynamo 追踪为 aten.matmul
# 但模式匹配注册的是 aten.mm
# 不标准化 → 匹配失败 → 优化漏掉
# 标准化：aten.matmul(2D,2D) → aten.mm

# early_patterns 为什么叫"early"：
# 在 SDPA 匹配之前先做。
# 如果图里有 pointless_view：matmul → [view] → softmax
# SDPA 模式要求 matmul → softmax 直连 → 匹配失败
# early_patterns 先删 view → matmul → softmax → SDPA 匹配成功！

# pad_mm 演示：
# [32, 100] × [100, 200] → pad 到 [32, 128] × [128, 200]
# Tensor Core 需要 8/16 对齐 → pad 后效率提升 > pad 开销
# 启发式判断：矩阵够大才 pad（太小不值得）
```

**Post-grad 阶段**：

| Pass | 做什么 | 为什么在最后 |
|------|--------|-----------|
| **eliminate_dead_code** | 删除用户数=0 的节点 | 前面的替换可能遗留死节点 |
| **group_batch_fusion（第二次）** | 再次扫描 batch 机会 | AOTAutograd 分离后暴露了新的独立操作（如反向图中多个独立 weight grad） |
| **pass_patterns[0..2]** | 三组不同模式各跑一遍 | 依赖顺序（先清理→再优化→最后收尾） |
| **reinplace_inplaceable_ops** | out-of-place → in-place（省内存） | 必须最后做——需要确认"没人再用旧值" |

```python
# 为什么 group_batch_fusion 要跑两次：
# Pre-grad：在完整图上找 batch 机会（如 QKV 三个 Linear）
# Post-grad：AOTAutograd 分离后反向图里可能新出现独立的 weight gradient 计算
#           → 这些是 pre-grad 时不存在的 → 需要再扫一遍

# reinplace_inplaceable_ops 演示：
# 原始：buf0 = add(x, bias)    → buf1 = relu(buf0)
#       buf0 之后没人再用了 → 可以安全地 in-place
# 转换：buf0 = add(x, bias)    → relu_(buf0)  ← 直接改 buf0，不分配 buf1

# 哪些操作可以 in-place：
#   add_, mul_, relu_, sigmoid_, zero_, fill_, copy_, scatter_
# 条件：被修改的 tensor 之后不再被其他节点使用（use-def 链分析）

# 为什么不一开始就用 in-place：
# 1. 正确性：用户写 y=x+1 语义是"x 不变"，不能偷改 x
# 2. AOTAutograd 需要前向中间值做反向 → 不能 in-place 改掉
# 3. 必须看完整 use-def 链才能判断安全性 → 所有其他 Pass 做完后再分析
```

### 第二部分：模式匹配引擎架构（PatternMatcherPass）

#### 执行流程

```
1. 收集候选锚点（find_anchor_nodes）
   "锚点" = 模式中最后一个节点的 (op, target) 类型
   例：SDPA 模式的锚点 = 最后那个 aten.matmul

2. 按逆拓扑序排列（reverse=True）
   先处理靠近输出端的节点

3. 逐个尝试匹配
   _TargetArgsExpr._match()：
     ├── 检查 target（算子类型是否匹配）
     └── 递归检查 args（从锚点沿 args 链向上追溯）

4. extra_check 通过后应用替换
   extra_check：用户自定义的额外验证（如形状约束、dtype 约束）
   通过 → 创建新节点 + 重定向 + DCE

5. 返回替换计数
```

#### 为什么从后往前（逆向）匹配？— fan-out vs fan-in

这是模式匹配算法最核心的设计决策：

```
FX Graph 的数据流结构：
  正向（数据流方向）：一个节点可能有 N 个消费者（fan-out，不确定）
  逆向（args 方向）：一个节点的输入来源是确定的（args 是固定列表）

         input_x
        /       \
   matmul_1   matmul_2      ← fan-out：input_x 有 2 个消费者
     |           |
   scale       linear       ← 从 input_x 正向找，每个分叉都要探索
     |           |
   softmax     relu
     |           |
   matmul_3    add
```

**正向找（从 input 到 output）**：

```
从 input_x 出发 → 有 2 个消费者 → 走哪条？
  走 matmul_1 → 继续看它的消费者 → 可能还有分叉
  走 matmul_2 → 走错了，不是 SDPA

fan-out = 分叉 = 每个分叉都要探索 = 指数级搜索空间
```

**逆向找（从 output 到 input）**：

```
从锚点 matmul_3 出发 → 检查 matmul_3.args：
  args[0] = softmax_node ← 确定的！（每个节点的 args 是固定列表）
  → 检查 softmax_node.args：
    args[0] = scale_node ← 确定的！
    → 检查 scale_node.args：
      args[0] = matmul_1 ← 确定的！
      → 匹配成功！

fan-in = 每步只有一条确定路径 = O(模式深度) 线性验证
```

> [!insight] 为什么逆向更快？
> `node.args`（输入来源）是确定的固定列表——Python 函数的参数就那几个。
> `node.users`（谁用了这个节点）可能有任意多个——一个 tensor 可能被 10 个操作使用。
> **逆向沿 args 追溯 = 每步路径唯一，不存在"走错路"。正向沿 users 探索 = 每步可能分叉。**

**逆拓扑序遍历的额外好处**：

```
好处 1：避免替换冲突
  如果图有 A → B → C → D → E：
    模式 1 要匹配 [C, D, E]
    模式 2 要匹配 [A, B, C]
    
  逆序（从 E 开始）：先匹配模式 1 → C 被替换
  → 模式 2 再看 C → 已经不存在了 → 正确跳过
  
  正序（从 A 开始）：先匹配模式 2 → C 被替换
  → 模式 1 再看 C → 已经不存在了 → 漏掉了更好的优化！

好处 2：与递归方向一致
  .match() 从锚点往 args 递归（逆向追溯）
  遍历也是逆向 → 自然对齐

好处 3：跟前向/反向传播无关
  这里的"逆"指的是 DAG 拓扑顺序的反向，不是"反向传播"
  前向图和反向图都用同一个逆拓扑序算法做模式匹配
```

#### 两种注册方式

**register_replacement（声明式）**：

```python
def search_fn(x, y):
    # 描述要匹配的模式
    return torch.matmul(x, y.transpose(-1, -2)) * scale

def replace_fn(x, y):
    # 描述替换后的实现
    return flash_attention(x, y)

register_replacement(search_fn, replace_fn, extra_check=check_shapes)
```

引擎自动 trace search_fn → 构建模式 DAG → 匹配 → 替换。适合**简单的一对一替换**。

**register_graph_pattern（命令式）**：

```python
@register_graph_pattern(pattern, extra_check=...)
def handler(match, x, y, z):
    # 完全自定义的图变换逻辑
    # 可以看上下文、做条件判断、创建多个新节点
    ...
```

提供 handler 函数，完全自定义图变换。适合**需要看上下文的复杂变换**（如根据输入形状选择不同替换策略）。

#### 模式 DAG 构建

```
Python 函数 → FX trace → FX Graph → 转为 PatternExpr DAG

PatternExpr 节点类型：
  ├── CallFunction  — 算子调用（如 torch.matmul）
  ├── KeywordArg    — 需要捕获的参数（如 x, y）
  └── Ignored       — 不参与匹配的部分
```

以 SDPA 模式为例：

```python
def sdpa_pattern(query, key, value, inv_scale):
    # FX trace 这个函数 → 得到 4 个 CallFunction 节点
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * inv_scale
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.matmul(attn_weight, value)
```

trace 后变为：

```
PatternExpr DAG:
  transpose_node = CallFunction(aten.transpose, KeywordArg("key"), -2, -1)
  matmul_1 = CallFunction(aten.matmul, KeywordArg("query"), transpose_node)
  scale_node = CallFunction(aten.mul, matmul_1, KeywordArg("inv_scale"))
  softmax_node = CallFunction(aten.softmax, scale_node, dim=-1)
  matmul_2 = CallFunction(aten.matmul, softmax_node, KeywordArg("value"))
```

#### 两次匹配机制

- **第一次（通用模式匹配）**：只匹配算子类型和结构
  - 快速过滤：DAG 结构匹配（target 类型 + args 数量 + 拓扑关系）
  - 不做形状检查（形状检查贵）
  - 过滤掉 99% 的非匹配节点

- **第二次（具体模式验证）**：用 FakeTensor 重新 trace → 形状精确验证
  - 用第一次匹配捕获的实际参数，跑一遍 search_fn 和 replace_fn
  - 验证输入形状是否真的满足替换条件
  - 验证替换后输出形状是否与原来一致

**为什么分两次？** 形状检查需要跑 FakeTensor（本质是做一遍 meta 计算），开销不小。先用结构匹配（纯图遍历，O(n) 级别）过滤掉绝大多数不可能匹配的节点，再对少数候选做精确验证。

#### 预编译模式（serialized_patterns/）

**问题**：28 个 SDPA 模式 + 数十个其他模式，每个都要 FX trace 构建 DAG。启动时全部 trace 一遍太慢。

**方案**：

```
开发时：
  PYTORCH_GEN_PATTERNS=1 python ... → trace 所有模式 → 序列化为 Python 文件
  → 存入 serialized_patterns/ 目录

生产环境：
  启动时直接 import serialized_patterns → 跳过 trace
  → 启动快几秒
```

**类比**：提前画好地图存文件 vs 每次启动现场画地图。地图（模式 DAG）不变，没必要每次重画。

### 第三部分：关键融合案例深入

#### Attention 融合（最重要）

**未融合**：4 个独立 kernel，3 次 HBM 读写

```
Kernel 1: matmul(Q, K^T) → 中间结果写 HBM（O(N²) 大小）
Kernel 2: scale(result)  → 读 HBM + 写 HBM
Kernel 3: softmax(result)→ 读 HBM + 写 HBM
Kernel 4: matmul(result, V) → 读 HBM + 写 HBM
```

**融合后**：1 个 flash_attention，0 次中间 HBM 读写

```
flash_attention(Q, K, V) → 直接输出最终结果
  - 中间 N² 矩阵永远不写 HBM（tiling 分块计算）
  - 带宽从 O(N²) 降到 O(N)
```

**28+ 种 SDPA 变体**：带 mask / dropout / causal / bias / 不同 scale / GQA / MQA / sliding window ...

每种变体都是一个独立的模式注册，匹配后替换为对应的高效实现。

> **为什么必须在 FX Pass 做？** Lowering 后 4 个 ATen 节点被碾碎为几十个循环级 ops（per-index lambda），再也看不出"这是 Attention"。语义信息只在 FX Graph 层面存在。

#### Conv-BN 融合

这不是"两个 kernel 合为一个"——是**数学上把 BN 参数折进 Conv 权重**：

```
原始：y = BN(Conv(x))
     = gamma * (Conv(x) - mean) / std + beta
     = gamma * (W*x + b - mean) / std + beta

折叠后：y = W_new * x + b_new
  W_new = W * gamma / std
  b_new = (b - mean) / std * gamma + beta
```

结果：推理时一个 Conv 搞定，完全省掉 BN 的计算和内存访问。

> **只在推理时可做**。训练时 BN 的 mean/std 每个 batch 都变（running statistics），无法提前折进权重。

#### Group Batch 融合

多个独立小算子合并为一次批量执行：

```
原始：
  out_1 = linear(x_1, W_1)   # kernel launch 1
  out_2 = linear(x_2, W_2)   # kernel launch 2
  out_3 = linear(x_3, W_3)   # kernel launch 3
  ...                         # N 次 launch

融合后：
  [out_1, out_2, ..., out_N] = batched_linear([x_1,...,x_N], [W_1,...,W_N])
                              # 1 次 launch
```

**收益**：
- 1 次 launch 替代 N 次 launch（节省 launch overhead）
- 大 batch 提高 GPU 利用率（单个小 GEMM 可能填不满 SM）

### 第四部分：为什么 Post-grad 有三轮模式匹配？

不是"同一组模式跑三遍"——是"三组不同模式各跑一遍"：

| 轮次 | pass_patterns[i] | 内容 | 为什么在这一轮 |
|------|-----------------|------|-------------|
| 0 | pass_patterns[0] | attention 相关模式 | 最复杂、最高优先级，先清理 |
| 1 | pass_patterns[1] | matmul 相关模式 | 依赖 attention 清理后的残余 |
| 2 | pass_patterns[2] | 杂项模式 | 其他零散优化 |

**分组原因**：
1. **依赖顺序**：第 0 轮的替换可能暴露第 1 轮能匹配的新模式
2. **性能**：避免无关模式互相干扰（attention 模式不需要和 matmul 模式竞争锚点）
3. **可维护性**：每组模式独立维护、独立测试

### 第五部分：SiLU 分解选择——为什么选方式 B

SiLU(x) = x * sigmoid(x)，分解有两种方式：

```
方式 A：x * sigmoid(x)
  → mul(x, div(1, add(1, exp(neg(x)))))
  节点：neg → exp → add(1,·) → div(1,·) → mul(x,·)

方式 B：x / (1 + exp(-x))
  → div(x, add(1, exp(neg(x))))
  节点：neg → exp → add(1,·) → div(x,·)
```

**选 B 的原因**：与 PyTorch eager 的 C++ 实现**数值行为 bit-exact 一致**。

```
方式 A 的 mul 最后一步：x * (1/(1+exp(-x)))
方式 B 的 div 最后一步：x / (1+exp(-x))

数学上等价，但浮点运算中 mul 和 div 的舍入行为不同。
PyTorch eager 模式的 C++ kernel 用的是除法形式（方式 B）。
```

**Inductor 原则**：编译结果必须和 eager 模式精确匹配（bit-exact）。用户不应该因为加了 `torch.compile` 就得到不同的数值结果。这是正确性的底线。

---

## 多角度追问

> [!question] XLA/Grappler 跟 FX Passes 是什么关系？
> **Grappler ≈ FX Passes**（图级优化器）：都是在高层图表示上做模式匹配、常量折叠、算子融合。
> **XLA ≈ Inductor**（编译器后端）：都是把优化后的图编译为可执行代码（XLA → HLO → LLVM IR → 机器码；Inductor → Loop IR → Triton/C++ → PTX/binary）。
> XLA 不是 TensorFlow 的一个 pass，而是一个完整的独立编译器。

> [!question] buffer 级别的 CSE 机会从哪来？
> **Decomposition 后出现重复子表达式**。例如 `layer_norm(x)` 分解后：
> ```
> mean = x.mean(-1)
> var = ((x - mean)^2).mean(-1)     # 这里算了 x - mean
> out = (x - mean) / sqrt(var + eps) # 这里又算了 x - mean
> ```
> `x - mean` 被计算了两次。如果不做 CSE（或 Inlining + Fusion），这个重复子表达式会导致两次 HBM 读写。Inductor 的方案是通过闭包组合 + Inlining 让两个消费者共享同一次计算。

> [!question] 为什么 Attention 融合一定要在 FX Pass 做？
> Lowering 后，`matmul(Q, K^T)` 变成一个 per-index 的 lambda（循环级 IR），`softmax` 变成另一组 lambda。此时你看到的是几十个零散的 `load`/`store`/`exp`/`sum` 操作，再也看不出"这 4 个节点合起来是 Scaled Dot-Product Attention"。语义信息在 Lowering 时被永久碾碎了——所以**只有在 FX Graph 层面才能识别并替换**。

> [!question] 模式匹配有误匹配风险吗？
> **有**，所以需要多重防护：
> 1. **extra_check**：注册模式时可附加自定义检查（如"Q 和 K 的最后一维必须相等"）
> 2. **FakeTensor 二次验证**：用真实形状重新 trace，确认替换后语义等价
> 3. **单元测试覆盖**：每个注册的模式都有对应的 test case，验证替换前后数值一致
> 4. **逆拓扑序保护**：同一个节点不会被多个模式重复匹配（先匹配的先替换）

---

## 与其他概念的关系

- 2026-07-09_inductor_phase_ordering — FX Passes 是六层管线的第三层（FX Passes 阶段），位于 Decomposition 之后、Lowering 之前
- 2026-07-09_inductor_ir_design — FX Passes 操作的是 FX Graph（图结构），Lowering 后变为 Python 函数 IR（循环级闭包）
- 2026-07-10_dynamo_tracing_mechanism — Dynamo 产出的 FX Graph 是 FX Passes 的输入；Dynamo 的忠实记录导致了 noop 等冗余
- 2026-06-25_kernel_fission_fusion — FX Pass 做的是"语义级融合"（模式匹配识别高层模式），Scheduling 做的是"循环级融合"（依赖分析 + 内存复用）

---

## 关键洞察

> [!insight] FX Pass 能做的优化 = Lowering 之后就做不了的优化
> 因为语义信息在 Lowering 后被碾碎。4 个 ATen 节点（matmul/scale/softmax/matmul）Lowering 后变成几十个 per-index 的 lambda 操作，再也无法识别"这是 SDPA"。所以所有需要高层语义的优化（Attention 融合、Conv-BN 权重折叠）必须在 FX Pass 阶段完成。

> [!insight] 模式匹配 + 算子分解 = 指令选择的 ML 特化版
> 经典编译器的指令选择：先把高层操作展开为中间表示（暴露子结构），再从中选择最优机器指令组合。Inductor 的做法完全同构：Decomposition 先把复合算子展开为 Core ATen 基础操作（暴露子结构），PatternMatcherPass 再从中匹配出最优的高效实现（flash_attention、fused_conv_bn）。

> [!insight] 三阶段架构不是随意的——每个阶段需要不同的信息
> Pre-grad 需要完整图（Conv-BN 融合需看到 BN 的 running statistics，这些在反向分离后会丢失）；Joint 需要标准化表示（SDPA 模式匹配需要统一的 ATen 方言）；Post-grad 需要精确形状（模式匹配的二次验证需要 FakeTensor 提供的精确 shape/dtype）。信息可用性决定了优化的阶段归属。

---

## 参考文献

- PyTorch 2 论文 Section 4.1-4.2（FX Graph Optimizations）
- EaC3 Ch.8 "Introduction to Optimization"、Ch.9 "Loop Optimizations"、Ch.10 "Instruction Selection"
- Inductor Overview §4.2 (FX 图优化)
- Phase 2: FX Optimization 教程 (skeleton_tutorial/phase2_fx_optimization.md)
