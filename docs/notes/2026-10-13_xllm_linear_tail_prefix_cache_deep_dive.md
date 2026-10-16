---
title: 深入解析 linear tail prefix cache
date: 2026-10-13
tags:
  - type/synthesis
  - domain/llm-inference
  - domain/systems
  - topic/npu
  - topic/prefix-cache
  - topic/linear-attention
  - topic/mtp
---

深入解析 linear tail prefix cache

## 总览：这是什么，为什么值得读

**一句话身份**：这份笔记深潜解析本 MR（xLLM 仓库，head `commit-4`，4 commits，34 文件 +1436/−277）——它给 GLM-5.3 系 GDN linear attention 的线性状态补上了 prefix cache 能力。

**解决的问题**：linear 状态与 KV cache 是两种东西——KV cache 天然按块可复用，而 linear 状态原本「只能整段恢复」：prompt 长度只要不落在检查点边界上，前面算好的状态全部作废重算。本 MR 在 HBM → host 内存 → Mooncake Store **每一个物理层级钉死同一套「可恢复边界 + 边界键」**，把 8191 这类不对齐 prompt 的前缀复用深度从 0 提到 8064（98.5%）。

**TL;DR（五章 = 五个台阶）**：
1. 一切机制的地基是一个 15 行的纯函数（第一章）：双排除规则 + tail 点查不截尾；
2. 复用的单位由「两把尺子」决定（第二章）：token 数 → 链上下标，KV 逻辑块 = hash 粒度 = 128，检查点步距 = 8192；
3. 运行时靠「账本四动作」维护状态（第三章）：mount / take / erase / replace + tail 便签；
4. 前缀复用真正发生在准入 probe（第四章）：gap-tolerant 查找 + 按块回退 + 发布守卫；
5. 边界键跨三层存储下沉（第五章）：配对下沉、盖键、EXPLICIT 发布——总览里「每一层级同一套边界」的完整实现。

> **姊妹篇**：vllm-sglang 深潜笔记 教参照系两主角（vLLM/SGLang 的缓存组织）；本卷教 xLLM 侧。三方对照表两卷各有一份、互为镜像。

**阅读路线**：全书自底向上——数学（第一章）→ 键空间（第二章）→ 运行时账本（第三章）→ 准入与发布（第四章）→ 三层存储（第五章），每章开头有「承上启下」一段串起弧线。只想快速理解 MR 干了什么：读本总览 + 第一章 §1.1-1.2 + 索引层 §索一。带着具体困惑来：§索七「追问精华」按你的原话困惑索引到对应章节。

## 目录

- **总览**：这是什么，为什么值得读
- **深潜层**
  - 第一章 · tail 检查点数学：纯函数、双排除与恢复边界
  - 第二章 · 键空间：两把尺子、域桶与 MTP lookahead 的物理本质
  - 第三章 · 账本四动作与双账统一：mount/take/erase/replace 和 tail 便签
  - 第四章 · 准入 probe 与发布流水线：gap-tolerant 查找、按块回退与发布守卫
  - 第五章 · 三层存储与 D2H 事务：配对下沉、盖键与 EXPLICIT 发布
- **索引层**（结论级速查）：§索一 MR 总览与作者意图 / §索二 设计评价汇总 / §索三 三方对照 / §索四 Review findings / §索五 学术脉络 / §索六 教学演化与勘误链 / §索七 追问精华 / §索八 手算示例总表 / §索九 来源与处置
- **附录**：结算与版本记录（从卷首迁来）

---

# 第一章 · tail 检查点数学：纯函数、双排除与恢复边界

> 🔗 **承上启下**：总览说 本 MR 钉死「可恢复边界」——边界由什么决定？本章拆出那个 15 行的纯函数：双排除规则各防一件什么事、探测口径差（match 截尾、tail 点查不截）、8064 为什么站在 0 与 8192 之间。**全书的数学地基在这一章**。

> **核验基准**：本 MR，head `commit-4`（base `base-commit`）。本章所有代码均用 `git show commit-4:<path>` 重新摘录，行号即该提交行号，逐块回读核验。
> **数值约定**：KV 逻辑块 = hash 粒度 = **128** token；LINEAR 常规检查点步距 = **8192** token。本章手算全部用这两个数。
> **本章正面回答的两个原始困惑**（用户原话，2026-10-12 / 10-13）：
> ① "画面二：排除规则一被破坏……如果 prompt 只有 8192 个，恢复了怎么就错了？那该怎么办？不管有没有 tail 都会有问题吧"
> ② "prompt=8192 时 checkpoint_tokens == 0 为什么""查常规检查点退了 token，查 tail 没退，8064 命中 tail 无新 token forward 不还是有问题吗"

---

### 1.0 前置概念：五块铺路石 + 主角先露脸 + 一张同名消歧表

**概念 1 · LINEAR 状态（整段折叠）。**
GLM-5.3 系的线性注意力（GDN）不按 token 存 KV，而是把读过的整段前缀折叠进一个固定大小的递归状态：

```
S_n = f(t_0, t_1, …, t_{n-1})
```

它是一坛"腌菜"：所有 token 的贡献腌进同一个张量，取出来只能整坛取。
后果一：状态只在 token 边界上有意义，没有"半个状态"。
后果二：恢复只能整段回到某个边界，无法"退半个恢复边界"。

**概念 2 · 常规检查点（checkpoint，stride=8192）。**
若每次都从 token 0 重新折叠，长 prompt 太亏。框架每隔 8192 个 token 给 LINEAR 状态拍一张快照。
第 k 个常规检查点站在 `k×8192` 的位置：k=1 在 8192，k=2 在 16384，依此类推。
它们是**全局网格**上的固定站位，编成序列，由游标（num_cached_blocks）计数。

**主角先露脸 · tail（尾检查点）。**
本章标题里的主角，必须在第一次被用到之前有名字和画面：
tail 检查点 = **站在两个常规检查点之间的一个物理状态源**，位置由 prompt 自己算出——
`⌊prompt/128⌋×128`，即 prompt 按 128（KV 块大小，数值约定已给出、概念 3 马上讲）向下取整，取到最后一个完整块边界（8191 → 8064）。
它不是全局网格的固定站位（那是概念 2 的地盘），而是给"站不进 8192 网格的 prompt"留的一扇侧门：有了它，LINEAR 状态在网格的缝里也有处可复用。
哪些边界有资格当 tail（两条排除规则）是 §1.1-§1.2 的主线；本节先把名字、位置和"站在缝里"的画面钉死，§1.1.3 路径 2 处再给完整定义。

**概念 3 · KV 块（block size=128）。**
KV cache 按 128 token 一块分配。tail 数学里所有"对齐"说的都是这把尺子。
（KV-split 场景下逻辑块 = 物理块 × kv_split；本笔记各章一律用 128 的逻辑块口径直算。）

**概念 4 · 前缀缓存与恢复边界。**
框架维护一张"键→块"的缓存。老请求 A 算过的状态发布进去，新请求 B 前缀相同即可命中复用。
**恢复边界** = B 开工前状态回到的 token 深度；B 只需 forward 边界之后的 token。

**概念 5 · forward 至少吃 1 个 token。**
模型要产出下一个 token 的概率，至少要前向计算 1 个新 token。
因此任何命中深度都必须**严格小于** prompt 长度——这是贯穿全章的不变量，记作：

```
不变量 U：恢复深度 ≤ prompt − 1（给 forward 留活干）
```

**同名消歧表**（复合块管理器下挂两片**叶子**：LINEAR 叶管线性状态、KV 叶管 KV 块。`block_size` 在两个叶子里含义不同，必须先钉死再往下走）：

| 名字 | LINEAR 叶的值 | 量纲 | 谁定的 | 控制什么 |
|---|---|---|---|---|
| `block_size()` | 8192 | token 数 | 构造期 chunk_stride | 常规检查点步距（尺子一） |
| `hash_block_size()` | 128 | token 数 | 构造期 token_block_size | tail 对齐粒度 + 键链粒度（尺子二） |
| KV 叶 `block_size()` | 128 | token 数 | KV 配置 | KV 分配粒度 |
| `prompt_tokens` | 如 8191 | token 数 | 请求自带 | 纯函数第一入参 |

两把尺子同框的源码证据（LINEAR 叶构造期把 chunk_stride 塞进 block_size、把 token_block_size 塞进 hash_block_size）：

```cpp
// linear_state_block_manager.cpp:41-45
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.hash_block_size(static_cast<uint32_t>(token_block_size));
  options.hasher_type(hasher_type);
```

读法：`block_size` 这个名字在 LINEAR 叶里装的**不是 KV 块**，而是检查点步距。
这是命名陷阱，后文所有公式都以此表为准。

---

### 1.1 是什么：一个 15 行的纯函数

#### 1.1.1 旅程图：tail 这个数字的一生

先看全景，再进逐行。tail 边界这个数字在五个环节被算、被用。

图中三个名字，进图前各给一句人话（细节后文各归其章）：

**prefill chunk（预填充分片）**——调度器把长 prompt 切成逐段前向的小段，一段算一次。

**trim（收口裁剪）**——把准入阶段两条路径各自申报的深度，裁成一份一致且有据可查的挂载深度。

**便签**——`linear_state_tail_tokens_` 的别名：账本上记 tail 深度的一格字段，发布 tail 时写入（linear:443），账本全景详见第三章。

```
【请求 B，自带 prompt】
① 调度封顶（scheduler_policy.cpp:514-519）
│   算一次 tail：用它把 prefill chunk 切在可发布边界上
▼
② 准入探测（composite_block_manager.cpp:593-615）
│   问一 常规链：allocate_shared 截 1 token → match 走检查点网格
│   问二 tail 点查：allocate_tail_shared 不截 → 纯函数把关 → 单键 find
▼
③ trim 汇合（composite_block_manager.cpp:686-695）
│   tail 深 → 用 tail；tail 浅或不整块 → 退回常规网格
▼
④ 挂载开工
│   恢复深度 = 命中深度；forward 余量 = prompt − 深度
▼
【请求 A（先来的那个），发布侧】
⑤ cache_completed_state（linear_state_block_manager.cpp:385-395）
│   chunk 恰好停在 tail 边界 → 盖键 → insert → 写 tail 便签
```

旅程图里准入探测与 trim 汇合两站的细节属第四章；本章钉死的是这一切的地基——那个纯函数。

#### 1.1.2 贴码：函数本体与入口包装

函数本体（第一块，路径 1 守门）：

```cpp
// scheduler_config.h:102-110
// Return a reusable tail boundary using the manager's immutable checkpoint
// stride, rather than reading the live scheduler configuration.
inline size_t linear_state_tail_checkpoint_tokens(size_t prompt_tokens,
                                                  size_t kv_block_size,
                                                  size_t checkpoint_stride) {
  if (checkpoint_stride == 0 || kv_block_size == 0 ||
      prompt_tokens <= kv_block_size) {
    return 0;
  }
```

函数本体（第二块，路径 2-4）：

```cpp
// scheduler_config.h:111-118
  const size_t tail_tokens = (prompt_tokens / kv_block_size) * kv_block_size;
  // An exact prompt hit must leave a token for the current forward. Dropping
  // a KV token alone would leave LINEAR at a different restore boundary.
  if (tail_tokens == prompt_tokens || tail_tokens % checkpoint_stride == 0) {
    return 0;
  }
  return tail_tokens;
```

上下文交代：这是一个头文件里的 `inline` 自由函数，不读任何全局状态。
三个入参全部由调用方传入；注释明说"用 manager 固化的 stride，不读活的调度配置"——防止运行中改配置导致边界漂移。

入口包装（LINEAR 叶成员函数，填好两把尺子再调用纯函数）：

```cpp
// linear_state_block_manager.cpp:343-350
size_t LinearStateBlockManager::tail_checkpoint_tokens(
    const Sequence* seq) const {
  if (seq == nullptr || !options_.enable_prefix_cache()) {
    return 0;
  }
  return linear_state_tail_checkpoint_tokens(
      seq->num_prompt_tokens(), options_.hash_block_size(), block_size());
}
```

三个入参的出处：
`seq->num_prompt_tokens()` 是**完整 prompt** 长度（不截尾，这点在 §1.3 是关键）。
`options_.hash_block_size()` = 128（尺子二，逻辑 KV 块）。
`block_size()` = 8192（尺子一，检查点步距——注意消歧表，别读成 KV 块）。

#### 1.1.3 四条路径逐行（人话翻译）

```
路径 1（守门）  ：stride==0 或 kv_bs==0 或 prompt<=kv_bs → 0
路径 2（算边界）：tail_tokens = ⌊prompt / kv_bs⌋ × kv_bs
路径 3（双排除）：tail_tokens == prompt        → 0   ← 规则一
                  tail_tokens % stride == 0    → 0   ← 规则二
路径 4（唯一出口）：return tail_tokens
```

逐条翻译：
路径 1：没开检查点、没开 KV 块、prompt 连一个完整 KV 块都填不满——任何一种都不值得谈复用。
路径 2：把 prompt 向下取整到**最后一个完整 KV 块边界**（§1.0 已预告）。8191 → 63 块 → 8064。
规则一：边界恰好等于 prompt 全长 → 拒绝（§1.2.1 展开它防什么）。
规则二：边界恰好落在常规检查点网格上 → 拒绝（§1.2.3 展开它防什么）。
路径 4：两条规则都没碰到的边界，才是 tail。

#### 1.1.4 手算：prompt=8192 → 0 的完整推导链

入参 `prompt_tokens=8192, kv_block_size=128, checkpoint_stride=8192`：

```
第 1 步  守门：8192≠0 ✓   128≠0 ✓   8192>128 ✓      → 不提前返回
第 2 步  边界：tail_tokens = ⌊8192/128⌋×128 = 64×128 = 8192
第 3 步  规则一：tail_tokens(8192) == prompt(8192)？ 是 → 命中
第 4 步  规则二：8192 % 8192 == 0？                  是 → 同时命中
第 5 步  结论：双条件在同一次判断里同时为真 → return 0
```

注意第 3、4 步的关系：源码里这是一个 `||` 条件（:114），两个子条件对 8192 **同时成立**。
这就是"prompt=8192 时 checkpoint_tokens == 0"的全部实现——函数从没单独说过 8192，是两条排除规则在这一点的**交汇**。

对照 prompt=8191（唯一出口的样子）：

```
第 1 步  守门：全部通过
第 2 步  边界：⌊8191/128⌋×128 = 63×128 = 8064
第 3 步  规则一：8064 == 8191？ 否
第 4 步  规则二：8064 % 8192 = 8064 ≠ 0？ 是（不触发）
第 5 步  结论：return 8064
```

---

### 1.2 双排除规则：各防一件什么事

#### 1.2.1 规则一（tail == prompt → 0）：防"无活可干"

源码注释直说（scheduler_config.h:112-113）：

```
An exact prompt hit must leave a token for the current forward. Dropping
a KV token alone would leave LINEAR at a different restore boundary.
```

第一句是概念 5 的不变量 U。第二句是 LINEAR 的物理性质（概念 1）。
两句合起来 = 规则一。下面用三幕失败画面把它演出来——**假设放宽规则一会怎样**。

**幕一（埋下）**：
- 请求 A（prompt=8192）把 8192 边界也当 tail 发布进缓存。
- 请求 B 带一模一样的 8192 prompt 进来，tail 点查命中。
- B 的恢复深度 = 8192 = prompt 全长。

**幕二（错在哪）**：
- B 组 batch（把就绪请求凑成一批同时开工）时算待计算量：`8192 − 8192 = 0`，没有 token 可 forward。
- 想救一把：让 KV 侧少挂 1 块，退到 `8192 − 128 = 8064`，重算最后 128 个 token。
- 但 LINEAR 状态是从 8192 的快照**整段**恢复的：S_8192 里已含全部 8192 个 token 的贡献。
- LINEAR 没有"8064 时刻"的中间态——整段折叠，退不了半个边界。
- KV 退了 128，LINEAR 一步不退：两条恢复边界相差 128，**永不收敛**。

**幕三（下游表现）**：
- 路 A（不退，硬凑 0 个 forward token）：产出不了任何输出，请求卡死。
- 路 B（KV@8064 + LINEAR@8192 硬开工）：forward 重算 8064..8191 这 128 个 token。
- 这 128 个 token 的贡献在 LINEAR 状态里已折算过一遍，再 forward = **折算两遍**。
- 之后每步 decode 在双份贡献的起点上复利叠加，输出静默漂移，不崩溃、不报错。

规则一在源头掐灭这一幕：`8064 == prompt → 0`，点查不发起，B 走全量重算。

#### 1.2.2 正面回答："恢复了怎么就错了？那该怎么办？"

先把术语对清：你说的"恢复了就错了"，严格说是**两条恢复边界对不上**——
KV 侧边界（块粒度，可退 128）与 LINEAR 侧边界（检查点粒度，整段）量纲不同。

再回答"那该怎么办"：8192 是**双重对齐边界**（= 1×8192 检查点 = 64×128 KV 块）。
它是"整除边界"，天然不存在一个"既对齐、又给 forward 留余量"的复用点。
你的直觉"不管有没有 tail 都会有问题"**完全正确**：这个 prompt 本来就无解。

规则的价值不是消除问题，而是**把"无解"提前识别成返回 0**。
于是 B 走唯一正确的路：全量重算 8192 个 token，从头折叠，每步都对。
宁可不错用，不可错用——这是整个 prefix cache 的第一性原则。

#### 1.2.3 规则二（tail % stride == 0 → 0）：防"同一把钥匙发两遍"

先手算一个触发例（prompt=8193）：

```
第 1 步  边界：⌊8193/128⌋×128 = 64×128 = 8192
第 2 步  规则一：8192 == 8193？ 否，不触发
第 3 步  规则二：8192 % 8192 == 0？ 是 → return 0
```

8193 没有 tail。但 8193 的复用由**常规链**接管：截 1 token 后 8192/8192 = 1 个可探测检查点（键 hashes[63]），恢复 8192，forward 1 个 token。两条排除规则把工作分得干干净净。

规则二防什么？看两条路径**取键的公式**——它们用的是同一个函数：

```cpp
// block_utils.h:27-36
// Map a completed state boundary to its key in the logical KV hash chain.
// Enforce alignment here for caches constructed without a LINEAR leaf too.
inline size_t linear_state_checkpoint_hash_index(size_t checkpoint_tokens,
                                                 size_t hash_block_size) {
  CHECK_GT(checkpoint_tokens, 0u);
  CHECK_GT(hash_block_size, 0u);
  CHECK_EQ(checkpoint_tokens % hash_block_size, 0u)
      << "Linear checkpoint boundary must align with hash block size.";
  return checkpoint_tokens / hash_block_size - 1;
}
```

常规路径发布 8192 检查点时（linear_state_block_manager.cpp:417-421）：`hash_index = 8192/128 − 1 = 63`。
tail 路径取键时（linear_state_block_manager.cpp:352-356）：`hash_index = token_count/128 − 1`。
若放宽规则二、让 8192 当 tail：tail 路径对 8192 也算出 **hashes[63]**——与常规路径**同一把钥匙**。

于是同一份物理状态、同一个键，被 insert 两遍：双份缓存条目、两本账（游标与便签）同时申报同一边界。
而两个发布守卫各读各的账、互相看不见——常规分支读游标（linear:414），tail 分支读便签（linear:430-431），谁也拦不住对方的 insert（守卫细节在第四章）。

**下游表现**（撞键之后，坏事具体长什么样）：

- 容量双份记账：同一 8192 边界在缓存里占两个名分，驱逐器（以条目为单位，第五章）把它数两次——满仓提前到来，别的新边界反而被挤出去。
- 命中撞大运：B 点查键[63]命中哪一份，取决于插入顺序与驱逐历史；且两份条目背后的账本量纲不同——游标语申报"第 1 个检查点"，便签语申报"tail 深度 8192"，同一把钥匙开出两种深度说法。
- 对账无从裁定：按第三章 insert 协议的结论，insert 只管"缓存知道了"，游标/便签才管"序列记得发到哪"；驱逐走其中一份、留下另一份后，账面显示"边界仍在缓存"，但两本账对它的申报已一存一失——"查到键 = 深度可信"退化成"查到键 = 撞到某段历史"。

规则二在源头掐灭这一幕：8192 % 8192 == 0 → return 0，网格上的边界只许常规路径发布一次。

规则二还有一层更干净的说法：**tail 的存在意义是网格外的缝隙**。
若 tail 落在网格上（%8192==0），它就不是缝隙而是网格本身——常规序列已覆盖，tail 是纯冗余。

#### 1.2.4 双排除合起来：有效 tail 的完整数学画像

```
tail = ⌊prompt / 128⌋ × 128     （对齐尺子二）
tail < prompt                    （规则一：留 forward 余量）
tail % 8192 ≠ 0                  （规则二：不与常规网格撞键）
```

三条联立，推出两个漂亮的不变量：

```
不变量 T1：k×8192 < tail < prompt ≤ (k+1)×8192   （tail 严格夹在两个相邻常规检查点之间）
不变量 T2：forward 余量 = prompt mod 128 ∈ [1, 127]（命中 tail 后恒有活干）
```

T2 的推导：规则一要求 tail < prompt 且 tail 是 128 的倍数，
所以 prompt 不是 128 的倍数，余量 = prompt − ⌊prompt/128⌋×128 = prompt mod 128 ≥ 1。
8191 命中后余 127、10000 命中后余 16、8065 命中后余 1——全部落在这个区间里。

---

### 1.3 探测口径差：match 截尾、tail 点查不截

#### 1.3.1 你的反例，先原样重构

用户原话（10-13 问题 1 后半）：
"查常规检查点的时候退了一个 token，但查 tail 点的时候没有退 token，比如你有 8064 个 token，命中了 tail checkpoint，但你没有新 token 用来 forward，那不还是有问题吗"

重构成推理链：

```
前提 1  常规链探测：调用方截掉最后一个 token          （见 §1.3.2 贴码一）
前提 2  tail 点查：  没看到任何截尾动作               （见 §1.3.2 贴码二）
推论    prompt=8064 时，tail 点查拿 8064 当边界
        → 命中 → 恢复 8064 = prompt 全长 → 0 个 token forward
```

前提 1、前提 2 都是**真的**——源码里确实一边截、一边不截。
但推论不成立。问题出在：tail 点查的输入不是 prompt 原文，是**纯函数的输出**。

#### 1.3.2 回源码裁定：反例构造不出来

贴码一（常规链，调用方截尾）：

```cpp
// linear_state_block_manager.cpp:571-580
std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes) {
  if (token_ids.size() == 0 || prefix_cache_ == nullptr) {
    return {};
  }
  std::vector<Block> blocks = prefix_cache_->match(
      token_ids.slice(0, token_ids.size() - 1), {}, MMData(), block_hashes);
```

读法：第 579-580 行把 token 片截掉最后一个再喂给 match。
为什么常规链必须截？match 按"第 k 个检查点"逐个试键，站位上界由喂进去的 token 数决定。
不截的话，prompt=8192 会探测站位 0（8192 边界）并可能命中——同样的死局。截尾把探测前沿压到 prompt−1。

贴码二（tail 点查，不截——但先过纯函数）：

```cpp
// linear_state_block_manager.cpp:365-371
std::optional<BlockManager::TailPrefixCacheMatch>
LinearStateBlockManager::allocate_tail_shared(Sequence* seq) {
  const size_t token_count = tail_checkpoint_tokens(seq);
  if (token_count == 0 || prefix_cache_ == nullptr) {
    return std::nullopt;
  }
  const std::optional<XXH3Key> hash = tail_checkpoint_hash(seq, token_count);
```

读法：第 367 行先调 §1.1.2 的入口包装——**完整 prompt** 过纯函数。
第 368 行：函数返回 0 就直接 `nullopt`，连键都不算、连 find 都不发起。

现在让 prompt=8064 走一遍：

```
第 1 步  纯函数：tail_tokens = ⌊8064/128⌋×128 = 63×128 = 8064
第 2 步  规则一：8064 == 8064？ 是 → return 0
第 3 步  allocate_tail_shared:368：token_count==0 → nullopt
第 4 步  "8064 命中 tail"这个事件在第一步就死了——点查根本不发起
```

**裁定：反例构造不出来。** 而且不止第一道门，后面还有两颗钉子（三重兜底）：

```
兜底一（规则源头，linear:367-368）
        纯函数保证有效 tail 恒满足 tail < prompt（不变量 T2：余量 ≥ 1）
兜底二（挂载断言，sequence_kv_state.cpp:246-256）
        KV 侧挂载发现深度 == 总长（精确重复）时主动退一块，再 CHECK_LT 断言
兜底三（消费侧，sequence.h:210-212）
        num_need_compute_tokens = 总长 − 恢复深度；深度==全长则待算 0，无法组 batch
```

兜底二的源码（精确重复的退块逻辑 + 断言）：

```cpp
// sequence_kv_state.cpp:246-256
  if (num_shared_tokens == current_total_num_tokens) {
    size_t block_size = bs[0].size();
    CHECK_GT(block_size, 0);
    num_shared_tokens =
        ((current_total_num_tokens - 1) / block_size) * block_size;
    if (shared > 0) {
      shared--;
      bs.pop_back();
    }
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
```

兜底三的源码：

```cpp
// sequence.h:209-212
  // get the number of tokens need compute
  size_t num_need_compute_tokens() const {
    return num_tokens_ - kv_cache_tokens_num();
  }
```

#### 1.3.3 不对称的本质：同一不变量的两种写法

你抓到的"一边截、一边不截"是真实存在的口径差，但它不是漏洞，是**同一不变量的两种实现**：

| | 常规链 | tail 点查 |
|---|---|---|
| 不变量 | 恢复深度 ≤ prompt−1 | 恢复深度 ≤ prompt−1 |
| 实现方式 | **缩输入**：调用方截 1 token 再喂 match | **拒边界**：输入不缩，边界==prompt 的候选在函数内被杀 |
| 把关位置 | 调用方（linear:579-580） | 纯函数规则一（scheduler_config.h:114） |
| 防护对象 | 站位上界（k×8192 ≤ 截后长度） | 候选边界（tail_tokens） |

手算证明"tail 若也切 1"没有任何收益：

```
prompt=8191，按完整 prompt 算：⌊8191/128⌋×128 = 63×128 = 8064
prompt=8191，若按截后 8190 算：⌊8190/128⌋×128 = 63×128 = 8064   ← 同值
```

有效 tail 算出来不变（截尾只动最后不足一块的零头），无效 tail 依然无效。
在 tail 路径上重复设防 = 零收益，所以源码选择了在函数内拒边界、而不是再截一次。

---

### 1.4 8064 站在 0 与 8192 之间：物理状态源，不是序列成员

先画数轴（kv_bs=128，stride=8192）：

```
0 ─────────────────────────── 8064 ─ 8192 ────────────────────────── 16384
│                              ▲     ▲                                 ▲
│                          tail 边界  常规检查点 1                     常规检查点 2
│                          （非成员） （序列成员 #1）                   （序列成员 #2）
│                          键[62]    键[63]                            键[127]
└────────── 63 个 KV 块 ────────┘
```

8064 与 8192 的键只差一格（62 vs 63），**相邻但不冲突**——两族键共用同一条 hash 链，各站各的格。

**为什么说 tail 是"物理状态源"而不是"序列成员"**，三个源码证据：

**证据一（注释直说）**——预取（提前把待复用状态备进现役介质的机制，第五章）进门先查 tail，命中则把物理源放在常规向量之外：

```cpp
// linear_state_block_manager.cpp:150-158
  KVCacheState& host_state = seq->host_kv_state();
  const size_t tail_tokens = host_state.linear_state_tail_tokens();
  if (tail_tokens != 0) {
    // The matched tail is the deepest complete boundary for this prompt.
    // Keep its one physical source outside the regular checkpoint vector,
    // including when the tail crosses one or more regular checkpoints.
    CHECK_GE(num_tokens, tail_tokens);
    CHECK_EQ(num_tokens / block_size(), tail_tokens / block_size());
    return true;
  }
```

注释原文："把它的唯一物理源放在常规检查点向量之外，**即使 tail 跨过一个或多个常规检查点**"。
比如 tail=19968 站在 16384 与 24576 之间、跨过了检查点 2——它依然只是一个源，不是序列里插队的成员。

**证据二（向量语义）**：常规向量按"逻辑检查点站位"索引，深度 = 站位数 × 8192。
tail 挂载后向量长度是 1，深度却是 8064——深度信息**不由向量位置承载**，由 `logical_token_count` 数据携带。

这个字段就是**哨兵**：ProbeResult 上"非 0 即 tail 命中"的标记位——trim 收口时把 tail 深度写进它（composite_block_manager.cpp:751），挂载方见到非零即知改走 tail 挂载（两步协议详见第三章）。

**证据三（两本账）**：常规深度记在 num_cached_blocks（量纲 = 检查点个数，1 基），tail 深度记在 linear_state_tail_tokens（量纲 = token 深度）——后者就是 §1.1.1 挂过名的**便签**。量纲不同，注定分账（账本细节在第三章）。

**为什么必须是内容寻址、不能编进序列**：8064 这个数是**这个 prompt 自己**算出来的（⌊prompt/128⌋×128），换一个 prompt 就是另一个数。
它不是全局网格的固定站位，无法用"第 k 个检查点"的序列语义表达。
每个 prompt 的 tail 键都由 prompt 内容决定。

这恰是 tail 能救 8191 这类不对齐 prompt 的原因：
网格覆盖不到的缝隙，由内容寻址的 tail 补上。

---

### 1.5 手算全程：四个 prompt，每步带中间状态

约定：以下全部 kv_bs=128、stride=8192；"键[N]"指 hash 链第 N 格（0 基）。

#### 1.5.1 幕一 · prompt=8191（tail 的招牌案例）

A 侧（第一个请求，从零算起）——先看调度怎么切 chunk：

```cpp
// scheduler_policy.cpp:509-521
  const size_t offset = current_tokens % chunk_size;
  size_t tokens_to_checkpoint = offset == 0 ? chunk_size : chunk_size - offset;
  if (state.kv_cache_manager != nullptr) {
    // The manager's logical KV block size already includes KV split, matching
    // the hash granularity used by the LINEAR leaf.
    const size_t tail_checkpoint = linear_state_tail_checkpoint_tokens(
        prompt_tokens, state.kv_cache_manager->block_size(), chunk_size);
    if (tail_checkpoint > current_tokens) {
      tokens_to_checkpoint =
          std::min(tokens_to_checkpoint, tail_checkpoint - current_tokens);
    }
  }
  return std::min(candidate_tokens, tokens_to_checkpoint);
```

逐 chunk 手算（封顶 = 到最近检查点的距离、到 tail 的距离、预算，三者取小）：

```
chunk 1：current=0，offset=0 → 到检查点距离 8192
         tail=8064 > 0 → min(8192, 8064−0) = 8064
         切 8064 —— 恰好停在 tail 边界上
chunk 2：current=8064，offset=8064 → 到检查点距离 8192−8064=128
         tail=8064 不> 8064 → 不改
         切 min(127, 128) = 127
分块结果：{8064, 127}
```

chunk 1 结束的瞬间，发布侧闸门放行：

```cpp
// linear_state_block_manager.cpp:385-395
void LinearStateBlockManager::cache_completed_state(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  const size_t checkpoint_tokens = tail_checkpoint_tokens(seq);
  if (checkpoint_tokens == 0 || cached_tokens != checkpoint_tokens) {
    return;
  }
  cache_read_source(seq, seq->kv_state());
}
```

读法：只有"已算到的深度 == tail 边界"才发布。
A 停在 8064 == tail → 发布，键 = 8064/128−1 = **键[62]**。
（调度封顶与发布闸门的配合正是 §1.1.1 旅程图的首尾两站：把 chunk 切在边界上，状态才有发布资格。）

B 侧（同 prompt 重复请求）：

```
问一 常规链：token 截成 8190 → ⌊8190/8192⌋ = 0 个可探测检查点 → 空手
问二 tail 点查：纯函数 → 8064 → 键[62] → find → 命中
深度比较（composite:606-609）：tail 8064 > 常规 0 → 用 tail
恢复深度 8064，forward 余量 = 8191 − 8064 = 127
复用率 = 8064 / 8191 ≈ 98.5%
```

没有 tail，8191 这个不对齐 prompt 在 LINEAR 侧复用深度是 0——这就是 MR 总纲里"8191 从全部作废变成只重算最后一段"的数学来源。

#### 1.5.2 幕二 · prompt=8192（全排除五步链）

```
第 1 步  调度不封顶：tail=0（§1.1.4 双条件交汇）→ 不触发 min
         → 整片 0→8192 一个 chunk 切完
第 2 步  检查点永不发布：
         发布闸门：checkpoint_tokens==0 → return（linear:390-393）
         常规分支只在下一分片的 allocate_prefill 里触发（linear:465）
         8192 一个 chunk 切完，没有"下一分片" → 常规也永远不发布
第 3 步  重复请求 LINEAR 零命中：
         常规链截成 8191 → ⌊8191/8192⌋ = 0 个可探测检查点
         tail 点查：纯函数返回 0（双排除）
第 4 步  KV 命中也全退：
         复合 trim 里 LINEAR 常规深度 regular_tokens=0
         → safe_count=0 → KV 命中块全进 to_drop（composite:697-699、:714-724）
         宁可少复用，不可错复用
第 5 步  正确的全量重算：待算 = 8192 − 0 = 8192，从头折叠
```

这正是 §1.2.2 那句"直觉对"的落地：'不管有没有 tail 都有问题'——
规则把"都有问题"提前变成返回 0，让请求走唯一正确的全量重算。

**第 2、4 步的三个代码名，现场补课**（各自是什么、谁算的、在五步链哪一步起作用）：

先说第 2 步的 `allocate_prefill`——LINEAR 叶给 prefill 分片分配槽位的入口（linear:446，由分配器在分片开工时调进，linear:136）。它在分配槽位之前会先调 `cache_read_source`（linear:465），而**常规检查点的发布只走这一条路**。

- 发布时机是关键：`cache_read_source` 补发布的是"上一分片停在检查点上的完成态"——常规检查点要**等下一分片开工分配槽位时**才被发布。
- 而 §1.5.1 贴过的发布闸门 `cache_completed_state`（linear:385-395）只放行 tail 分支：它调 `cache_read_source` 的前提是 tail 边界非 0 且恰好停在上面，对 8192 直接早退（linear:390-393）。
- 于是 8192 一个 chunk 切完、没有下一分片：`allocate_prefill` 不再被调用 → 第 2 步"检查点永不发布"成立。

```cpp
// linear_state_block_manager.cpp:465-471（allocate_prefill 尾段：先补发布，再分配）
  cache_read_source(seq, kv_state);
  retain_read_source(kv_state);
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    return std::nullopt;
  }
  return allocated;
```

（前半段 linear:449-463 是"恢复边界超前于本层游标"的换窗分支，与本题无关，未贴。）

再说第 4 步的 `safe_count` 与 `to_drop`——它们住在**复合 trim** 里（FLAT_KV_LINEAR 模式的收口函数 `trim_flat_kv_linear`，composite:676）。
函数头注释一句话：只保留"token 边界恰好对上 LINEAR 检查点"的 KV 前缀。三个量各是什么：

- `safe_count`（复合管理器在 trim 里现场算）= KV 侧被允许保留的命中块数：tail 命中按 tail 深度折算，否则按常规深度 `regular_tokens` 折算（贴码一 composite:698-699）；
- `to_drop` = KV 命中块里超出 `safe_count` 的尾部块，被摘进"退还清单"随后释放（贴码二 composite:714-724）；
- `safe_hit_tokens` = `safe_count × 128`，即裁完后的 KV 恢复深度（composite:726）。

8192 走到这：LINEAR 常规深度 0、无 tail → `regular_tokens = 0` → `safe_count = 0` → KV 命中块全进 `to_drop`——第 4 步"宁可少复用，不可错复用"落到实处。

```cpp
// composite_block_manager.cpp:684-699（贴码一 · trim_flat_kv_linear 节选：safe_count 怎么来）
  size_t safe_count = 0;
  bool use_tail_checkpoint = false;
  if (linear.has_value() && kv_block_size > 0 && linear->block_size > 0) {
    if (linear->tail_block.has_value()) {
      const size_t tail_tokens = linear->tail_token_count;
      const size_t tail_safe_count =
          std::min(kv.blocks.size(), tail_tokens / kv_block_size);
      if (tail_safe_count * kv_block_size == tail_tokens &&
          linear->tail_block->is_valid()) {
        safe_count = tail_safe_count;
        use_tail_checkpoint = true;
      }
    }
    if (!use_tail_checkpoint) {
      const size_t regular_tokens = linear->blocks.size() * linear->block_size;
      safe_count = std::min(kv.blocks.size(), regular_tokens / kv_block_size);
```

```cpp
// composite_block_manager.cpp:714-726（贴码二 · 同函数节选：to_drop 怎么退）
  if (safe_count < kv.blocks.size()) {
    ProbeResult drop = {kv.type,
                        kv.leaf,
                        /*blocks=*/{},
                        /*block_size=*/kv.block_size};
    drop.blocks.reserve(kv.blocks.size() - safe_count);
    for (size_t i = safe_count; i < kv.blocks.size(); ++i) {
      drop.blocks.emplace_back(std::move(kv.blocks[i]));
    }
    kv.blocks.resize(safe_count);
    out.to_drop.emplace_back(std::move(drop));
  }
  out.safe_hit_tokens = safe_count * kv_block_size;
```

#### 1.5.3 幕三 · prompt=8064（你的反例数字本身）

```
第 1 步  纯函数：tail_tokens = ⌊8064/128⌋×128 = 8064 == prompt → 规则一 → 0
第 2 步  tail 点查（linear:368）：token_count==0 → nullopt，"命中"不发生
第 3 步  常规链：截成 8063 → ⌊8063/8192⌋ = 0 个可探测检查点 → 空手
第 4 步  8064 < 8192，网格上它以内没有任何常规检查点
第 5 步  LINEAR 侧零复用，全量重算 8064
```

"8064 个 token 命中 tail 且没有新 token forward"的完整答案是：
**能凑出 8064 边界的 prompt 只有 8064 自己，而它恰好被规则一排除。**
能命中 tail=8064 的 prompt 只能落在 (8064, 8192) 区间内（如 8191），余量恒 ≥ 1。

#### 1.5.4 幕四 · prompt=10000（常规与 tail 同场的反例手算）

纯函数逐行：

```
第 1 步  守门：10000 > 128 ✓ → 不提前返回
第 2 步  边界：tail_tokens = ⌊10000/128⌋×128 = 78×128 = 9984
第 3 步  规则一：9984 == 10000？ 否
第 4 步  规则二：9984 % 8192 = 1792 ≠ 0？ 是（不触发）
第 5 步  出口：tail = 9984，键 = 9984/128−1 = 键[77]，余量 = 10000−9984 = 16
```

A 侧分块（封顶三取小逐 chunk）：

```
chunk 1：current=0 → 到检查点 8192；tail=9984>0 → min(8192, 9984) = 8192
         切 8192 —— 停在常规检查点 1 → 发布检查点 1（键[63]）
chunk 2：current=8192，offset=0 → 到检查点 8192
         tail=9984 > 8192 → min(8192, 9984−8192=1792) = 1792
         切 1792 —— 停在 9984 → 发布 tail（键[77]）
chunk 3：切剩下的 16
分块结果：{8192, 1792, 16}，前两段各停在一个可发布边界上
```

B 侧（同 prompt 重复请求）：

```
问一 常规链：token 截成 9999 → ⌊9999/8192⌋ = 1 个可探测检查点
         试键[63] → 命中（最深常规命中 8192）
问二 tail 点查：纯函数 → 9984 → 键[77] → 命中
深度比较（composite:606-609）：tail 9984 > 常规 8192 → 用 tail，常规退还
恢复深度 9984，forward 余量 16
```

这一幕同时演示了 §1.4 的两个论断：
9984 站在检查点 1（8192）与检查点 2（16384）之间的缝隙里；
两族键（63 与 77）同链共存、各站各格，谁也不挡谁。

> **算式勘误登记**：任务底稿给的示例算式"tail = 9999 − 9999%8192"与源码不符。
> 源码公式是 `⌊prompt/128⌋×128`（用完整 prompt、按 KV 块向下取整），不是按 stride 做减法。
> `9999 − 9999%8192 = 8192` 算出来的是常规检查点 1 的位置，不是 tail。本章按源码写 9984。

#### 1.5.5 数字守恒复核（与 v1 笔记 §2.1 对账）

```
prompt=8191 → tail 8064，余量 127        ✓ 源码重算一致
prompt=8192 → 0（双条件同时命中）          ✓ 一致
prompt=8064 → 0（规则一）                 ✓ 一致
prompt=8193 → 0（规则二），常规链接管      ✓ 一致
prompt=20000 → ⌊20000/128⌋×128 = 156×128 = 19968；19968%8192=3584≠0 → 有效，余量 32   ✓ 一致
prompt=16383（逻辑块 256 口径）→ ⌊16383/256⌋×256 = 63×256 = 16128；16128%8192=7936≠0 → 有效   ✓ 一致
键下标：8064→62、8192→63、9984→77、16384→127（均为 0 基；63 块/64 块是块数非下标）    ✓ 一致
```

---

### 1.6 章末自检表

| # | 检查项 | 自评 | 依据 |
|---|---|---|---|
| 1 | 冷读者能否复述主干 | 9/10 | 纯函数四路径 → 双排除各防什么 → 口径差=同一不变量两种写法 → tail 非成员三证据，四层递进各自闭环；"给 forward 留活干"一根线贯穿 |
| 2 | 零跳步跟算 | 9/10 | §1.1.4/§1.5.1-1.5.4 共五组手算，每步带中间状态；prompt=8192→0 给到 `||` 两子条件同时成立的逐步链 |
| 3 | 先教后用 | 通过 | v1 冷读判 5 处 ✗（cold-read-ch1.md D1-D8）后逐项补齐：tail 于 §1.0 先露脸、§1.1.3 注明已预告；便签/prefill chunk/trim 首用即教（§1.1.1）；"叶"半句（§1.0 消歧表引言）；哨兵首用即教（§1.4 证据二）；allocate_prefill/safe_count/to_drop 现场教学+贴码（§1.5.2）；预取/组 batch 首用带注；`block_size` 双义、恢复边界、不变量 U 维持首现即教 |
| 4 | 排版规则①（枚举独立成行） | 通过 | ①②③/幕N/兜底N/第N步 均独立成行，无段内挤排 |
| 5 | 排版规则②（段落 ≤3 行） | 通过 | 全章无超 3 渲染行段落；长推理一律竖排 |
| 6 | 排版规则③（代码只进代码块） | 通过 | 15 个代码块全部从 `git show commit-4` 重新摘录（v1 的 12 个 + §1.5.2 补课新增 3 个），无码字混行 |
| 7 | 排版规则④（旅程/动线配图或表） | 通过 | §1.1.1 旅程图、§1.4 数轴图、§1.3.3 对照表、§1.0 消歧表 |
| 8 | 排版规则⑤（手算每步一行） | 通过 | 五组手算全部每步一行，算式独立成行不埋句子里 |
| 9 | 排版规则⑥（### 编号小节） | 通过 | 章标题 `# 第一章 · …`，小节 `### 1.x` 编号 |
| 10 | 失败画面三段叙事 | 通过 | §1.2.1 幕一埋下/幕二错在哪/幕三下游表现；规则二补齐为对称三段——埋下（同键）→ 错在哪（insert 两遍+守卫互盲）→ 下游表现（容量双份记账/命中撞大运/对账无从裁定） |
| 11 | 用户两个靶子正面回答 | 通过 | §1.2.2 直答"恢复了怎么就错了/该怎么办"；§1.3.2 直答"8064 命中无 token forward"并裁定反例不成立 |

**本质就一句话**：tail 检查点是给"不对齐 prompt"在常规网格的缝隙里开的一扇内容寻址的侧门——纯函数用两条排除规则（留活干、不撞键）保证这扇门只在数学上安全的边界打开。


# 第二章 · 键空间：两把尺子、域桶与 MTP lookahead 的物理本质

> 🔗 **承上启下**：上一章定了「边界在哪」，本章回答「边界怎么变成键」：两把尺子（边界 token 数 → 链上下标）、域桶（为什么固定值还要分桶）、MTP lookahead = end+1 的物理本质。**缓存复用的单位由这章定死**（块 128 / 步距 8192）。

> 本章对应事实底料 §三。核验基准：本 MR head `commit-4`（base `base-commit`）。
> 所有代码经 `git show commit-4:<path>` 逐行回读核验；行号以该 commit 为准。
> 用户靶子：10-12 消息问题 3/4（域桶、invalidate、键盖）、10-13 消息问题 5/6/7（成员变量、lookahead 账目掰扯、巧妙还是补丁的诚实裁决）。

## 0. 前置知识（首现即教）

读本章前，先把六个词钉死。后面任何一处用到，都不再重复定义。

**前缀缓存（prefix cache）**：一个 `键 → 物理块` 的字典。

- 键 = 一段 token 序列的摘要（16 字节）。
- 值 = 存着这段序列 KV 缓存的物理块。
- 语义契约：**同键必同内容**。两个请求算出同一个键，就允许共享同一个块。

**链式哈希（chained hash）**：把序列切成等长格子，逐格叠摘要。

- 第 i 格的键 = hash(上一格的键, 第 i 格的 token)。
- 好处：前缀相同的两个请求，走到任何一格，算出的键都相同——不用逐 token 比较。

**XXH3Key**：16 字节的摘要值（`XXH3_128BITS_HASH_VALUE_LEN = 16`），链上每格一个。

**Block**：物理块句柄，带 5 个字段（id / size / ref_count / manager / hash_value_）。

- `hash_value_[16]` 是块自带的"名牌"（block.h:199）。
- 发布进缓存前，要把链上算出的键抄到这块牌上——这就是后文"盖键"。

**两把尺子**：本章主角。

- 尺子一 checkpoint stride（默认 8192 token）：LINEAR 状态多久存一次检查点。
- 尺子二 hash_block_size（默认 128，= KV 逻辑块）：哈希链一格切多宽。
- 两把尺子**正交**：检查点边界必须落在格子上，但格子比检查点密得多。

**lookahead（前瞻量）**：每个 hash 域自带的属性——MTP 域的哈希键比内容多覆盖下一块首 token 的前瞻量（MTP 域恒 1、TEXT 域恒 0）；数值定义见 §3.3，物理本质（draft 输入右移一位）见 §4。

---

## 1. 两把尺子与统一映射：边界 token 数 → 链上下标

### 1.1 问题重述

用户的原始困惑（10-12 消息问题 3 的引子）：`hash_type_` 和 `hash_block_size_` 在一个 sequence 里看起来是固定的，那 `block_hashes_by_domain_` 这个双层 map 里难道只有一个桶？

要回答它，得先讲清楚键是怎么从"边界 token 数"算出来的——这是整个键空间的地基。

### 1.2 尺子二先行：一格 = hash_block_size 个 token

哈希链的格子宽度由谁定？看 LINEAR 叶管理器的构造选项：

```cpp
// linear_state_block_manager.cpp:33-52（make_linear_state_options，节选）
BlockManager::Options options;
options.num_blocks(num_slots);
options.block_size(chunk_stride);                        // 尺子一：检查点步距
options.hash_block_size(static_cast<uint32_t>(token_block_size));  // 尺子二：哈希格宽
options.hasher_type(hasher_type);                        // hash 域（§2 详讲）
```

三个参数同框出现，各管一件事：

- `block_size(chunk_stride)`：LINEAR 检查点每隔多少 token 存一次（默认 8192）。
- `hash_block_size(token_block_size)`：哈希链一格多少 token（= KV 逻辑块，默认 128）。
- `hasher_type`：用哪个域的身份策略算摘要（§2 详讲）。

构造期还有一条对齐硬门：

```cpp
// linear_state_block_manager.cpp:114-117
if (enable_prefix_cache) {
  CHECK_EQ(chunk_stride % token_block_size, 0)
      << "linear-state checkpoint stride must align with KV block size";
}
```

为什么必须整除：检查点边界要当键用，键要从格子上取——边界落在格子中间就取不出键（§1.5 给失败画面）。

### 1.3 统一映射：一个 8 行的纯函数

本 MR 第 4 个 commit（`commit-4`，"centralize linear checkpoint hash mapping"）把"边界 token 数 → 链上下标"的换算收口成唯一事实源：

```cpp
// block_utils.h:29-36
// Map a completed state boundary to its key in the logical KV hash chain.
// Enforce alignment here for caches constructed without a LINEAR leaf too.
inline size_t linear_state_checkpoint_hash_index(size_t checkpoint_tokens,
                                                 size_t hash_block_size) {
  CHECK_GT(checkpoint_tokens, 0u);
  CHECK_GT(hash_block_size, 0u);
  CHECK_EQ(checkpoint_tokens % hash_block_size, 0u)
      << "Linear checkpoint boundary must align with hash block size.";
  return checkpoint_tokens / hash_block_size - 1;
}
```

逐行读：

- 输入：`checkpoint_tokens`（边界处的 token 深度，如 8192）、`hash_block_size`（格宽，如 128）。
- 两条守卫：边界必须 > 0、格宽必须 > 0。
- 第三条守卫：边界必须整除格宽——8192 % 128 == 0 通过，8192 % 300 = 92 会被拦下。
- 输出：链上第几个格子的键。

三个消费方全部改走这个函数（`git grep` 核验）：

- `linear_state_prefix_cache.cpp:27-32`（probe 侧算键）
- `linear_state_block_manager.cpp:120-125`（发布侧算键）
- `hierarchy_block_manager_pool.cpp:551-552`（D2H 下沉前盖键）

收口动机：三处各自手写 `tokens / bs - 1`，任何一处漂移（写成 `/ bs`、漏了 `-1`）就是静默 cache miss——存 62 取 63，永远对不上，还不报错。

### 1.4 四把钥匙：手算

设定：hash_block_size = 256（KV-split 2 时物理 128 × 2 = 逻辑 256），stride = 8192。

动笔前先交代两件事，各一句：

- 减 1 先照算（边界 N 的键取第 N/bs − 1 格），覆盖关系 §1.5 一页讲透。
- tail 边界的来历（本章只算不展开）：tail 检查点位置 = ⌊prompt / 格宽⌋ × 格宽——把不足一格的尾巴截掉，踩在最后一个完整格的边界上，机制详见第一章。prompt 8191 在格宽 256 下 tail = ⌊8191/256⌋ × 256 = **7936**；同一 prompt 换格宽 128 则 tail = **8064**——token 数相同，格宽不同，tail 落点不同。

- 常规检查点 8192：8192 / 256 = 32，32 − 1 = **31** → 取链上第 31 格的键。
- 常规检查点 16384：16384 / 256 − 1 = **63**。
- tail 边界 7936（prompt 8191 的 tail）：7936 / 256 − 1 = **30**。
- tail 边界 16128（prompt 16383 的 tail）：16128 / 256 − 1 = **62**。

注意两件事：

- tail 键 30 与常规键 31 **同链相邻不冲突**——tail 站在两个常规检查点之间，但它的键落在自己覆盖它的那格上。
- 相邻两个常规检查点隔 8192 / 256 = 32 格。

再算一把测试锚点的钥匙（hash_block_size = 128，无 KV-split）：

- prompt = 8191 的 tail 边界 = ⌊8191/128⌋ × 128 = **8064**（§1.4 开头的 tail 公式）。
- 8064 / 128 − 1 = **62**。
- 即 63 个 KV 块（下标 0..62）的第 62 号——**62 是下标，63 是块数**，两者差 1，四代教学书 round1 在这摔过 5 次。

测试实证（block_manager_test.cpp:787-823，`LargeChunkTailUsesKvHashChainAndLogicalCheckpointCursor`）：

- prompt 8191、boundary_tokens 8064（:802-809）。
- 命中后 `shared_blocks_num(KV) == 63u`（:819）——63 是块数。
- `kv_cache_tokens_num() == 8064u`（:822）——tail 深度。

### 1.5 为什么减 1：覆盖关系

链上第 i 格覆盖 token 区间 `[i·bs, (i+1)·bs)`。

- 边界 N token 的检查点，由**最后一个覆盖它的格子**见证。
- 覆盖 N 的格子是 N/bs − 1（N 是 bs 的倍数时，格子 N/bs−1 的右端点恰好是 N）。
- 不减 1 会取到第 N/bs 格——那格覆盖的是 N 之后的 token，键对不上内容。

### 1.6 对齐 CHECK 的失败画面

假如没有 `CHECK_EQ`，格宽 300、边界 8192：

- 8192 / 300 = 27（整除截断），公式给下标 26。
- 但 27 格（下标 0..26）整条链只覆盖前 8100 token——第 26 格本身覆盖 [7800, 8100)——8100..8192 这 92 个 token 没进任何摘要。
- 两个前 8100 token 相同、后 92 个不同的请求算出**同一个键** → 撞键 → 错误共享。
- CHECK 把这个未来事故提前到启动时刻崩掉（宁崩不错）。

### 1.7 KV-split：为什么必须用逻辑块

```cpp
// scheduler_policy.cpp:512-514
// The manager's logical KV block size already includes KV split, matching
// the hash granularity used by the LINEAR leaf.
```

- KV-split 2 时：物理块 128，逻辑块 256（一条逻辑块 = 2 条物理块拼成）。
- 哈希链的格子是**逻辑**块——因为键要和 KV 共享链对齐（共享以逻辑块为单位）。
- 用物理 128 算键的下场：16128 / 128 − 1 = 125，而正确答案是 62。
- 存的时候键在 62，取的时候算出 125 → 永远 miss，无日志无报错。

**本节核心结论**：键 = 链上第 `边界/格宽 − 1` 格的摘要；两把尺子正交，检查点必须踩在格子上。

---

## 2. 域桶：`block_hashes_by_domain_` 多桶结构与"台牌"

### 2.1 用户问题 3 重述

> "hash_type_ 和 hash_block_size_ 都是固定的吧，那 block_hashes_by_domain_ 里面难道只有一个桶吗？我只看到 update_block_hashes 的时候更新了一下全局的 hash_type_，这是什么情况呢？什么时候调用这个函数？调用了这个函数，那别的 key 还能拿到吗？"

这题包含三问，逐一说破。

### 2.2 概念地图：一图看懂数据结构

```text
Sequence（一个请求的序列对象）
 │
 └─ block_hashes_by_domain_          ← 双层 map（sequence.h:648-655）
     │
     ├─ [TEXT]        ← 第一维：hash 域（谁的身份策略）
     │   ├─ [128] → [h0, h1, ..., h62]   ← 第二维：格宽 → 键向量
     │   └─ [256] → [h0, h1, ..., h31]
     │
     └─ [MTP_TEXT]
         └─ [128] → [h0', h1', ...]      ← 另一条独立的链
 │
 ├─ hash_type_        ← "台牌"：最近一次 update 选中的域（sequence.h:658）
 └─ hash_block_size_  ← "台牌"：最近一次 update 选中的格宽（sequence.h:657）
```

贴码（成员声明原文）：

```cpp
// sequence.h:648-658
// Hash domains and strides keep independent incremental chains. LINEAR
// uses the base domain while speculative KV uses the MTP domain; neither
// probe discards the other's cache. Inserting another bucket preserves
// existing vector storage, so previously returned Slices remain valid.
std::unordered_map<BlockHasherType,
                   std::unordered_map<uint32_t, std::vector<XXH3Key>>>
    block_hashes_by_domain_;

// Domain and stride selected by the most recent update_block_hashes().
uint32_t hash_block_size_ = 0;
BlockHasherType hash_type_ = BlockHasherType::TEXT;
```

注释第一句就是答案的一半：**各域各格宽保有独立增量链，LINEAR 用基础域、投机 KV 用 MTP 域，任何一方 probe 都不丢弃另一方的缓存**。

### 2.3 第一问正解："固定值为何多桶"

用户的误解在哪：把 `hash_type_` 当成了配置项（构造期定死）。它不是配置，是**最近选择器**——"当前台牌"。

真相分三层：

- **多桶并存是常态**：一个序列同时有 KV 页和 LINEAR 页，probe 时两页各查各的键——KV 用 MTP 域（若开了 MTP），LINEAR 用 TEXT 域。两条链必须同时在 map 里活着。
- **运行中会切台牌**：`update_block_hashes(256, MTP_TEXT)` 把台牌切到 MTP 域 256 格；紧接着 `update_block_hashes(128, TEXT)` 又切回 TEXT 域 128 格。DSV4 甚至连续用三种格宽（base / 4·base / 128·base）probe（sequence.h:300-305 注释原文）。
- **桶的数量**：无 MTP 时 1 个域（KV+LINEAR 同用 TEXT@128，各格宽一桶）；开 MTP 时 2 个域——MTP_TEXT@128 给 KV 用，TEXT@128 给 LINEAR 用。

### 2.4 为什么 LINEAR 单独剥出 TEXT 域：贴码

```cpp
// composite_block_manager.cpp:178-190（构造 LINEAR 叶，节选）
leaves.emplace(
    BlockType::LINEAR,
    CompositeBlockManager::LeafEntry{
        maybe_concurrent(
            std::make_unique<LinearStateBlockManager>(
                ...
                options.hasher_type() == BlockHasherType::MTP_MM
                    ? BlockHasherType::MM            // :182-183 剥掉 MTP_ 前缀
                : options.hasher_type() == BlockHasherType::MTP_TEXT
                    ? BlockHasherType::TEXT           // :184-185 同上
                    : options.hasher_type(),          // :186 原样透传
                ...),
```

这个三元式的意思是：引擎级选项带着 MTP 变体（MTP_TEXT）进来，构造 LINEAR 叶时**强制剥掉 MTP 前缀**、只留基础域。

为什么剥：LINEAR 状态是递归折叠，不前瞻（§5 专门裁决）。这里先记住事实——**键空间因此分成两个域**，这就是多桶的物理来源。

### 2.5 第二问正解：update_block_hashes 全流程

什么时候调用？13 处调用点（`git grep` 核验：block_manager_impl 2 处、composite 3 处、linear_state 4 处、sliding_window 2 处、hierarchy 2 处）。典型三处：

- probe 前：composite_block_manager.cpp:583-587（LINEAR 用 TEXT 域、KV 用叶子自带域，各自先 update 再取链）。
- 发布前：linear_state_block_manager.cpp:417-420。
- D2H 配对前：hierarchy_block_manager_pool.cpp:553-555。

全流程贴码（很短，值得整读）：

```cpp
// sequence.cpp:997-1011
void Sequence::update_block_hashes(uint32_t block_size,
                                   BlockHasherType hasher_type) {
  if (block_size == 0) {
    return;
  }
  hash_type_ = hasher_type;                 // ① 切台牌：域
  hash_block_size_ = block_size;            // ② 切台牌：格宽
  const Slice<int32_t> tokens = hash_tokens(hasher_type);   // ③ 备料（§2.7）
  extend_prefix_hashes(hasher_type,         // ④ 增量延长"这个"桶
                       mm_data_,
                       tokens,
                       block_size,
                       num_hash_blocks(hasher_type, tokens.size(), block_size),
                       block_hashes_by_domain_[hasher_type][block_size]);
}
```

四步读法：

- ①② 把台牌换成本次要的 (域, 格宽)。
- ③ 从 tokens_ 里切出"能进哈希的 token"（MTP 域在 overlap 模式下要裁草稿，§2.7）。
- ④ `block_hashes_by_domain_[hasher_type][block_size]` 这个下标表达式**只触碰这一个桶**——双层 map 的 `operator[]` 找到已有桶就地延长；没有则新建。
- 关键：`extend_prefix_hashes` 从 `hashes.size()` 起接龙（block_hasher.cpp:156），已有的格子一个不重算。

**它对其他桶一个字节都不碰**——既不删也不改。这就是第三问的答案。

### 2.6 第三问正解："调用了这个函数，别的 key 还能拿到吗"

能。两条证据链：

证据一，读取函数是"按台牌两级 find"，不依赖调用顺序：

```cpp
// sequence.cpp:976-986
Slice<XXH3Key> Sequence::block_hashes() const {
  const auto domain = block_hashes_by_domain_.find(hash_type_);
  if (domain == block_hashes_by_domain_.end()) {
    return {};
  }
  const auto it = domain->second.find(hash_block_size_);
  if (it == domain->second.end()) {
    return {};
  }
  return it->second;
}
```

- 读的是台牌指向的桶；find 不命中返回空链，不抛不崩。
- 想读别的桶？下次 `update_block_hashes` 切台牌即可——旧桶原地不动，切回来还在。

证据二，测试直接演练了"切台牌不丢桶"（mtp_prefix_cache_test.cpp:194-211）：

- 65 token，先 update(16, MTP_TEXT) 得 4 格，再 update(32, MTP_TEXT) 得 2 格（65/16 = 4、65/32 = 2 可直接复算；MTP 域格数公式 (N−1)/bs 到 §4.3 才教，此参数下与 TEXT 公式同值）。
- 改写 token 32 后两桶全部失效（invalidate 遍历**所有**桶，§3）。
- 重新 update(16) / update(32)，两个桶各自重算。
- 最后 update(16) 后 `block_hashes()[0] == first_hash`——第一格摘要与最初完全一致（确定性）。

### 2.7 hash_tokens：token 从哪来、MTP 为什么要裁

```cpp
// sequence.cpp:988-995
Slice<int32_t> Sequence::hash_tokens(BlockHasherType hasher_type) const {
  if (block_hash_lookahead(hasher_type) == 0 ||
      !sequence_params_.enable_schedule_overlap) {
    return tokens();
  }
  return tokens().slice(
      0, std::min(num_tokens_, static_cast<size_t>(cur_generated_token_idx_)));
}
```

- token 从 `tokens_`（序列自己的 token 数组）来。
- TEXT 域（lookahead=0）或不开 overlap：全量返回。
- MTP 域 + overlap 开：**裁到确认游标** `cur_generated_token_idx_`——草稿/占位 token 不许进 MTP 身份（sequence.h:297 注释原文 "MTP identity must never include overlap placeholders or unverified tokens"）。
- 测试锚点：mtp_prefix_cache_test.cpp:216-226（`OverlapPlaceholdersNeverEnterHashChain`）。参数：16 个已确认 token、bs=16、overlap 开。追加占位 token（id=-1）后 update，能进 MTP 身份的仍只有 16 个确认 token，可算格数 (16−1)/16 = **0**——链为空；确认该占位（写入真实 token）后确认数变 17，(17−1)/16 = **1**——出 1 格（格数公式为何减 1：§4.3）。

### 2.8 用户问题 5：临时状态为何做成成员变量（诚实裁决）

> "既然 hash_type_ 和 hash_block_size_ 都是临时状态，那有必要把他们作为成员变量呢，当成临时变量不行吗，看着挺别扭的。"

先数代价。`block_hashes()` 在 xllm/ 下有 **13 处调用点**（回源码 grep 重数；v1 笔记记 12 处，差 1，已登记）。

- 方案 A（成员变量台牌，现状）：13 处读点零参数 `seq->block_hashes()`。
- 方案 B（临时变量）：13 处全部改两参 `seq->block_hashes(type, size)`，每次调用都要回答"我该传哪个域"——而答案散在各叶子的 options 里，传错编译器不报错。

裁决：**合理妥协，不是好设计也不是烂设计**。

- 得：读点简洁、不可能传错参数。
- 失：隐藏可变状态——`block_hashes()` 的返回值依赖"最近一次谁调过 update"，这是时序耦合，编译期无防护。
- 失败画面：某个新调用方忘了先 update，`block_hashes()` 按旧台牌返回别的域的链 → LINEAR 发布拿到 MTP 域的键 → 块盖错键 → 未来 probe 永远 miss，无日志。
- 这类 bug 只能靠 code review 和测试兜（上面 :194-211 的测试就是兜底之一）。

**本节核心结论**：多桶 = 多 hash 域并存 + 运行中切台牌；update 只延长自己那桶，切台牌不删桶。

---

## 3. invalidate_block_hashes_from：脏起点回退的逐步手算

### 3.1 用户问题 4 重述

> "'stale_token（脏起点）：token_index − lookahead。为什么要回退？'……stride 4 改写 token 8 为例……是什么意思，看不懂。""为什么 token_index − lookahead 就能得到 stale_token，太细节了，看不懂，讲清楚。"

### 3.2 什么时候调用：两个入口

```cpp
// sequence.cpp:567-570（update_last_step_token：MTP/overlap decode 提交确认 token）
tokens_[cur_generated_token_idx_] = token_id;
// Overlap/MTP may rewrite tokens at decode positions; drop any cached block
// hash from this position onward so it is recomputed when next needed.
invalidate_block_hashes_from(cur_generated_token_idx_);
```

```cpp
// sequence.cpp:587-596（update_token：PD 回填 / beam 改写任意位置）
void Sequence::update_token(size_t index, const Token& token) {
  ...
  tokens_[index] = token_id;
  // A rewritten token invalidates the cached hash of its block and all
  // subsequent blocks; recompute lazily on the next update_block_hashes().
  invalidate_block_hashes_from(index);
```

共同点：**有 token 被改写**（占位被真实值覆盖、beam 换路、PD 回填）。改写让"链上已有的摘要"和"token 现值"对不上了。

### 3.3 逐行贴码

```cpp
// sequence.cpp:1013-1027
void Sequence::invalidate_block_hashes_from(size_t token_index) {
  // A rewrite also invalidates the preceding MTP block when its lookahead
  // consumes the changed token. Each cached domain owns that dependency.
  for (auto& [hasher_type, strides] : block_hashes_by_domain_) {   // ① 遍历所有域
    const size_t lookahead = block_hash_lookahead(hasher_type);    // ② 各域自己的前瞻量
    const size_t stale_token =
        token_index > lookahead ? token_index - lookahead : 0;     // ③ 回退
    for (auto& [block_size, hashes] : strides) {                   // ④ 遍历该域所有格宽
      const size_t first_stale_block = stale_token / block_size;   // ⑤ 脏起点所在块
      if (first_stale_block < hashes.size()) {
        hashes.resize(first_stale_block);                          // ⑥ 从这里砍断
      }
    }
  }
}
```

六步读法：

- ① 不是只清当前台牌的桶——**所有域的所有桶一起判脏**（改一个 token，所有域里掺过它的摘要都脏了）。
- ② 每个域查自己的前瞻量：MTP 域 = 1，TEXT 域 = 0（block_hasher.cpp:72-75）。
- ③ 脏起点 = 改写位置 − 该域前瞻量（下节手算为什么）。
- ⑤ 脏起点落在第几块：整除。
- ⑥ `resize` 砍断——保留 [0, first_stale_block)，后面的下次 update 懒重算。

### 3.4 核心手算：改写 token 8，格宽 4（用户点名的例子）

先画覆盖图。横轴是 token 下标，两行分别是两个域里"每格摘要实际吃进哪些 token"：

```text
token 下标:    0   1   2   3 | 4   5   6   7 | 8   9  10  11 |
               └─────────────┼───────────────┼───────────────┘
TEXT 域（lookahead=0，每格摘要吃 4 个）:
  格 0 摘要吃: [0,1,2,3]      格 1 吃: [4,5,6,7]     格 2 吃: [8,9,10,11]
MTP 域（lookahead=1，每格摘要多吃下一格首 token）:
  格 0 摘要吃: [0,1,2,3,4]    格 1 吃: [4,5,6,7,8]   格 2 吃: [8,9,10,11,12]
                                        ^^^                 ^^^^^^^^^^^
改写 token 8。谁脏？
```

图中 `^^^` 标记的是 token 8 的两个落点：MTP 格 1 靠前瞻吃进它，MTP 格 2 靠本格覆盖吃进它（`^^^^^^^^^^^` 即格 2 的覆盖区间）。

逐格判断：

- TEXT 格 0：吃 [0..3]，不含 8 → 干净。
- TEXT 格 1：吃 [4..7]，不含 8 → 干净。
- TEXT 格 2：吃 [8..11]，**含 8 → 脏**。
- MTP 格 0：吃 [0..4]，不含 8 → 干净。
- MTP 格 1：吃 [4..7] **再加 8** → **脏**（前瞻吃过的 token 变了，摘要作废）。
- MTP 格 2：吃 [8..12]，含 8 → 脏。

现在验证公式，逐步代入：

MTP 域（lookahead = 1）：

- 第 ③ 步：stale_token = 8 − 1 = **7**。
- 第 ⑤ 步：first_stale_block = 7 / 4 = **1**（整除，1.75 截断为 1）。
- 判定：首脏块 = 1——与逐格判断一致：格 1 脏、格 0 保住。**不多裁不少裁**。

TEXT 域（lookahead = 0）：

- 第 ③ 步：stale_token = 8 − 0 = **8**。
- 第 ⑤ 步：first_stale_block = 8 / 4 = **2**。
- 判定：首脏块 = 2——格 0、格 1 保住。同样精确。

回退的物理意义一句话：**回退量 = 覆盖量的逆运算**。

- MTP 格 1 的摘要当初多吃了 token 8（覆盖量 +1）。
- 现在 token 8 变了，找首脏块时就得往回多退 1 格（回退量 +1）。
- 覆盖时多吃了谁，作废时就多退给谁——账目两边严格对齐。

### 3.5 失败画面：三段叙事（假如不回退）

**第一段（脏起点裁少了）**：错误版 invalidate 不做 lookahead 回退，stale_token 直接取 8，首脏块 = 8/4 = 2，只砍掉格 2。MTP 域的格 1——那个摘要里掺着**旧 token 8** 的格子——原封不动留在链上。

**第二段（命中错块）**：序列继续走，token 8 的新值写进 KV，后续 forward 从 8 起重算。发布时格 1 挂着**旧键**（按旧 t8 算的摘要）进缓存。

未来来了个请求 B，它在位置 8 的 token 恰好等于**旧值**。B 按自己的 token 算链：格 1 摘要 = hash(h0, t4..t7, 旧 t8) = 旧键 → find 命中。但 B 拿到的块，内容是按**新 t8** 写的。

**第三段（恢复错位）**：B 挂载这个块恢复 KV——键说"这是 [t4..t8=旧值] 的块"，内容却是"按 t8=新值 算的"。MTP 草稿半边从第一步就是错的，后续每步 decode 在错误状态上复利叠加。同键不同内容，前缀缓存的身份契约（§0）从根上被打破。

补一句诚实的幅度说明：target 半边的 KV 因果地只依赖 t4..t7，仍是干净的；脏的是 draft 半边——所以这个事故的实测表现是提案质量下降、verify 拒绝率升高、**静默掉速**，不是立即输出错误。但"静默"恰恰比"报错"更难排查。

### 3.6 用户问题 4b：什么是"键盖"

"盖键"= 把链上算出的 16 字节摘要抄到 Block 自带的 `hash_value_` 名牌上：

```cpp
// block.h:163-171
// NOTE: Below block `hash_value_` is used for prefix cache and
// for recording the hash value of the current block and previous blocks.
// hash_value_ = Hash(current_block, Hash(pre_block)).
const uint8_t* get_immutable_hash_value() const { return hash_value_; }
uint8_t* get_mutable_hash_value() { return hash_value_; }

void set_hash_value(const uint8_t* hash_value) {
  memcpy(hash_value_, hash_value, XXH3_128BITS_HASH_VALUE_LEN);
}
```

- 类比：行李箱挂名牌。数据（行李）和键（名牌）本来是两样东西，发布前把名牌钉上去，键跟着数据走。
- 为什么需要：D2H 异步拷贝只搬块字节，不搬"链上下标"这种上下文。目的块落地后要进 host 缓存，必须自带键才能被对账和命中。
- 三个盖键时机（发布 linear:440 / 预取分配 block_manager_impl.cpp:302 / D2H 配对）里最典型的是配对：
  `pair->dst.set_hash_value(pair->src.get_immutable_hash_value())`（hierarchy_block_manager_pool.cpp:565）。
- 为什么块上"可能没有 hash"：哈希链节节都有（每 128 token 一格），但**键只盖完成可复用边界的块**——decode 活写槽的内容还没定形，盖了也是假键。这是设计不是失误。
- 不盖的失败画面：host 侧拿到一份 D2H 来的数据，块上 hash_value_ 是出厂值（未初始化垃圾）→ 拿垃圾键登记进缓存 → 预取命中判定永远 miss → 白拷一份永远用不上的状态。

**本节核心结论**：stale_token = 改写位 − 前瞻量，回退量是覆盖量的逆运算；不回退就把"掺过旧 token 的摘要"留在链上，等未来撞键。

---

## 4. MTP lookahead = end+1：用户"掰扯"题的完整账本

### 4.1 用户问题 6 重述

> "是不是算 hash 的时候要多算一个 token，但是算 block 数却减去 lookahead，这是什么逻辑的，给我掰扯掰扯。那 hash 和内容能对应上吗？mtp 为什么这么特殊呢？这是最佳做法吗？"

四问，本节逐一对账。

### 4.2 机制本体：一个只做加一的包装器

```cpp
// block_hasher.cpp:31-49（匿名命名空间）
class MtpBlockHasher final : public BlockHasher {
 public:
  explicit MtpBlockHasher(std::unique_ptr<BlockHasher> hasher)
      : hasher_(std::move(hasher)) {}

  void compute(const Slice<int32_t>& token_ids,
               int32_t start_token_idx,
               int32_t end_token_idx,
               const uint8_t* pre_hash_value,
               XXH3Key& hash_key) override {
    CHECK_LT(static_cast<size_t>(end_token_idx), token_ids.size());   // :41
    hasher_->compute(token_ids,
                     start_token_idx,
                     end_token_idx + 1,        // :44 ← 全部实现就是这一行
                     pre_hash_value,
                     hash_key);
  }

 private:
  std::unique_ptr<BlockHasher> hasher_;
};
```

逐行读：

- 上层 `extend_prefix_hashes` 说"算第 i 格"，给的区间是 `[i·bs, (i+1)·bs)`（block_hasher.cpp:173-175）。
- 包装器把右端点 +1 再喂给底层（:44）。
- 于是 MTP 域第 i 格的摘要覆盖 `[i·bs, (i+1)·bs]`——**自己那 bs 个 token，加下一格的首 token**。
- `CHECK_LT`（:41）保证 +1 之后仍在数组内——越界即崩，宁崩不错。

这就是"lookahead=1"的全部实现：一个 20 行的装饰器（decorator），一行改动。

### 4.3 账本另一边：块数减 lookahead

```cpp
// block_hasher.cpp:78-86
size_t num_hash_blocks(BlockHasherType type,
                       size_t num_tokens,
                       size_t block_size) {
  const size_t lookahead = block_hash_lookahead(type);
  if (block_size == 0 || num_tokens <= lookahead) {
    return 0;
  }
  return (num_tokens - lookahead) / block_size;
}
```

- TEXT 域：可算格数 = N / bs。
- MTP 域：可算格数 = (N − 1) / bs。

### 4.4 掰扯：多算一个 token 与少算块数，怎么对上的

用户觉得矛盾的两件事，其实是**同一条依赖链在两头的投影**。分四步对账。

**第一步：每格多算一个 token（覆盖侧）**。

- 第 i 格摘要 = hash(上格键, t[i·bs .. (i+1)·bs] )，bs+1 个 token。
- 最后一格（i = 格数−1）要吃到下标 N−1 之外的一个 token——**只有当数组里真有多出来的 token 时，这格才可算**。

**第二步：格数按 lookahead 收缩（可用性侧）**。

- N 个 token（下标 0..N−1），第 i 格可算的条件：`(i+1)·bs ≤ N−1`。
- 解出最大格数 = (N−1)/bs——恰好是"块数减 lookahead"。
- 手算对照（bs=16）：
  - N=33：TEXT 33/16=2 格；MTP (33−1)/16=2 格——**相同**。
  - N=32：TEXT 32/16=2 格；MTP (32−1)/16=**1 格**——MTP 少一格。
- 结论：两域格数**只在 N 恰为 bs 整倍数时差 1**。平时不差，差的那一下正是"最后一格想多吃一个 token、数组里没有"的时刻。

**第三步：为什么键必须多吃（物理侧）**。

- MTP 的 KV 块是**复合材料**：一个 Block 的字节 = target 层 KV + draft 层 KV。
- draft 输入右移一位：draft 层槽位 j 的 KV 依赖 t_{j+1}（"看着当前 token 算下一个 token 的草稿"）。
- 推论：块 i 的最后一个槽位（j = (i+1)·bs − 1）的 draft 半边依赖 t_{(i+1)·bs}——**下一格的首 token**。
- 所以块 i 的**内容**天然依赖 bs+1 个 token。键要如实地把这 bs+1 个 token 全部吃进去——键覆盖 ⊇ 内容依赖，这是 key→value map 语义（同键必同内容，§0）的基本要求，不是性能优化。

**第四步：合账**。

```text
内容侧：块 i 的字节依赖 t[i·bs .. (i+1)·bs]        ← bs+1 个 token
   ↕ 必须严格对齐
键  侧：格 i 的摘要覆盖 t[i·bs .. (i+1)·bs]        ← end+1（§4.2）
   ↕ 因此推出
可用侧：最后一格需要下标 N 存在 → 格数 = (N−1)/bs   ← 减 lookahead（§4.3）
```

三处用的是**同一个** `block_hash_lookahead(type)`（block_hasher.cpp:72-75，MTP 域恒 1、其余恒 0）——单一事实源，账本两边不可能对不齐。

发布侧还有第四处对齐：

```cpp
// composite_block_manager.cpp:62
const size_t token_end = end * block_size + block_hash_lookahead(hasher_type);
```

发布也要等到依赖 token 齐了才发——四处（键覆盖、格数、失效回退、发布边界）共用一个前瞻量。

### 4.5 hash 和内容能对应上吗：测试实证

对应得上，而且有反事实测试钉死（mtp_prefix_cache_test.cpp:578-609，`HostMatchRejectsDifferentNextToken`）。

设定：源请求 33 个 token 全是 7，bs=16，已发布（MTP 域 2 格）。两个分叉请求各改一个 token 为 8：

- 分叉一改 t_16（fork[16]=8，:592-594）：
  - 格 0 摘要覆盖 t[0..16]，含 t_16 → 摘要变 → miss。
  - 格 1 依赖格 0 → 也 miss。
  - 结果：`kv_cache_tokens_num() == 0u`（:597）——**零命中**。
- 分叉二改 t_32（fork[32]=8，:600-602）：
  - 格 0 摘要覆盖 t[0..16]，不含 32 → **不变，命中**。
  - 格 1 摘要覆盖 t[16..32]，含 t_32 → miss。
  - 结果：`kv_cache_tokens_num() == 16u`（:605）——**恰好一块**。

这两行断言就是"键精确镜像内容依赖"的实验证明：

- 每格的拒绝共享，对应那格 draft 半边**真实存在的内容差异**——没有一格是过保守。
- 对照组（TEXT 域，:228-246 `CopyRetainsMtpInvalidationPolicy`）：同样的 33 token、TEXT 域、改 t_16 后格 0 键不变（EXPECT_EQ，:242-243）——证明"多算一个 token"确实只发生在 MTP 域，是域差异不是全局行为。

### 4.6 mtp 为什么特殊、是不是最佳做法

**为什么特殊**：唯一一个"块内容依赖未来 token"的块类型。KV 的 target 半边因果（只看过去）；LINEAR 状态递归（只折叠过去）；只有 MTP 的 draft 半边，因为输入右移一位，物理上就要吃下一个 token。特殊的是**物理**，不是作者的心情。

**是不是最佳做法**：在"键层解决"这条路线内，是的——复用条件被编码进键本身，零运行时校验、零特判分支。

但"键层解决"不是唯一路线：vLLM 选择在消费层丢边界块重算（每次命中无条件少复用一块），SGLang 的树结构天然免疫（draft 不进树）。三家的取舍对照属另章范围。

本章只钉死一件事：**xLLM 选键层方案后，end+1 与减 lookahead 是这套方案的必然账目，不是两处独立拍脑袋**。

**本节核心结论**：键多吃一个 token 与格数减 lookahead 是同一条物理依赖（draft 右移）在覆盖侧和可用性侧的两次显影，账本严格对齐。

---

## 5. 问题 7 诚实裁决：普通 KV 回退、LINEAR 不前瞻，是巧妙还是补丁

### 5.1 用户原话重述

> "按照你的说法应该只有普通 kv 才会回退 hash 吧，因为 linear 页是非 mtp 的，不会前瞻，那为什么普通 kv 多吃呢，为什么会区别对待，这到底是巧妙的设计还是傻逼设计，这么多弯弯绕，是开发者为了炫技吗，还是开发者脑子不灵光，乱打补丁？"

先纠正题面里的一个方向偏差（术语对清）：

- "普通 KV 回退"——回退发生在 **MTP 域**（开着 MTP 时 KV 页用的域）。
- 不开 MTP 时 KV 也用 TEXT 域、也不回退。
- 所以区别对待的分界线不是"KV vs LINEAR"，是"**域的前瞻量**"：`block_hash_lookahead(type)` 谁非零谁回退。

### 5.2 为什么 MTP 域要回退、LINEAR 不需要

MTP 域回退的理由 §3.4 已手算：格 1 的摘要掺过 t_8，t_8 变则格 1 脏——不回退就留脏键。

LINEAR 不回退的理由是数学，不是工程偏好：

- LINEAR 检查点是递归状态的快照：S_n = f(t_0 .. t_{n−1})。
- 递归状态**不可能**依赖未来 token——这不是实现选择，是状态定义本身。
- 所以 TEXT 域（LINEAR 用的域）lookahead=0，改 token 8 时格 1（覆盖 [4..7]）干干净净，不该裁也不必裁。

反事实四行表（bs=16，33 个已确认 token，改 t_16 触发 invalidate(16)）：

| 策略 | lookahead | stale | 首脏块 | 作废量 | 有无道理 |
|---|---|---|---|---|---|
| ① MTP 域现状（KV 用） | 1 | 15 | 0 | 2 格全废 | 有：draft 半边真吃 t_16 |
| ② TEXT 域现状（LINEAR 用） | 0 | 16 | 1 | 只废格 1 | 有：格 0 的 S_16=f(t_0..t_15) 不含 t_16 |
| ③ LINEAR 若改用 MTP 键 | 1 | 15 | 0 | 2 格全废 | **无：纯自残**——多废一格好键，买到零防御 |
| ④ MTP 若改用 TEXT 键 | 0 | 16 | 1 | 留脏格 0 | **无：留脏键**，§3.5 三段失败画面 |

④ 就是撞键事故，①③ 说明两个域各自的 lookahead 都取在"物理依赖到哪里、键就覆盖到哪里"这条线上。**LINEAR 剥 MTP（composite:182-186）和 MTP 吃 end+1（block_hasher.cpp:44）是同一条公理的正反两个方向**。

### 5.3 "区别对待"在代码里的真实形态

不是 if-else 特判，是**参数化**：

```cpp
// sequence.cpp:1016-1019（invalidate 内层）
for (auto& [hasher_type, strides] : block_hashes_by_domain_) {
  const size_t lookahead = block_hash_lookahead(hasher_type);   // 每域自查
  const size_t stale_token =
      token_index > lookahead ? token_index - lookahead : 0;
```

- 一条循环，各域按各自的 lookahead 值回退——没有一处 `if (type == MTP)` 的分支写死在逻辑里。
- 新增一个域（比如 MTP_MM）：`block_hash_lookahead` 返回 1，invalidate 自动跟上。
- 域差异的全部知识收口在两个纯函数里：`block_hash_lookahead`（前瞻量）与 `mtp_hasher_type`（何时升域，block_hasher.cpp:55-67，只在 num_speculative_tokens>0 且算法是 MTP 时才升）。

### 5.4 三类证据齐开（禁顺着夸、禁辩护）

**好设计的证据（机制层）**：

- 回退量 = 覆盖量的逆运算，账目可手算复核（§3.4，不多裁不少裁）。
- 参数化而非特判（§5.3）。
- 反事实测试把两个方向的错误都钉死了（四行表 ③④ 对应的测试都在）。
- 与 vLLM 独立遇到同一物理问题、写出同款解法（vLLM find_longest_cache_hit docstring 亲口承认 draft KV 越界依赖）——独立收敛是物理必然性的旁证。

**合理妥协的证据（路线层）**：

- 键层方案的税：全链 13 处 update-then-read 时序耦合（§2.8）、三消费方对齐（§1.3）。
- 域分桶选择器的复杂度：读代码的人必须同时持有"域、格宽、台牌"三个变量才能读懂 block_hashes() 的返回值。
- 这些税换来"复用条件零运行时校验"——路线内自洽，但路线本身有对手（消费层方案、结构层方案）。

**欠债的证据（文档层）**：

- **没有一处注释把"块是复合材料所以键要 lookahead"这条因果链串起来**。
  代码里散落着碎片：block_hasher.h:39 "MTP variants cover the next confirmed token as well as the block itself"，
  以及 invalidate 里那句 "A rewrite also invalidates the preceding MTP block when its lookahead consumes the changed token"。
  但没有从 draft 右移 → 内容依赖 → 键覆盖 → 格数收缩的完整叙述。
- 用户三轮追问才得到答案（本笔记的 §4.4 四步对账，源码里不存在对应文字）——这不是读者笨，是文档欠债的直接后果。
- invalidate 的新增域忘登记是静默 0（lookahead 默认 0，忘了写就少裁）——无断言防护。

**工程痕迹（判断"乱打补丁"的反证）**：

- lookahead 机制不是 本 MR 拍的：出身是独立 bugfix commit `lookahead 原始 commit`（"bugfix: include mtp lookahead in prefix cache identities"，cherry-pick 自 上游原始 commit），经 #179（`#179 合入 commit`）进入主线，本 MR 的 base `base-commit` 已包含。
- 该 commit 自述四条对齐项（invalidation / publication cursors / Host-Store restore / decode SWA offload 用同一 identity 规则对齐）并附 450 行专项测试——是有准备的定向修复，不是见一个漏堵一个。
- 本 MR 自己的贡献是把键域**收口**（§1.3 单一事实源、§2 域桶结构），沿用了 lookahead 机制未动其分毫。

### 5.5 裁决

**深思熟虑的工程决策 + 实打实的文档欠债，两笔账分开算。**

- "弯弯绕"的感受是真实的，来源可指认：键、格数、发布、失效四处分散的同源逻辑 + 无串联注释。感受有据，但归因到"炫技/脑子不灵光"不成立——四条可证伪证据（定向 bugfix 的对齐清单、450 行测试、参数化实现、跨项目独立收敛）都指向同一结论。
- 若那条 10 行的机制注释早存在，用户的三轮追问至少两轮不会发生。
- 欠债的利息还在滚：当前 checkout 的上游 main（上游 main HEAD，另一谱系）**没有** MtpBlockHasher，那套实现还是单维桶 + 独立 linear_state_hashes_ 的旧设计——本 MR 的谱系合并回去之前，两套键空间并存的认知成本由所有读者承担。

**本节核心结论**：域间区别对待的分界线是前瞻量参数化；机制是好设计、文档是欠债，"乱打补丁"的指控不成立但"不讲清楚"的指控成立。

---

## 6. MtpBlockHasher 的出身考古（差异登记的事实基础）

本章核验中发现的一处需要与 v1 笔记对齐的事实：

- v1 笔记称"MtpBlockHasher 只在 base commit 存在"。
- 回源码重验：`MtpBlockHasher` 在 base `base-commit` 与 head `commit-4` **都存在**（block_hasher.cpp:31-49，两版完全一致）。
- 它的出身：`lookahead 原始 commit`（原始作者 commit，对应公开合入请求 #179）→ `#179 合入 commit`（#179 合入主线）→ 进入 本 MR base。
- 当前 checkout 的 HEAD（`上游 main HEAD`，上游 xLLM-AI main 谱系）**不含** MtpBlockHasher——那边是旧单维桶 + `linear_state_hashes_` 独立链（正是 本 MR 第四个 commit 杀死的设计）。
- 教学含义：lookahead 机制**早于** 本 MR 存在；本 MR 的增量是域桶结构（block_hashes_by_domain_）与键映射收口（block_utils.h），不是 lookahead 本身。讲"谁设计了什么"时不要记错账主。

---

## 7. 章末自检表

| # | 检查项 | 状态 | 锚点 |
|---|---|---|---|
| 1 | 两把尺子正交、多桶正解（多域并存+切台牌不删桶） | ✅ | §1.2 / §2.2-2.6 |
| 2 | update_block_hashes 全流程贴码、其他桶不动 | ✅ | §2.5（sequence.cpp:997-1011） |
| 3 | hash_tokens 返回什么、MTP 裁草稿 | ✅ | §2.7（sequence.cpp:988-995） |
| 4 | 成员变量 vs 临时变量裁决（问题 5） | ✅ | §2.8（13 读点，v1 记 12 已登记） |
| 5 | invalidate 两调用点 + 逐行六步 | ✅ | §3.2-3.3（sequence.cpp:1013-1027） |
| 6 | token 8 例全程手算 + 覆盖图 + 逐格判断 | ✅ | §3.4（MTP 首脏块 1 / TEXT 首脏块 2） |
| 7 | 失败画面三段叙事（裁少了→命中错块→恢复错位） | ✅ | §3.5 |
| 8 | 键盖定义与三个时机、不盖的失败画面 | ✅ | §3.6（block.h:163-171） |
| 9 | end+1 包装器贴码 + 每格多吃与格数收缩的四步对账 | ✅ | §4.2-4.4 |
| 10 | hash 与内容对应性测试实证（0 命中 / 恰一块） | ✅ | §4.5（test:597/:605） |
| 11 | 问题 7 诚实裁决，三类证据齐、禁夸禁辩护 | ✅ | §5.2-5.5（四行表含反事实 ③④） |
| 12 | MtpBlockHasher 出身考古 + 与 v1 差异登记 | ✅ | §6 |
| 13 | 四把钥匙手算（31/63/30/62、8064→62）回源码重验 | ✅ | §1.4（block_manager_test:787-823） |
| 14 | 每处 file:line 经 `git show commit-4` 回读 | ✅ | 全章；差异登记见下 |

**差异登记（v1 笔记 → 本章回源码核验）**：

1. `block_hashes()` 读点数：v1 记 12 处 → 重数 **13 处**（block_manager_impl 2 / composite 3 / linear_state 4 / sliding_window 2 / hierarchy 2）。
2. "MtpBlockHasher 只在 base commit 存在" → 修正为：base 与 head 都有；**当前 checkout HEAD 没有**（§6）。
3. v1 称 invalidate 调用点为 "commit_token:567" → 源码函数名是 `update_last_step_token`（sequence.cpp:526），invalidate 在 :570；"commit_token" 是 v1 的叙述性称呼，源码无此函数名。
4. v1 "block_hasher.cpp:42-46 end+1" → 精确为 :37-47（compute 体内），`end_token_idx + 1` 在 :44。
5. v1 "block_hasher.cpp:160-165 首格空指针" → 注释 :160-163、代码 :164-169，语义一致。
6. v1 "sequence.h:648-658 / block_utils.h:29-36 / composite:182-186" → 行号全部一致，无差异。

（本章完，计 777 行——冷读验收后修订 +9 行（§0 lookahead 定义、§1.4 tail 来源与减 1 前向指针等）；超出 350-600 预估上限的部分为贴码块、覆盖图与自检表所占行数）


# 第三章 · 账本四动作与双账统一：mount/take/erase/replace 和 tail 便签

> 🔗 **承上启下**：有了边界和键，还需要有人记账。本章讲账本怎么动：mount / take / erase / replace 四动作 + tail 便签，把前两章的数学和键落到运行时的状态迁移——一个序列怎么记账、账本怎么动、账随序列怎么死。

> 源码基准：commit `commit-4`。本章所有 `file:line` 均已用 `git show commit-4:<path>` 逐行回读核验。
> 数值约定：KV 逻辑块 = hash 粒度 = **128 token**；LINEAR 常规检查点步距 chunk_stride = **8192 token**。下标一律 0 基（第 0 号、第 1 号……），凡 1 基语义（"前 k 个"）显式注明。

## 本章导航：你在讨论里卡住的九个问题（计数口径：带问号的问句恰 9 问，另有若干无问号的追问穿插其中）

这一章回答的全部问题，长在同一条主线上：**一个序列怎么记账、账本怎么动、账随序列怎么死**。

```
主线：probe 探到共享块 → 账本怎么记（双账结构）
        │                    │
        ├─ Q"两个独立的账怎么统一用？:414 用哪个账"      → §3.1 双账结构图
        ├─ Q"erase 是把本 type 所有 block 都删了吗？
        │      要么不删要么删完，这对吗"                  → §3.2 四动作 + 分工裁决
        ├─ Q"mount 是干嘛的，take_blocks 干什么的？
        │      用别人的块别人不用了吗？从 host prefetch
        │      来的吗"                                  → §3.3 两步协议
        ├─ Q"mount tail 后游标不清零，就算错了呢，不懂"
        │  Q"没看到 tail 分支判断在哪"                    → §3.4 游标虚高失败画面
        ├─ Q"皮之不存毛将焉附：sequence finish 之后
        │      便签还在吗？块被回收走去哪了"              → §3.5 finish 生命周期
        ├─ Q"probe 证浅的块还回 LRU 吗？放回前头
        │      是不是不公平、假阳性"                      → §3.6 LRU 公平性
        └─ Q"就这样就发布了吗？source 上带 type/hash/index
               吗？都已发布还要 set_num_cached_blocks
               干啥？hashes[hash_index] 从哪放入"         → §3.7 insert 发布协议
```

## 3.0 前置：先认识三个物件

理解账本之前，先分清三个东西。它们分属三层：**物理内存、框架账本、全局登记处**。

**物件一：Block 句柄（物理块的"遥控器"）**。

Block 不是数据本身，是指向物理块的一个句柄，五个字段（block.h:184-199）：

- `id_`：物理块编号（int32，分配器里的槽位号）。
- `size_`：这个块装多少 token（量纲：token/块）。
- `ref_count_`：引用计数，**指向一片原子内存**——同一个物理块的所有 Block 别名共享这一个计数。
- `manager_`：谁分配的它（析构时 ref 减到 0 就还给谁）。
- `hash_value_`：16 字节，前缀缓存的键（§3.7 展开）。

复制一个 Block = 引用计数 +1；销毁一个 Block = 引用计数 −1；减到 0 = 物理块回空闲池。

`is_shared()` 就是问 `ref_count() > 1`（block.h:149）——"除了我还有别人在用吗"。

**物件二：KVCacheState（序列私有的账本）**。

每个 Sequence 挂两本这样的账：`kv_state_`（设备 HBM 侧）和 `host_kv_state_`（host DRAM 侧）。

它记的是"**这个序列**在每一类缓存（BlockType：KV / LINEAR / SWA / C4 / C128……）下拥有哪些块、拥有到多深"。

关键事实先钉死：**账本里没有单块元数据**。想删"某一个块"需要按块的信息（哪个块？还剩谁在用？），而账本只按 BlockType 整类记账——这件事是 §3.2 一切分工的根源。

**物件三：PrefixCache（全局登记处，跨序列存活）**。

`key(16字节hash) → Node{ Block 别名, last_access_time }` 的 map，外加一条 LRU 链。

它不属于任何序列；序列死了它还在。发布（insert）= 往这里登记；命中（find/match）= 从这里取一个 Block 别名走（ref +1）。

**三件物的关系一句话**：物理块被 Block 句柄指着；序列账本记"我持有哪些句柄"；登记处记"哪些句柄可供未来的序列共享"。

## 3.1 账本本体：四张账页 + 一张便签

### 3.1.1 数据结构贴码

先贴账本的成员（sequence_kv_state.h:197-227，每个成员的注释原文保留）：

```cpp
// sequence_kv_state.h:197-227
// KV cache blocks keyed by cache role. ... std::map
// keeps deterministic iteration for reset / dealloc / debugging ...
std::map<BlockType, std::vector<Block>> composite_blocks_;      // :197

// Logical block size is layout metadata, not Block ownership state. ...
std::map<BlockType, size_t> block_sizes_;                       // :203

// shared blocks number per block type.
std::map<BlockType, uint32_t> num_owned_shared_blocks_;         // :222

// Pre-grow cache insert cursor per block type. See num_cached_blocks() above.
std::map<BlockType, size_t> num_cached_blocks_;                 // :225

size_t linear_state_tail_tokens_ = 0;                           // :227
```

四张 map 都以 BlockType 为键——**每类缓存一页账**。逐页过：

- `composite_blocks_`：这页的主体。一个 `vector<Block>`，按逻辑位置排。LINEAR 页的向量按下标 = 检查点位次排（位置 i = 第 i 个检查点，0 基），中间可以是无效占位块（block id = −1）。
- `block_sizes_`：这一页的块多大。注意它存的是"版面规格"不是块的属性——host 下沉后（块实体挪进 host 账本、属另一条时间线，见 §3.3.2 末段）向量里只剩无效占位，块本身读不出 size 了（注释 :199-202 原话）。
- `num_owned_shared_blocks_`：这页里有多少块是**从缓存共享来的**（别人算好、我只是挂载），多少是自己算的。trim（部分删除的真身=三步流水线，§3.2.3 展开）与 exact-repeat pop 都要读它。
- `num_cached_blocks_`：**发布游标**。语义见头文件注释（:105-114）："本类型的前缀缓存游标，从 probe 的实际逻辑触达起步，随新转发的块插入而单调推进"。

第五个成员 `linear_state_tail_tokens_`（:227）不在任何 map 里——它是 LINEAR 页边上一张**便签**，注释（:120-124）原文：

> "A tail checkpoint has a KV-block token boundary, not a regular LINEAR position. … Zero means no tail was retained or published by this tier."

翻译：tail 检查点的边界落在 KV 块粒度（128 的倍数）上，不落在常规检查点位置（8192 的倍数）上。这个数字记录"本层保留/发布过边界在这个 token 深度的 tail"。0 = 没有。

### 3.1.2 双账结构图（本章核心图）

`num_cached_blocks_` 和 `linear_state_tail_tokens_` 是**两本独立的账**。你问"两个独立的账怎么统一用"，先看它们到底差在哪：

| 维度 | 常规账 `num_cached_blocks_[LINEAR]` | tail 账 `linear_state_tail_tokens_` |
|---|---|---|
| 记什么 | 已拥有/已发布**前 k 个常规检查点** | 保留/发布过**一个 tail 状态源**，边界在此深度 |
| 量纲 | 检查点**个数**（1 基计数） | token **深度**（绝对值，128 的倍数） |
| 例值 | 2（= 已覆盖到 16384） | 19968（= tail 站在 16384 与 24576 之间） |
| 谁写 | mount 初设（sequence_kv_state.cpp:284，游标=向量长度，§3.2.4）、发布守卫通过后 `set_num_cached_blocks`（linear:426）、挂载修正（composite:928）、replace 重申报（sequence_kv_state.cpp:299） | 发布（linear:443）、准入挂载补设（composite:929）、预取放回（linear:268） |
| 谁读 | 发布守卫（linear:414）、预取补齐起点（linear:161-163） | 发布幂等守卫（linear:430-431）、预取 tail 短路（linear:151-152） |
| 清零点 | erase:160 级联 / take:181 级联 / replace 重申报 | erase:158 / take:168 / mount:271 / replace:294 / reset:431 |

ASCII 全景（以 prompt=20000、检查点步距 8192、KV 块 128 为例）：

```
token 轴：0 ──── 8192 ──── 16384 ──── 19968 ──── 20000 ────→
                    │          │          │         │
常规检查点：        CP1        CP2                 （无 CP3，20000<24576）
tail 边界：                              TAIL（(20000/128)*128=19968，
                                          且 19968%8192=3584≠0 → 合法 tail）

常规账（个数）：     [CP1✓][CP2✓]           →  num_cached_blocks = 2
tail 账（深度）：                          19968   →  linear_state_tail_tokens = 19968
物理源：            CP1 块     CP2 块      tail 块   ← 三个物理源，各记各的账
```

**成长期两账并存**（序列 A 从 0 算到 20000）：

- 8192 处发布 CP1：常规账 0 → 1；
- 16384 处发布 CP2：常规账 1 → 2；
- 19968 处收尾发布 tail：便签 0 → 19968。

此刻账本同时写着 (常规 2, tail 19968)，对应三份物理源——**这不是矛盾，是两类边界各上各的账**。

**两本账怎么统一用？——答案是：不统一，只在四个汇合点碰头**。我逐个回源码核验过，全库没有任何一行同时写两账；它们只在读侧的四个确定位置相遇：

1. **准入比大小**（composite:607）：probe 同时拿到常规命中深度和 tail 深度，`tail_match->token_count > regular_tokens` 才保留 tail——比的是 probe 结果（两本账的"实物"；probe=用自己的 prompt 算链式 hash 查缓存，机制 §3.3.2 展开），不是直接读账面。
2. **挂载此消彼长**（composite:927-929）：tail 胜出挂载时，游标清 0、便签设深度——**两本账在这里显式换防**（§3.3、§3.4 展开）。
3. **发布各自守卫**：常规分支读常规账（linear:414 `checkpoint_index <= num_cached_blocks`），tail 分支读便签（linear:430-431 幂等检查）。
4. **预取分岔**：进门先看便签（linear:151-152，tail 短路），否则走常规分支用游标当补齐起点（linear:161-163）。

**你问 `if (checkpoint_index <= num_cached_blocks) return;` 用的是哪个账——常规账**，且只可能是常规账。贴码看这段守卫所在的函数 `cache_read_source`（linear_state_block_manager.cpp:397-444）的常规分支：

```cpp
// linear_state_block_manager.cpp:412-427
if (cached_tokens % block_size() == 0) {
  const size_t checkpoint_index = cached_tokens / block_size();
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return;                                     // :414-415 守卫
  }
  seq->update_block_hashes(...);
  const Slice<XXH3Key> hashes = seq->block_hashes();
  const size_t hash_index = linear_state_checkpoint_hash_index(
      cached_tokens, options_.hash_block_size());
  CHECK_LT(hash_index, hashes.size());
  source->back().set_hash_value(hashes[hash_index].data);   // 盖键
  prefix_cache_->insert(
      Slice<Block>(*source).slice(source->size() - 1, source->size()));
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);  // :426
  return;
}
```

三个变量逐个溯源：

- `cached_tokens`：`seq->kv_cache_tokens_num()`（:408），序列已算完的 token 数。
- `checkpoint_index`：`cached_tokens / 8192`——**量纲是检查点序号**，和常规账同量纲，所以可比。
- `num_cached_blocks(LINEAR)`：常规账——"前 k 个检查点已经发布过了"。

守卫的含义：序列算完的边界恰好是第 index 个检查点时，若 index ≤ k（前 k 个已发布），**跳过**；否则发布并把账推进到 index。

**为什么量纲不同注定分账**，正反两个方向：

- 正着塞：tail 深度 19968 ÷ 8192 = 2.44——既不是 2 也不是 3，塞进"检查点个数"账里必然丢精度或错位。
- 反着换：把检查点个数换算成 token 深度记进便签，会跟"便签=一个 tail 源的边界"的语义打架（两个检查点 ≠ 一个 tail）。

**分账不是设计偷懒，是量纲物理上不可通约**——这是合理妥协。欠债在别处：成长期并存无断言，漏一处 set 静默错位；§3.4 的失败画面就是这笔欠债的账单。

## 3.2 四动作机械语义表

四动作都是 `KVCacheState` 的成员函数，都是**纯账本动作**——只动账页和引用计数，不搬运任何数据。总表：

| 动作 | 行号 | 机械语义 | 元数据 | 何时用 |
|---|---|---|---|---|
| `mount_composite_shared` | :265-285 | 把 probe 拿来的共享块**装进空账页** | 全部新写 | 准入挂载（composite:926/:940） |
| `take_blocks` | :166-183 | 把整页**取出搬走**，账页清空 | **全抹** | 部分删除流水线第一步 |
| `erase_blocks` | :156-164 | 把整页**作废** | 全删 | 整页失去存在意义时 |
| `replace_composite_blocks` | :287-300 | 把修剪后的向量**放回**，账目重申报 | 调用方显式给 | 部分删除流水线第三步 |

### 3.2.1 erase：整页作废（你的 Q"要么不删要么删完，这对吗"）

先贴码（sequence_kv_state.cpp:156-164）：

```cpp
// sequence_kv_state.cpp:156-164
void KVCacheState::erase_blocks(BlockType type) {
  if (type == BlockType::LINEAR) {
    linear_state_tail_tokens_ = 0;        // 便签随账页作废
  }
  composite_blocks_.erase(type);          // 块向量整页删除
  block_sizes_.erase(type);               // 四张 map 各删本 type 的条目
  num_owned_shared_blocks_.erase(type);
  num_cached_blocks_.erase(type);
}
```

逐行读：

- 传入 `type`，删的是**这一类的整页**——没有块号参数，想给也没有（§3.0 钉过：账本没存单块信息）。
- 向量删除时，页内每个 Block 别名析构 → 引用计数各 −1。**注意"删账"≠"删块"**：物理块死活由引用计数决定，前缀缓存还持着别名的块照样活（§3.5 展开）。

你的直觉"要么不删要么删完，这么极端对吗"——**对，而且是数据结构的直接后果，是好设计**。裁决链三步：

1. 想支持"只删第 i 块"，账本必须存按块元数据（谁在用、什么键）——这些恰好都下沉在 Block 句柄上（block.h:184-199），账本刻意不重复存。
2. 组级账本 + 单块元数据下沉 Block 的分工，换来的是：reset/析构/调试都是整页遍历，`std::map` 保序（头文件注释 :194-196 原话"deterministic iteration for reset / dealloc / debugging"）。
3. 那"删一部分"怎么办？走**三步流水线**（下一小节）。

erase 生产代码共 8 个调用点（`git grep erase_blocks` 核验；计数口径：仅生产代码，不含测试里的 3 处直调，sequence.cpp:375-378 四行各计 1），每一处都是**整页失去存在意义**：

- fork/beam 构造（sequence.cpp:375-378，4 处）：子序列的 LINEAR/EMBEDDING 槽是私产，不继承（注释 :367-371）。
- KV 预取收尾裁光（block_manager_impl.cpp:339-341）、host 裁光（hierarchy:724-725）、LINEAR trim 无源（linear:301-303）、SWA 页清（sliding_window:245）——各 1 处，共 4 处。
- 共同点：留下空页没有意义，干脆整页撕掉。

欠债一笔：四张 map 靠调用方保持同步，`描述-向量同步` 无断言——erase 若漏删一张，账面与向量错位只能靠下游 CHECK 撞出来。

### 3.2.2 take：整串取出，元数据全抹

```cpp
// sequence_kv_state.cpp:166-183
std::vector<Block> KVCacheState::take_blocks(BlockType type) {
  if (type == BlockType::LINEAR) {
    linear_state_tail_tokens_ = 0;        // 便签先作废
  }
  auto it = composite_blocks_.find(type);
  if (it == composite_blocks_.end()) { ... return {}; }
  std::vector<Block> blocks = std::move(it->second);  // 向量整体搬走，引用不减
  composite_blocks_.erase(it);            // 四张 map 全抹
  block_sizes_.erase(type);
  num_owned_shared_blocks_.erase(type);
  num_cached_blocks_.erase(type);
  return blocks;                          // 责任移交调用方
}
```

关键区别于 erase：**`std::move` 搬走向量，一个引用都不减**。头文件注释（:94-97）写明契约："调用方对返回的别名负责，可以在合并另一层的 probe 后重新 mount"。

take 之后账本是**空白页**——此刻任何读账的人看到的都是"这个序列没有 LINEAR 状态"。真值暂时只活在调用方手里的临时向量里。

### 3.2.3 部分删除三步流水线（trim 的真身）

你说"trim 的时候我怎么看是全部清空的呢"——因为你看的 erase 是流水线的**兜底分支**。单块级删减的真身是三步：

```
① take_blocks(LINEAR)      整串取出（:166），账面清空、便签作废
② vector 修剪               resize 到保留长度 / deallocate 多余别名（引用在此减）
③ replace_composite_blocks 放回 + 重申报 (num_shared, cursor)（:287-300）
```

replace 贴码：

```cpp
// sequence_kv_state.cpp:287-300
void KVCacheState::replace_composite_blocks(BlockType type,
                                            std::vector<Block>&& blocks,
                                            size_t num_shared_blocks,
                                            size_t cache_publish_cursor) {
  CHECK_LE(num_shared_blocks, blocks.size());
  CHECK_LE(cache_publish_cursor, blocks.size());
  if (type == BlockType::LINEAR) {
    linear_state_tail_tokens_ = 0;        // 放回也先作废便签
  }
  remember_block_size(type, blocks);
  composite_blocks_[type] = std::move(blocks);
  num_owned_shared_blocks_[type] = static_cast<uint32_t>(num_shared_blocks);
  num_cached_blocks_[type] = cache_publish_cursor;
}
```

两个申报数字由**掌握语义的调用方**显式给出：`num_shared_blocks`（几块是共享来的）、`cache_publish_cursor`（发布游标设到哪）。replace 自己不猜。

流水线的四个实例（逐一回源码核验）：

1. **KV 预取收尾**（block_manager_impl.cpp:320-341）：take :324 → resize 到 keep → 裁空的走 erase :340，否则 cache+replace。
2. **设备侧 trim**（hierarchy:722-728）：向量搬出 → `trim_blocks_from_back` 从尾部裁 → 裁空 erase :725，否则 replace :727-728。
3. **LINEAR 预取收尾**（linear:253-269）：take :253 → tail 路径 replace(cursor=0)+补设便签 :264-268 / 常规路径裁剪+发布+replace :323-326。
4. **retain_read_source 原地修剪**（linear:329-341）：变体——不动账页只裁向量头部，`set_shared_blocks_num` 单独重申报 :340。

所以分工的正解（你的 Q3 终答）：

- **erase = 整 type 页面级语义**，只服务"整页作废"的场景。
- **单块删减 = take → 修剪 → replace 的三步流水线**，由调用方掌握语义。
- 你看到"trim 全清"是因为撞上了裁空兜底分支（`if (blocks.empty()) erase`），不是 trim 只有全清一档。

### 3.2.4 mount：装共享块进账本（只搬块，不问出处）

```cpp
// sequence_kv_state.cpp:265-285
void KVCacheState::mount_composite_shared(BlockType type,
                                          std::vector<Block>&& shared_blocks) {
  if (shared_blocks.empty()) { return; }
  if (type == BlockType::LINEAR) {
    linear_state_tail_tokens_ = 0;        // ← 便签先清零
  }
  remember_block_size(type, shared_blocks);
  std::vector<Block>& bs = composite_blocks_[type];
  CHECK(bs.empty()) << "mount_composite_shared: existing blocks ...";
  const size_t count = shared_blocks.size();
  bs = std::move(shared_blocks);          // 块装进账页
  num_owned_shared_blocks_[type] = static_cast<uint32_t>(count);
  // The probe vector length is the actual matched logical reach ...
  num_cached_blocks_[type] = count;       // ← 游标 = 向量长度
}
```

机械语义四条：

- `CHECK(bs.empty())`：mount 只装**空白页**（页上有东西说明调用方流程错了，宁可崩不错挂）。
- 便签无条件清零（:270-272）。
- `num_owned_shared_blocks_ = count`：装进来的全部是共享块。
- `num_cached_blocks_ = count`：**游标设成向量长度**——注释 :280-283 解释：probe 向量的逻辑长度就是本次命中的逻辑触达。

## 3.3 mount 的两步协议：为什么"先清零、调用方再补设"

### 3.3.1 你没看懂的那段话，拆开

原文是："mount_composite_shared 只负责搬块，不知道搬的是不是 tail；设值发生在它的调用方。" 拆成三问：

**问一：mount 为什么不知道是不是 tail？**

看 mount 的参数表：`(BlockType, vector<Block>&&)`。它只拿到一堆块。tail 与否的信息在 `ProbeResult.logical_token_count` 里（composite 侧的 probe 结果结构体），**mount 根本收不到这个参数**。同一个 mount 函数服务三种挂载：

- KV/SWA/C4/C128 的普通挂载（composite:940）——游标=向量长度，正确。
- LINEAR 常规检查点挂载——向量是 `[占位…, 最深有效源]`，游标=逻辑触达，正确。
- LINEAR tail 挂载——向量是 `[tail 源]`，游标=1 是**假账**（1 个块 ≠ 拥有第 1 个检查点）。

只有第三种需要修正，而区分信息 mount 看不见。**这就是"只搬块、不问出处"的全部含义**。

**问二：调用方怎么补设？** 贴码（composite_block_manager.cpp:924-930）：

```cpp
// composite_block_manager.cpp:924-930
for (ProbeResult& probe : trimmed.to_mount) {
  if (probe.type == BlockType::LINEAR) {
    kv_state.mount_composite_shared(probe.type, std::move(probe.blocks)); // :926
    if (probe.logical_token_count != 0) {          // :927 tail 哨兵
      kv_state.set_num_cached_blocks(probe.type, 0);               // :928 游标清零
      kv_state.set_linear_state_tail_tokens(probe.logical_token_count); // :929 补设便签
    }
  } ...
```

两步协议：mount 先机械装账（便签清零、游标=1），掌握语义的调用方看到哨兵非零，再把游标改 0、便签写成 tail 深度。

**问三：你问"没看到 tail 分支判断在哪"（10-13 消息问题 2）**——分支就是 `:927` 那行 `if (probe.logical_token_count != 0)`。这个字段是**哨兵**：

- `ProbeResult.logical_token_count` 默认 0；
- 全库唯一非零赋值点在 trim 的 tail 胜出分支（composite:751 `linear->logical_token_count = linear->tail_token_count;`）；
- 挂载侧唯一读点就是 composite:927。

所以"是 tail"的判定**不靠显式 flag，靠数据携带**：非零 logical_token_count + （mount 后）游标归零，组合表达"这页挂的是 tail"。你之前没找到分支，是因为分支条件是一个默认 0 的字段而不是名字里带 tail 的布尔——这是可读性欠债，不是逻辑缺失。

### 3.3.2 mount 是干嘛的？take 是干什么的？场景串一遍

你问"mount 是干嘛的，take_blocks 干什么的，在什么场景用到"。完整旅程（设备侧准入）：

```
① 新序列 B 进来，scheduler 调 allocate_shared
② probe：B 用自己的 prompt 算链式 hash → 到 PrefixCache 查
     ├─ 常规链：逐检查点查（linear_state_prefix_cache.cpp:39-125）
     └─ tail 点查：单键 find（linear:365-383）
③ trim：KV 与 LINEAR 裁到共同可用的边界（composite:676-782）
④ mount：裁完的共享块装进 B 的账本（composite:926）
⑤ B 从恢复边界起 forward 剩余 token
```

mount 是第 ④ 步——**把别人算好的块记到我的账上，从此我可以直接用**。take 出现在需要"先取出再修剪再放回"的地方（§3.2.3 四实例），最常见是预取收尾。

**"mount 是用别人的块，那别人不用了吗？"**——不是。引用计数机制：

- probe 命中 `find`/`match` 返回的是 Block **别名**（拷贝构造，ref +1）。
- 发布者 A 早已 finish、掉掉了自己的别名；此刻持别名的是**缓存登记处**（LRU 节点里的那份）。
- B mount 后，物理块 ref = 缓存 1 份 + B 1 份 = 2。谁都没让出，**共享**而非转让。
- 发布过的块是不可变只读的（状态折叠到边界后的快照），所以两个读者并发读绝对安全。
- 缓存那份别名的意义就是留给未来的 C、D……；LRU 驱逐时才真正放手（§3.6）。

**"是从 host prefetch 过来的吗？"**——不是。准入 mount 的块来自**设备侧 HBM 的 PrefixCache**（probe 直接 find 到）。

host prefetch 是另一条时间线（入队前 host 准入 → H2D 搬运）：那条线的"挂载"发生在 host 账本上，D2H/H2D 细节属于"三层存储与 prefetch"章的地盘。

本章只钉一句：**mount 动作本身与介质无关，HBM 和 host 两本 KVCacheState 用同一套四动作**。

## 3.4 游标虚高的失败画面："就算错了呢，不懂"——现在算给你看

### 3.4.1 正确状态长什么样

prompt=8191、KV 块 128、步距 8192。tail 检查点数学：`(8191/128)*128 = 8064`，且 8064 % 8192 ≠ 0 → 合法 tail，深度 8064。

B 命中 tail 挂载后，**正确**的账面（有 :927-929 修正）：

```
composite_blocks_[LINEAR]   = [tail块]        （向量 1 块）
num_owned_shared_blocks_    = 1               （1 块共享来的）
num_cached_blocks_[LINEAR]  = 0               （游标清零：一个常规检查点都没有）
linear_state_tail_tokens_   = 8064            （便签：tail 边界在此）
```

### 3.4.2 坏掉的字段：假如 :928 不清零

反事实（教学假设，代码里钉死了不会发生——见 3.4.4）：mount 后不修正游标。mount 的机械规则是"游标 = 向量长度"（:284），tail 挂载向量恒为 1 块，于是：

```
num_cached_blocks_[LINEAR]  = 1               （假账：向量 1 块 → 游标报 1）
```

假账的语义读法：**"本序列已拥有第 1 个常规检查点（8192 边界）"**。

但向量 0 号槽里躺的是 **8064 深度的 tail 状态**，不是 8192 边界的检查点。位置被错标了：账面说 8192，实物在 8064，**浅了 128 token**。

### 3.4.3 下游读到什么：三个受害点逐步算

**受害点一：发布守卫跳过真发布**（linear:414）。

守卫挂在 `cache_read_source` 的常规分支上——凡序列在检查点边界完成并走到这个函数（chunked prefill 的分片收尾是典型入口），守卫就用假游标做判断：

```
守卫判断：  checkpoint_index(1) <= num_cached_blocks(1)  →  1 ≤ 1 成立
结果：      return，跳过真发布
账面后果：  第 1 个检查点的真实状态（live 槽算出来的）不进缓存
```

诚实标注边界：decode 侧的检查点保持走另一条路（`preserve_decode_checkpoint_` 分支，linear:497-503，检查点搬进 OffloadBlockPair 由层级池处置），不经这道守卫——所以本受害点是"条件成立即中招"，不是 prompt 8191 一例必现；但下面两个受害点只需要账面被信，必现。

**受害点二：预取补齐起点错位**（linear:161-163）。

host 预取分支 `cursor = min(cached_blocks, target_blocks)`，假游标 1 让它相信"检查点 1 已在手"：

```
cursor           = min(1, target) = 1
补齐循环起点     = index 1          ← 第 0 号单元（检查点 1）不去 Store 取
源块落位         = blocks[cursor-1] = blocks[0]   （linear:203）
```

0 号槽里的 tail 源（8064）被当作检查点 1（8192）占据名分。

**受害点三：恢复边界错位，复利放大**。

未来任何一次从这页账恢复（decode 实例读源、restore 选择），按账面语义取"检查点 1 = 8192 边界状态"，实际拿到 8064 的折叠状态：

```
账面边界：  1 × 8192          = 8192
实际内容：  tail 深度          = 8064
错位：      8192 − 8064       = 128 token 的折叠贡献凭空缺失
后果：      LINEAR 递归状态每一步 decode 都在少算 128 token 贡献的起点上叠加
            ——不崩溃、不报错，长输出质量静默劣化
```

这就是"就算错了呢"的完整答案：**错在账面与实物的边界错位，毒性是复利式的静默错算，不是当场崩溃**。

### 3.4.4 通式与 v1 笔记数字的回验（差异登记）

把假游标一般化为 k（向量长度 k 的挂载），错位量的通式：

```
账面边界  = k × 8192
实际内容  = tail 深度
错位      = k × 8192 − tail
```

v1 事实底料写"k=3 时账面 24576 vs 实际 8064，错位 16448"。**回源码重验的结果：k=3 的前提在现行代码构造不出来**：

- 设备准入的 tail 胜出分支，trim 会把常规命中块全部丢进 `to_drop`（composite:743-748），向量只剩追加的 tail 源 1 块（composite:749）；
- host 预取的 tail 放回路径，`CHECK_EQ(blocks.size(), 1u)`（linear:260）**用断言钉死了向量恒为 1 块**。

所以现行代码里假游标只可能是 k=1，错位恒为 8192 − 8064 = **128 token**（对 prompt 8191 的例）。

v1 的 16448 是把通式代入 k=3 的演示数字，其向量前提（tail 挂载带 3 块）已回验不成立——**登记为差异，以本节 k=1 的推导为准**。

通式本身保留，因为它解释了"为什么虚高越多毒性越大"。

另有一层保险值得记：linear:259-260 的两个 CHECK（`shared_blocks == 1`、`blocks.size() == 1`）说明作者**知道**这条不变量（tail 挂载向量恒 1 块）并选择用断言钉死，而不是靠文档。

### 3.4.5 "固定 prompt 下 tail 挂载后常规账再动"构造不出来

有人会追问：B 挂了 tail 后，游标归 0；B 继续把 prompt 算完，会不会又跨过检查点、让两本账重新纠缠？三步数学证明（固定 prompt，prefill 范围内）：

1. tail 深度 d 必是 128 的倍数（KV 块边界 floor 出来的），且 d % 8192 ≠ 0（tail 排除规则二），所以 d 到下一个检查点的距离 ≥ 128。
2. B 挂载后剩余待算 r = prompt − d ≤ 127（floor 到 128 粒度损失的余量）。
3. r ≤ 127 < 128 ≤ 距离 → B 在 prompt 内永远够不到下一个检查点 → 常规账冻结在挂载值。

两本账在挂载点换防之后，prefill 范围内互不打扰。decode 续写跨检查点属"发布与准入"章的地盘。

## 3.5 sequence finish 的完整生命周期："皮之不存毛将焉附"

### 3.5.1 先纠正一个前提：finish 不走 erase_blocks

你引用的旧书写着"第 3 步 序列结束，块被回收，走 erase_blocks(LINEAR)"——**这步写错了**（历代书已勘误）。finish 的正路是 `HierarchyBlockManagerPool::deallocate`，全程没有 erase_blocks。贴码（hierarchy_block_manager_pool.cpp:491-510）：

```cpp
// hierarchy_block_manager_pool.cpp:491-510
void HierarchyBlockManagerPool::deallocate(Sequence* sequence) {
  ...
  composite->cache_full_blocks_for_sequence(sequence);   // ① 最后一搏发布
  collect_offload_pairs(sequence);                        // ② D2H 配对下沉
  host_manager->deallocate_for_sequence(sequence,
      sequence->host_kv_state());                         // ③ host 账本掉引用
  composite->deallocate_for_sequence(sequence);           // ④ 设备块掉引用
  sequence->reset();                                      // ⑤ 只抹账面
}
```

五步逐个说清：

- **①**：finish 前把已完成到可复用边界的块尽量发布进缓存（"最后一搏"）。
- **②**：设备上已完成的块和 host 新分配的空目的块配成 `OffloadBlockPair` 进异步 D2H 队列——**配对是 move 搬家，零减引用**；减引用只发生在传输完成回调。
- **③**：host 侧各叶对序列 host 账本里的块掉引用。
- **④**：设备侧，`CompositeBlockManager::deallocate_for_sequence`（composite:538-549）先 `cache_for_sequence` 再逐叶 `leaf->deallocate(kv_state.blocks(type))`——"ref 2→1"就发生在这里。
- **⑤**：`Sequence::reset()`（sequence.cpp:962-969）→ 两个 `KVCacheState::reset()`（sequence_kv_state.cpp:430-446）——**只抹账面**，块别名在第 ③④ 步已经掉完引用。

`KVCacheState::reset()` 贴开头三行：

```cpp
// sequence_kv_state.cpp:430-434
void KVCacheState::reset() {
  linear_state_tail_tokens_ = 0;    // :431 便签归零
  kv_cache_tokens_num_ = 0;
  prefix_cache_matched_ = false;
  ...                                // 四张 map 全 clear
}
```

### 3.5.2 正解：便签随账页死，发布出去的状态在缓存里另有实体

你的问题原文："linear_state_tail_tokens_ 是属于 KVCacheState 的，但 KVCacheState 是跟 sequence 绑定的吧，sequence finish 之后呢？皮之不存毛将焉附？"

答案分两半，对应**两个不同的东西**：

- **便签（账面数字）**：序列私有，finish 第 ⑤ 步随 `reset()` 归零。它死了，无足轻重——它只是"本序列此刻挂着/发布过 tail"的**记账**。
- **tail 状态（物理实体）**：早在发布那一刻（linear:441-443 insert），实体就**锚定在全局登记处**——登记的是别名，不是搬数据（与 §3.7.1"零数据搬运"同一口径）：`cached_blocks_` map 里一条 `key → Node{Block 别名}`，LRU 链上一个节点。**它不依附任何序列，依附缓存**。

"皮"和"毛"的依附关系恰好和你的直觉相反：便签是毛，缓存条目才是皮——**毛掉了（reset）皮还在（缓存条目）**。

未来序列 B 怎么用上它，三步：

- B 用自己的 prompt 重算链式 hash（确定性：同样的 token 算出同样的 16 字节键），算出 tail 键；
- `find` 命中 → 拿走 Block 别名（ref +1）→ mount；
- B 从头到尾不需要读 A 的便签。

**账本死了，键还活着，这就是发布的全部意义**。

### 3.5.3 B-Lifecycle 手算：块 77 的引用轨迹

设定：prompt=8191，A 的 tail 块 id=77，chunk 划分 {8064, 127}。场景假设（钉死 ref 轨迹的前提）：本例只画设备侧 HBM 时间线——A 没走 host 准入，host 账本没有恢复目的槽，finish 第②步 D2H 配对（两条分支都要求 host 目的槽就绪）不成立，块 77 不被配对，走"留缓存"结局（§3.5.4）。逐步（每步一行）：

```
T1  A 发布块 77：insert 登记缓存 + A 向量持别名
     → ref = 2（缓存 1 + A 1）
T2  A 继续算完剩余 127 token，decode……
T3  A finish：第④步 deallocate 掉设备别名
     → ref = 2 → 1（只剩缓存持）
     同时第⑤步 reset：A 的便签 8064 → 0（账面死亡）
T4  序列 B（同 prompt）probe：算出同一 tail 键 → find 命中
     → ref = 1 → 2（B 拿走别名）
T5  B mount + :929 补设便签 8064
     → B 账本：向量 [77]，游标 0，便签 8064
T6  B 只需 forward 8191 − 8064 = 127 token
```

消费侧兜底（为什么 T6 不会是 0）：`num_need_compute_tokens() = num_tokens_ − kv_cache_tokens_num()`（sequence.h:210-211），恢复深度若等于 prompt 则待算 0、无法组 batch——而 tail 数学保证有效 tail ≤ prompt − 1，恒有 token 可算。

### 3.5.4 块的三结局："被回收走的块去哪了"

你问"erase_blocks(LINEAR) 这些块干什么去了，去 host 了还是被覆盖了，被覆盖了就用不成了吧"。块在 finish 时刻的分岔（引用计数说了算）：

| 结局 | 条件 | 之后 |
|---|---|---|
| **留缓存** | 发布过（ref ≥ 1：缓存持） | 活到 LRU 驱逐那天；驱逐 = 抹键出 map，此后才可能被覆盖 |
| **回池待覆盖** | 从未发布，只有序列持（ref 1→0） | `free()` 回空闲池，下次 allocate 拿去写新数据 |
| **D2H 下沉** | 跨检查点的源块被 ② 配对 | 传输完成后 host 目的块发布进 host 缓存（第二次生命），设备源块才释放 |

"被覆盖了就用不成了吧"——对，但覆盖**只会发生在归零回池之后**；只要缓存还持着别名（键还在 map 里），物理块就不可能被分配走。键与块共存亡：驱逐抹键 → 引用归零 → 回池 → 才轮得到覆盖。这个顺序保证了"能查到键 = 数据一定还在"。

## 3.6 release_probes 与 LRU 公平性："假阳性白赚头部位置？"

### 3.6.1 probe lambda 贴码（你引的那段）

```cpp
// linear_state_prefix_cache.cpp:62-72
auto probe = [&](const XXH3Key& key) -> bool {
  auto iter = cached_blocks_.find(key);
  if (iter == cached_blocks_.end()) {
    blocks.emplace_back();              // miss：无效占位块，继续走（gap-tolerant）
    return false;
  }
  blocks.emplace_back(iter->second->block);   // hit：拿走共享块别名
  lru_lst_.remove_node(iter->second);         // 从 LRU 摘出（临时）
  node_list.push_front(iter->second);         // 挂到本地工作链
  return true;
};
```

match 收尾（:111-114）：

```cpp
// linear_state_prefix_cache.cpp:111-114
while (!node_list.is_empty()) {
  Node* node = node_list.pop_front();
  lru_lst_.push_back(node);            // 统一放回 LRU 尾部
}
```

先钉死方向（你的原话"放回前头的位置吧"——**方向反了**）：

- 这条 `DNodeList` 里 **头部 = 最旧**，**尾部 = 最新**。
- 证据一：驱逐从 `get_first()`（头部）开始扫（prefix_cache.cpp:243）。
- 证据二：所有刷新都是 `push_back` / `move_back`（:92/:174/:229）。
- 所以命中块被放到**最新端 = 受保护端**，不是"前头等死"。

### 3.6.2 四节点手算

LRU 现有 4 个节点，左=最旧、右=最新：

```
probe 前：      [N1, N2, N3, N4]
请求逐键探测：  N1 命中 → 摘出；N2 命中 → 摘出；N3/N4 未被探测
探测期间：      LRU = [N3, N4]，本地链 node_list = [N2, N1]（push_front，后进在前）
match 收尾：    pop_front 逐个 push_back（LIFO：先 N2 后 N1）
probe 后：      [N3, N4, N2, N1]      ← N1/N2 跳到最新端
trim 砍掉检查点 2（release_probes 退还别名）：LRU 纹丝不动
```

（v1 底料把收尾写成 [N3,N4,N1,N2]，漏算了 node_list 的 LIFO 顺序——`push_front` 进、`pop_front` 出，后命中的先回去。集合 {N1,N2} 落在最新端这一结论不变，登记为差异 #7。）

最后一行是关键：**release_probes 只掉引用，不动 LRU**。贴码（composite_block_manager.cpp:621-640）：

```cpp
// composite_block_manager.cpp:627-637（节选）
for (ProbeResult& probe : *probes) {
  if (!probe.blocks.empty()) {
    probe.leaf->deallocate(probe.blocks);   // 只减引用计数
    probe.blocks.clear();
  }
  if (probe.tail_block.has_value()) { ...同上... }
}
```

调用时序（composite:901-916）：probe → trim → `release_probes(&trimmed.to_drop)` → mount。退还发生在 trim 判决**之后**，退还的块此刻早已被 probe 刷新过 LRU 位置了。

### 3.6.3 诚实裁决：不公平吗？假阳性吗？

三问三答，全部对着源码说：

**问一：被 trim 丢弃的"浅命中"块是假阳性吗？**——不是。键是真的（链式 hash 如实算出）、内容是真的（确实是一个合法检查点/tail 状态）。它被丢是因为**跨叶对齐**：KV 侧只能对齐到更浅的边界，或 tail 胜出后常规块让位——"队友拖后腿"，不是这块自己说谎。

**问二：探测了却没用上的块被刷到最新端，公平吗？**——这是标准的"访问即 touch"语义：探测本身证明了**最近有真实前缀需求**摸过这个键。

- 行业同类做法一致（vLLM 命中即刷新 ⚠️证据边界：vLLM 侧细节为推演，未逐行核验）。
- 反事实——probe 不刷新——会让正被高频探测的前缀和无人问津的前缀同速老化，那才是不公平。

**问三：有没有真浪费？**——有，但要算对账：

- 浪费的是**浅命中块的 LRU 位置被抬高**（多活一段）——轻度过度保留，有界（一次刷新）、无正确性影响。
- 真正的浪费根源在跨叶对齐丢块，不在 LRU。
- 另有真保护在别处：驱逐扫描跳过 `is_shared()`（ref > 1）的节点（prefix_cache.cpp:249-253）——**正在被序列挂载使用的块根本驱逐不掉**，这是正确性保证而非优化。

## 3.7 insert 发布协议："就这样就发布了吗？"

### 3.7.1 发布到底发布了什么

你问"source 上会带 type、hash、index 这些内容吗"。先看 Block 到底携带什么（§3.0 已列五字段）：`id / size / ref_count / manager / hash_value_(16B)`。**不带 BlockType，不带逻辑下标**。逐个解释为什么不需要带：

- **type 不带**：每个叶子（KV 叶、LINEAR 叶）各有一个**独立的 PrefixCache 实例**（`LinearStatePrefixCache` 与 `PrefixCache` 是两个类，各自 `cached_blocks_`）——键空间天然按类型隔离，登记时不需要自报类型。
- **index（深度）不带**：深度编码在键本身里。链式 hash 的第 N 节代表"前 (N+1)×128 个 token"的前缀摘要——**键即深度**，查到键就知道了覆盖范围，不需要额外字段。
- **hash 带**：就是 `hash_value_` 那 16 字节。LINEAR 发布前调用方先 `set_hash_value` 盖键（§3.1.2 贴码 :423），把键**写进块的行李**。

发布动作本身（linear:424-425）：

```cpp
prefix_cache_->insert(
    Slice<Block>(*source).slice(source->size() - 1, source->size()));
```

只传**最后一块**的切片。insert 的已盖键重载（prefix_cache.cpp:185-222）逐块做三件事：

1. 从块上读键（`get_immutable_hash_value`，:194）；
2. 查 map：键已存在 → 只刷新时间戳 + LRU（:196-201，**幂等，不重复建节点**）；不存在 → 新建 Node、`cached_blocks_.emplace`（:203-212）；
3. Node 里存一份 Block 别名（ref +1）。

**发布 = 登记一条 map 记录 + 引用计数 +1，零数据搬运**。物理数据从头到尾没挪过窝。

### 3.7.2 "都已发布了，set_num_cached_blocks 还有啥用"

你的质疑有道理一半：**缓存确实不需要这个数字**——map 里有没有这个键，跟序列账本记到几没关系。

但游标不是给缓存的，是给**序列自己**的——它是账本侧的独立记录，两个真实消费者：

1. **发布守卫的幂等依据**（linear:414）：`checkpoint_index <= num_cached_blocks` → 跳过。没有它，每次 chunk 边界都要重算 hash、重查 map、重走 insert——功能上多数情形幂等（map 已存在只刷新），但游标让"已发布到哪"成为 O(1) 的账面事实。
2. **预取补齐起点**（linear:163）：`cursor = min(cached_blocks, target_blocks)`——host 预取从游标处开始补缺，已发布的单元不去 Store 重复取。

tail 分支同理：便签的消费者是 `:430-431` 的幂等守卫 `kv_state.linear_state_tail_tokens() == cached_tokens → return`——头文件注释 :122-124 原话："During growth, this boundary also prevents republishing the same completion from the newly allocated live slot"（增长期防止新分配的活槽把同一完成重复发布）。

一句话：**insert 管"缓存知道了"，游标/便签管"序列自己记得发到哪了"**——两套记录服务两套决策，删掉后者不丢数据，丢的是发布侧的幂等与预取侧的起点。

### 3.7.3 "hashes[hash_index] 的内容从哪儿放入、怎么保证取的时候有"

链式 hash 数组的填充与保证，机制属"键空间"章（hash 链与域分桶）的地盘，这里钉三件事实：

1. **从哪放入**：`update_block_hashes` → `extend_prefix_hashes` 增量接龙——第 i 格 = hash(第 i 块 128 token + 第 i−1 格摘要)，首格父摘要为空指针标记。每算完 128 token 接一节。
2. **取之前的保证**：LINEAR 每个取键点都**先 update 再读**——`tail_checkpoint_hash` 里 `update_block_hashes` 在前（linear:357），读在后（:358-362）；不足时返回 `nullopt` 优雅放弃（:359-361），常规分支则 `CHECK_LT(hash_index, hashes.size())` 硬断言（:422）。
3. **为什么"恰好有"**：tail 边界 8064 是 128 的倍数（tail 数学用 KV 块边界 floor 出来），第 8064/128−1 = **62** 格（共 63 格，0 基）必然已在链上——边界落在块边界上，覆盖该边界的格必然已接龙完成。这是 tail 数学（"tail 检查点数学"章）与 hash 链（"键空间"章）的咬合点。

测试锚点（回读核验）：`block_manager_test.cpp:787-824`（LargeChunkTailUsesKvHashChainAndLogicalCheckpointCursor）——prompt 8191、hash_bs 128，断言 `shared_blocks_num(KV) == 63u`（:819，63 是**块数**，键下标最大 62——0 基/1 基双写纪律）、`num_cached_blocks(LINEAR) == 0u`（:821，tail 挂载游标清零）、`kv_cache_tokens_num() == 8064u`（:822）。

## 3.8 章末自检表

| # | 自检问题 | 答案锚点 |
|---|---|---|
| 1 | 账本有哪几页？单块元数据存在哪？ | 四 map（:197/:203/:222/:225）+ 便签（:227）；单块元数据在 Block 句柄（block.h:184-199） |
| 2 | 两本账各记什么、量纲差在哪？ | 常规账=检查点个数（1 基）；tail 账=token 深度；§3.1.2 双账结构图 |
| 3 | `checkpoint_index <= num_cached_blocks` 用哪本账？ | 常规账，量纲同构才可比（linear:412-414） |
| 4 | 两本账在哪四个点汇合？ | 准入比大小 / 挂载换防 / 发布各自守卫 / 预取分岔（§3.1.2） |
| 5 | erase 为什么只能整页删？ | 账本无单块元数据；单块删减走 take→修剪→replace（§3.2.3 四实例） |
| 6 | mount 为什么"先清零、调用方补设"？ | mount 收不到 tail 哨兵（ProbeResult.logical_token_count）；composite:926-929 两步协议 |
| 7 | tail 分支判断在哪？ | composite:927 哨兵非零；唯一赋值点 composite:751 |
| 8 | 游标虚高 k 错在哪、毒性多大？ | 账面 k×8192 vs 实物 tail；现行代码 k 恒为 1（linear:260 CHECK 钉死），错位 128；复利静默错算（§3.4） |
| 9 | finish 走 erase_blocks 吗？五步是什么？ | 不走；发布→D2H 配对→host 掉引用→设备掉引用→reset 只抹账面（hierarchy:491-510） |
| 10 | 便签死了 tail 还能用吗？ | 能：发布时实体已进缓存（key→Node），B 重算键 find 命中（§3.5.2-3.5.3） |
| 11 | 块三结局？ | 留缓存 / 回池待覆盖 / D2H 下沉（§3.5.4） |
| 12 | 浅命中块回 LRU 哪端？公平吗？ | 最新端（push_back）；非假阳性，touch 语义行业常规，浪费根源在跨叶对齐（§3.6.3） |
| 13 | insert 带不带 type/index？ | 都不带：每叶独立缓存隔离 type，深度编在链式键里（§3.7.1） |
| 14 | 发布后还要 set_num_cached_blocks 干啥？ | 序列侧幂等依据 + 预取起点，缓存不需要它（§3.7.2） |

## 3.9 核验声明与差异登记

**核验声明**：本章全部 `file:line` 引用均以 `git show commit-4:<path>` 逐行回读核验，关键贴码（erase/take/mount/replace、发布守卫 :412-427、tail 发布 :430-443、挂载修正 :924-930、release_probes :621-640、probe lambda :62-72、finish 五步 :491-510、reset :430-434）与现行源码逐字一致；测试锚点（block_manager_test.cpp:763-784 纯 tail、:787-824 大块 tail、:826-859 TailMismatch 回退）断言值已回读核对。手算数字（8064/63 块/62 下标/128 错位/B-Lifecycle 引用轨迹/LRU 四节点）均可由贴出的代码与参数复算。

**差异登记**（v1 事实底料 vs 本章回源码重验）：

| # | v1 底料说法 | 回验结果 | 处置 |
|---|---|---|---|
| 1 | 游标虚高"k=3 时账面 24576 vs 实际 8064，错位 16448" | tail 挂载向量恒 1 块（composite:743-749 丢常规别名；linear:260 `CHECK_EQ(blocks.size(),1u)` 钉死）→ k 恒为 1，错位恒 128 | 以 §3.4.4 重推导为准；16448 降级为通式代入示例，前提不成立 |
| 2 | 成长期并存"（常规 3，tail 19968）" | prompt 20000 只有 2 个完整检查点（8192/16384），游标最大 2；"3"应为三个物理源（cp1/cp2/tail）之误 | §3.1.2 改写为（常规 2，tail 19968，物理源 3 份） |
| 3 | 清零点"erase:158/take:168/mount:271/replace:294/reset:431" | 逐行回读全部一致 | 维持 |
| 4 | 四 map 行号"191-227"（区间） | 精确行：197/203/222/225/227 | 本章用精确行号 |
| 5 | "block_manager_test.cpp:788" | 测试起于 :787-788，断言在 :819-822 | 本章引 :787-824 |
| 6 | finish 前提"走 erase_blocks(LINEAR)"（用户转述旧书） | finish 走五步 deallocate，无 erase_blocks；erase 调用方仅 fork/trim 裁光等 8 处（生产代码，不含测试） | §3.5.1 显式纠正 |
| 7 | LRU 四节点手算收尾"[N3,N4,N1,N2]" | node_list 为 LIFO（push_front 进、pop_front 出，:70/:111-113），收尾实为 [N3,N4,N2,N1] | §3.6.2 已改；"浅命中块落到最新端"结论不变 |


# 第四章 · 准入 probe 与发布流水线：gap-tolerant 查找、按块回退与发布守卫

> 🔗 **承上启下**：账本之上是准入：新请求来了怎么查（gap-tolerant，容忍中间的洞）、查不到怎么按块回退、查到了怎么发布（发布守卫）。**前缀复用真正发生的时刻在第四章**——前面三章的机制都是为这一刻服务的。

> 本章对应 factbase §五（发布与准入）。核验基准：head `commit-4`，所有代码引用均已用 `git show commit-4:<path>` 逐行回读。
> 数值约定：KV 逻辑块 = hash 粒度 = **128** token；LINEAR 检查点步距 chunk_stride = **8192** token；hash 下标 0 基（8064/128−1 = **62**，链上共 63 格，下标 0..62）。

## 0. 本章地图与前置速览

本章回答五个长在「一条准入生命线」上的问题：

```
新请求到达
   │
   ├─ ① probe 两问（问一常规链 gap-tolerant 逐块 / 问二 tail 单键点查）   ← 用户"两方案"裁决
   │
   ├─ ② tail_checkpoint_hash 取键（键从哪来、何时写入、为何取时有货）
   │
   ├─ ③ trim 决策（tail 优先 / 常规按块回退 / max_hit<tail 保不住）
   │
   ├─ ④ mount + base_tokens（游标清零补设 / host 命中深度 :1039）
   │
   └─ ⑤ 发布守卫（幂等防重发 / 盖键 / 单块 insert 协议）——上一个请求在另一头干的事
```

probe 与发布是同一条流水线的两端：**上一个请求"发布"把键和块登记进缓存，下一个请求"probe"拿自己算出的键去查**。

前置概念（首现即教，已读前三章可跳过）：

- **probe（探测）**：新序列在分配块之前，先拿自己的 token 算出的哈希键去缓存里查一遍，看有多少前缀可以白拿（`probe_prefix_cache`，composite_block_manager.cpp:560）。
- **tail 检查点**：LINEAR 状态除了每 8192 token 一个的常规检查点外，额外钉在"最后一个完整 KV 块边界"上的检查点（`linear_state_tail_checkpoint_tokens`，scheduler_config.h:104-118）。
- **mount（挂载）**：把 probe 命中的共享块装进本序列的账本（`mount_composite_shared`，sequence_kv_state.cpp:265）。
- **游标 `num_cached_blocks(LINEAR)`**：账本字段，量纲 = 检查点个数（1 基），语义是"本序列已拥有前 k 个常规检查点"。
- **tail 便签 `linear_state_tail_tokens`**：账本字段，量纲 = token 深度，语义是"本序列挂载/发布过 tail，边界在这个数"。

---

## 1. 用户"两方案"的正面裁决：probe 是两问并列，不是回退链

### 1.1 用户原话重述

> "probe 是让新的 sequence 去算一个自己的 tail，然后去找 prefix cache 中有没有这样的 tail，如果没有的话，那就按 chunk 找吗，还是说按 block 回退算 hash 找 match，直到最近的 chunk 边界？前者的好处是减少无效探索，后者的好处是增加 tail 的 match 可能性"

用户提出了两个候选方案：

- **方案一**：先算自己的 tail 去点查；没有就退到按 chunk（检查点）找。
- **方案二**：按 block（128 token 一格）逐块回退算 hash 找 match，直到最近的 chunk 边界。

### 1.2 实际实现：两问并列，各查各的

裁决：**方案一的方向正确，但结构不是"没有才退"，而是两问并列同时问**；方案二的"按 block 回退"不在 probe 里，在 trim 里。

实际实现的 probe 入口（`LinearStateBlockManager::allocate_shared`，linear_state_block_manager.cpp:575-585）调用 `prefix_cache_->match(token_ids.slice(0, token_ids.size() - 1), ...)`。注意两件事：

- 传入的 token 片被截掉最后一个（`slice(0, size - 1)`，:580）——给 forward 留至少 1 个 token（口径差详见 §2）。
- `prefix_cache_` 这里是 `LinearStatePrefixCache`，它的 match 是 **gap-tolerant** 的。

这是**问一：常规链逐块探测**，核心循环（linear_state_prefix_cache.cpp:62-84）：

```cpp
// linear_state_prefix_cache.cpp:61-84
  // Hit: emplace + LRU-touch. Miss: emplace invalid placeholder.
  auto probe = [&](const XXH3Key& key) -> bool {
    auto iter = cached_blocks_.find(key);
    if (iter == cached_blocks_.end()) {
      blocks.emplace_back();          // ← miss：放一个「无效占位块」
      return false;
    }
    blocks.emplace_back(iter->second->block);  // ← hit：拿出共享块
    lru_lst_.remove_node(iter->second);
    node_list.push_front(iter->second);
    return true;
  };

  size_t last_hit_idx = 0;
  bool has_any_hit = false;
  ...
    for (size_t checkpoint = start_block; checkpoint < n_checkpoints;
         ++checkpoint) {
      if (probe(block_hashes[hash_block_index(checkpoint)])) {
        last_hit_idx = checkpoint;    // ← 记录最深命中
        has_any_hit = true;
      }
    }
```

循环里两个下标变量先给出读法：`checkpoint` 0 基计数（checkpoint=0 即第 1 个检查点，边界 (checkpoint+1)×8192 token）；`start_block` = 已有共享块数（直通搬入结果向量、不重查，探测只走新增段）。

`hash_block_index(checkpoint)` 是检查点→hash 链下标的映射：**边界 token / 128 − 1**——检查点 1（8192 token）→ hashes[63]，检查点 2（16384 token）→ hashes[127]（公式与对齐 CHECK 的贴码见 §3.2）。

逐行读出三个事实：

- 循环体里 **miss 不 break**——`probe` 返回 false 后循环继续走下一个检查点。
- miss 的位置放**无效占位块**（`blocks.emplace_back()`，默认构造的 Block `is_valid()` 为 false）。
- 只记**最深命中** `last_hit_idx`，收尾时 `blocks.resize(last_hit_idx + 1)` 截到最深命中（:107-109）。

**问二：tail 单键点查**（`allocate_tail_shared`，linear_state_block_manager.cpp:365-383）：

```cpp
// linear_state_block_manager.cpp:365-383
std::optional<BlockManager::TailPrefixCacheMatch>
LinearStateBlockManager::allocate_tail_shared(Sequence* seq) {
  const size_t token_count = tail_checkpoint_tokens(seq);
  if (token_count == 0 || prefix_cache_ == nullptr) {
    return std::nullopt;
  }
  const std::optional<XXH3Key> hash = tail_checkpoint_hash(seq, token_count);
  if (!hash.has_value()) {
    return std::nullopt;
  }
  Block block = prefix_cache_->find(*hash);   // ← 一个键，一次 find
  if (!block.is_valid()) {
    return std::nullopt;
  }
  ...
  return TailPrefixCacheMatch{std::move(block), token_count};
}
```

零探索：算出一个 tail 边界 → 算出一个键 → `find` 一次，结束。

两问在 `probe_prefix_cache` 里**并列触发、互不依赖**（composite_block_manager.cpp:602-614）：

```cpp
// composite_block_manager.cpp:602-614
    if (type == BlockType::LINEAR) {
      std::optional<BlockManager::TailPrefixCacheMatch> tail_match =
          leaf.allocate_tail_shared(seq);          // ← 问二总被问，不依赖问一结果
      if (tail_match.has_value()) {
        const size_t regular_tokens = probe.blocks.size() * probe.block_size;
        if (tail_match->token_count > regular_tokens) {
          probe.tail_block.emplace(std::move(tail_match->block));  // tail 更深才保留
          probe.tail_token_count = tail_match->token_count;
        } else {
          std::vector<Block> unused_tail{std::move(tail_match->block)};
          leaf.deallocate(unused_tail);            // ← tail 不比常规深，当场退还
        }
      }
    }
```

### 1.3 为什么 LINEAR 链可以 gap-tolerant，KV 链却 miss 即停

对照 KV 侧的 `PrefixCache::match`（prefix_cache.cpp:64-69）：

```cpp
// prefix_cache.cpp:64-69
  if (block_hashes.size() >= n_blocks) {
    for (size_t b = start_block; b < n_blocks; ++b) {
      if (!match_block(block_hashes[b])) {
        break;                      // ← KV：miss 即停
      }
    }
  }
```

两条链的探测策略不同，物理原因是**复用的东西自不自包含**：

- KV 块链是**连续依赖**的：注意力计算需要前缀里**每一个** token 的 KV。第 5 块命中而第 3 块缺失，第 4、5 块照样不能用——继续深探没有意义，miss 即停。
- LINEAR 检查点是**自包含快照**：`S_16384 = f(t_0..t_16383)` 整段折叠，检查点 2 的物理状态不依赖检查点 1 是否在缓存里。中间 miss 一个，更深的检查点照样可用——所以值得跳过 miss 继续走。

这正是头注释"deeper hit subsumes earlier ones"（更深命中吞没更浅命中）的物理基础。

### 1.4 用户的利弊分析对在哪、实际取舍是什么

用户说"前者减少无效探索，后者增加 match 可能性"。逐半边裁决：

- **"减少无效探索"——对，且实现真的省**：tail 点查只算一个键、查一次表（linear:365-383 全程无循环）。
- **"按 block 回退增加 tail match 可能性"——tail 侧没有这个空间**：tail 边界是 prompt 的确定函数，**候选 tail 深度只有一个**——别的深度对应别的 prompt，键根本不同，不存在"回退一格试试别的 tail"。
- **"增加 match 可能性"的真实兑现处在问一**：gap-tolerant 让常规链 miss 一个检查点后继续深探（§1.3 的自包含性买单），确实提高常规命中概率。
- **方案二"按 block 回退"的真实住址在 trim**：probe 结束后，`trim_flat_kv_linear` 的常规回退分支每次减 1 个 KV 块（128 token）向下让步，直到踩在有效的 LINEAR 检查点上（composite_block_manager.cpp:700-710，详见 §5.1）。
- **回退发生在"对齐两边深度"的裁剪阶段，不在"查找缓存"的探测阶段**：查找阶段 LINEAR 检查点互相独立，无需回退；对齐阶段 KV 与 LINEAR 必须停在同一 token 边界，才需要逐块让步。

一句话总结取舍：**查找侧为自包含性付 gap-tolerant 的账（占位块 + 记最深命中），对齐侧为边界一致性付按块回退的账；tail 因候选唯一而享受零探索的点查。**

附带一笔：gap-tolerant 的一个副产品——浅命中块也会被 LRU 刷新到最新端（probe lambda 里 hit 即 `remove_node` + `push_front`，linear_state_prefix_cache.cpp:69-70），被 trim 丢掉的浅命中块等于"探测性保鲜"。

这是行业常规的合理妥协（vLLM `get_cached_block` 同样刷新 [待验证]），不是假阳性——键真、内容真，被丢是跨叶对齐的"队友拖后腿"。

---

## 2. 探测口径差：常规链截 1 token，tail 点查不截

### 2.1 用户原话重述

> "你在查常规检查点的时候退了一个 token（`token_ids.slice(0, token_ids.size() - 1)`），但在查 tail 点的时候没有退 token……比如你有 8064 个 token，命中了 tail checkpoint，但你没有新 token 用来 forward，那不还是有问题吗"

两个口径并排：

- 常规链：`token_ids.slice(0, token_ids.size() - 1)`（linear:580）——**截**。
- tail 点查：`tail_checkpoint_tokens(seq)` 读的是 `seq->num_prompt_tokens()`（linear:348-349）——**不截**。

### 2.2 裁决：构造不出来，两个口径是同一不变量的两种写法

担心的事：prompt = 8064，tail = 8064，命中后一个 token 都不用算——forward 无法产出输出。**这个场景构造不出来**，三重兜底：

1. **规则源头就堵死**（scheduler_config.h:111-115）：

```cpp
// scheduler_config.h:111-115
  const size_t tail_tokens = (prompt_tokens / kv_block_size) * kv_block_size;
  // An exact prompt hit must leave a token for the current forward. Dropping
  // a KV token alone would leave LINEAR at a different restore boundary.
  if (tail_tokens == prompt_tokens || tail_tokens % checkpoint_stride == 0) {
    return 0;
  }
```

   prompt = 8064 → `tail_tokens == prompt_tokens` → 返回 0，点查根本不发起。**有效 tail 恒 ≤ prompt − 1**（prompt ∈ [8065, 8191] 时 forward 余量 ≥ 1）。

2. **挂载断言**：`CHECK_LT(num_shared_tokens, current_total_num_tokens)`（sequence_kv_state.cpp:256）——恢复深度必须小于总 token 数，否则进程崩给你看。

3. **消费侧兜底**：`num_need_compute_tokens() = num_tokens_ - kv_cache_tokens_num()`（sequence.h:210-211）——深度 == prompt 则待算 0，无法组 batch。

手算证明"tail 若也截 1"是重复设防：

```
prompt=8191，若 tail 点查也先截 1：
  (8191−1)/128 = 63（整除）→ 63×128 = 8064   ← 与不截的结果同值
```

因为 `floor((n−1)/128)*128` 与 `floor(n/128)*128` 只在 n 恰为 128 整倍数时不同——而那种情况已被规则一（tail==prompt→0）挡在门外。**截尾不对称 = 同一不变量（恢复深度 < prompt）的两种写法**，不是漏洞。

---

## 3. tail_checkpoint_hash 取键：键从哪来、何时写入、为何取时有货

### 3.1 用户原话重述

> "tail_checkpoint_hash（:352-363）取键：hash_index = cached_tokens / hash_block_size − 1，再从 KV 链取 hashes[hash_index]。这个 hashes[hash_index] 的内容是从哪儿放入的，怎么保证每次取的时候都有内容呢"

### 3.2 公式贴码

```cpp
// block_utils.h:29-36
// Map a completed state boundary to its key in the logical KV hash chain.
inline size_t linear_state_checkpoint_hash_index(size_t checkpoint_tokens,
                                                 size_t hash_block_size) {
  CHECK_GT(checkpoint_tokens, 0u);
  CHECK_GT(hash_block_size, 0u);
  CHECK_EQ(checkpoint_tokens % hash_block_size, 0u)
      << "Linear checkpoint boundary must align with hash block size.";
  return checkpoint_tokens / hash_block_size - 1;
}
```

```cpp
// linear_state_block_manager.cpp:352-363
std::optional<XXH3Key> LinearStateBlockManager::tail_checkpoint_hash(
    Sequence* seq, size_t token_count) {
  const size_t hash_index = linear_state_checkpoint_hash_index(
      token_count, options_.hash_block_size());
  seq->update_block_hashes(options_.hash_block_size(), options_.hasher_type());
  const Slice<XXH3Key> hashes = seq->block_hashes();
  if (hashes.size() <= hash_index) {
    return std::nullopt;             // ← 链不够长，点查不发起
  }
  return XXH3Key(hashes[hash_index].data);
}
```

减 1 的原因：链式第 i 格覆盖 token 区间 `[i*128, (i+1)*128)`，边界 N（如 8064）由**最后覆盖它的那格**见证——第 `8064/128−1 = 62` 格。

### 3.3 关键消歧：这里读的是"序列自己的链"，不是"块上的键"

用户问"hashes[hash_index] 的内容从哪儿放入"，答案要先分清**两层键**：

| 层 | 住哪 | 谁写入 | 内容 |
|---|---|---|---|
| 序列 hash 链 | `Sequence::block_hashes_by_domain_`（每序列私有） | `update_block_hashes` 从 token 现算 | 链式摘要：第 i 格 = hash(本格 128 token + 上格摘要) |
| 块上的键 | `Block::hash_value_`（16 字节，物理块私有） | 发布者盖键（`set_hash_value`） | 从序列链上抄来的某一格 |

`tail_checkpoint_hash` 读的是**第一层**：B 序列拿自己的 prompt 现算一条链（链式 hash 对相同 token 确定性可复现），取第 62 格当查询键。

A 当年发布时，把**自己链上的第 62 格**盖到了物理块上再 insert 进缓存——同 token 必得同格，所以 B 能 find 到 A 的条目。

### 3.4 "取的时候有货"的保证：update-then-read 时序

序列链不是预先算好的，而是**每次读之前当场增量延长**。全库四个读点全部遵守 update-then-read：

- tail 点查：`update_block_hashes` 在读链前一步（linear:357→358）。
- 常规发布：linear:417-419。
- 常规链 probe 的备料：composite:583-585。
- host 预取分配：linear:206。

双保险：读之前有尺寸守卫——`hashes.size() <= hash_index` 则返回 `nullopt`（linear:359-361），发布路径则是硬 `CHECK_LT`（linear:422）。链长不够就不查/不发，不会拿到垃圾。

### 3.5 发布侧盖键时点链：谁在哪个时机把键写上块

第二层（块上的键）的写入时点，按物理位置列全：

- 设备常规发布：linear:423 `source->back().set_hash_value(hashes[hash_index].data)`。
- 设备 tail 发布：linear:440（注释原文 "Stamp the source itself so D2H carries the same tail key"——盖在源块上，D2H 下沉时键跟着内容走）。
- host 预取分配：linear:206 备链，:230 新分配的目的块当场盖检查点键。
- D2H 配对：hierarchy_block_manager_pool.cpp:565，源块的键抄给目的块。
- host 预取 trim 发布：linear_state_prefix_cache.cpp:173，insert 内部当场从传入的链上取键盖块。

不变量一句话：**键必在 insert 登记之前盖在块上（或由 insert 当场盖）；hash 链节节都有（每 128 token 一格），键只盖"完成可复用边界"的块**——这是设计不是失误。

失败画面（去掉盖键，三段叙事）：

1. 正确状态：tail 源块发布时键已盖好，D2H 把"带着钥匙的内容"拷去 host。
2. 被破坏：跳过 linear:440，块上的 `hash_value_` 保持出厂值（无效）。
3. 下游读到什么：host 侧 insert 拿垃圾键登记 → 未来请求的点查永远 miss → **白白拷了一份永远用不上的状态**，且无任何日志。

---

## 4. 发布守卫逐行与单块 insert 协议

### 4.1 用户原话重述

> "就这样就发布了吗，source 上会带 type 呀、hash 呀、index 呀这些内容吗，都已经发布了还 `kv_state.set_num_cached_blocks` 有啥用呢"

### 4.2 守门函数：cache_completed_state

```cpp
// linear_state_block_manager.cpp:385-395（守门两行）
void LinearStateBlockManager::cache_completed_state(Sequence* seq) {
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  const size_t checkpoint_tokens = tail_checkpoint_tokens(seq);
  if (checkpoint_tokens == 0 || cached_tokens != checkpoint_tokens) {
    return;                          // ← 只发布恰好停在 tail 边界的完整状态
  }
  cache_read_source(seq, seq->kv_state());
}
```

守门语义：`cached_tokens == tail` 才走 tail 发布。常规检查点则由 `cache_read_source` 经 chunked prefill 每步顺路发布。

`cache_read_source` 进门还有三道闸（linear:397-410）：

- host 槽直接 return（:401-403）——host 槽是 D2H 的目的地，**预定座位 ≠ 数据已到**，只有传输完成回调可发布。
- graph warmup return。
- 无 prefix_cache 或块向量空 return。

### 4.3 常规分支逐行（linear:412-427）

```cpp
// linear_state_block_manager.cpp:412-427
  if (cached_tokens % block_size() == 0) {
    const size_t checkpoint_index = cached_tokens / block_size();
    if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
      return;                        // ← 幂等守卫：已发布过（或账面已更深）不再发
    }
    seq->update_block_hashes(options_.hash_block_size(),
                             options_.hasher_type());
    const Slice<XXH3Key> hashes = seq->block_hashes();
    const size_t hash_index = linear_state_checkpoint_hash_index(
        cached_tokens, options_.hash_block_size());
    CHECK_LT(hash_index, hashes.size());
    source->back().set_hash_value(hashes[hash_index].data);   // 盖键
    prefix_cache_->insert(
        Slice<Block>(*source).slice(source->size() - 1, source->size()));  // 只发尾块
    kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);   // :426 推进游标
    return;
  }
```

五行动作：幂等守卫 → 备链 → 取键 → 盖键 → insert 尾块切片 → **游标推进（:426）**。

**幂等守卫（:414）的回答**：`set_num_cached_blocks` 的用处就是让这个守卫有账可查。它是"本序列已发布到第 k 个检查点"的游标；没有它，同一边界被反复触发（chunked prefill 每步都调 `cache_read_source`）就会重复 insert。

失败画面（去掉幂等守卫，三段叙事）：

1. 正确状态：边界 8192 发布一次，缓存里该键一个 LRU 节点，ref = 序列 + 缓存 = 2。
2. 被破坏：守卫删除，8192 边界在下一步分配时再发一次。
3. 下游读到什么：同键登记出第二个 LRU 节点（或同节点 ref 再 +1）——**引用计数虚高、占着槽位不干活**；LRU 逐出按 ref 判断"在用"，这个虚账让块多活很久。

### 4.4 tail 分支逐行（linear:429-443）

```cpp
// linear_state_block_manager.cpp:429-443
  if (tail_checkpoint_tokens(seq) != cached_tokens ||
      kv_state.linear_state_tail_tokens() == cached_tokens) {
    return;                          // ← tail 版幂等：便签已记此边界则不发
  }
  const std::optional<XXH3Key> hash = tail_checkpoint_hash(seq, cached_tokens);
  if (!hash.has_value()) {
    return;
  }
  // This completed slot becomes the immutable read source for the next
  // chunk. Stamp the source itself so D2H carries the same tail key.
  source->back().set_hash_value(hash->data);
  prefix_cache_->insert(
      Slice<Block>(*source).slice(source->size() - 1, source->size()));
  kv_state.set_linear_state_tail_tokens(cached_tokens);   // ← tail 便签 = tail 版游标
```

tail 分支的幂等不查常规游标、查 tail 便签（:430-432）——**双账本各自守卫**，因为量纲不同（检查点个数 vs token 深度）。

### 4.5 单块 insert 发布了什么：Block 五字段核验

两个发布分支都走 `insert(Slice<Block>)` 单块重载（prefix_cache.cpp:185-222），它读**块上已盖的键**登记：

```cpp
// prefix_cache.cpp:190-194
  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i].is_valid()) {
      continue;
    }
    token_hash_key.set(blocks[i].get_immutable_hash_value());  // ← 键从块上读
```

发布的东西 = 一个 Block 句柄。回源码核验 Block 的全部字段（block.h:184-199）：

| 字段 | 行号 | 量纲/内容 | 发布时是否随块走 |
|---|---|---|---|
| `id_` | block.h:185 | int32 块号 | 是 |
| `size_` | block.h:188 | uint32 token 数 | 是 |
| `ref_count_` | block.h:194 | 原子引用计数（跨线程别名共享） | 是（insert 时 +1） |
| `manager_` | block.h:197 | 归属管理器指针 | 是 |
| `hash_value_` | block.h:199 | 16 字节 XXH3 键 | 是（盖键后） |

**五字段，就这些。** 用户问的三样东西：

- **type 不带**：每个叶有自己独立的 PrefixCache，登记进 LINEAR 叶的缓存天然是 LINEAR 的——类型由"住在哪个缓存"隐含。
- **hash 带**：就是 `hash_value_`，发布前一步刚盖的。
- **index 不带**：深度编在链式键本身里——链上第 N 格天然代表"前 (N+1)×128 token"，不需要另设下标。

发布 = 登记进 `cached_blocks_` map + ref+1，**无任何数据搬运**。

### 4.6 发布协议全景（host 预取路径的另一种 insert）

host 预取 trim 的常规发布走带 token 的 insert（linear_state_prefix_cache.cpp:161-173）：键由 insert **当场**从传入的链上取并盖块（`blocks[checkpoint].set_hash_value(token_hash_key.data)`，:173），且跳过无效占位块（:170-172）。与设备侧"先盖键再 insert"是同一不变量的两种实现。

---

## 5. trim 决策、mount 双动作与 base_tokens

### 5.1 trim_flat_kv_linear 逐行：tail 优先 + 常规按块回退

probe 拿回的各叶结果必须裁到**同一个 token 边界**（KV 与 LINEAR 恢复深度必须一致），这是 `trim_flat_kv_linear`（composite_block_manager.cpp:676-782）的工作。

tail 优先分支（:687-696）：

```cpp
// composite_block_manager.cpp:687-696
    if (linear->tail_block.has_value()) {
      const size_t tail_tokens = linear->tail_token_count;
      const size_t tail_safe_count =
          std::min(kv.blocks.size(), tail_tokens / kv_block_size);
      if (tail_safe_count * kv_block_size == tail_tokens &&
          linear->tail_block->is_valid()) {
        safe_count = tail_safe_count;      // ← 整块盖满且块有效，才用 tail
        use_tail_checkpoint = true;
      }
    }
```

两个条件缺一不可：KV 命中块数**整块盖满** tail 深度（`tail_safe_count * kv_block_size == tail_tokens`），且 tail 块本身有效。

常规回退分支（:697-711）——**"按 block 回退"的真身**：

```cpp
// composite_block_manager.cpp:697-710
    if (!use_tail_checkpoint) {
      const size_t regular_tokens = linear->blocks.size() * linear->block_size;
      safe_count = std::min(kv.blocks.size(), regular_tokens / kv_block_size);
      while (safe_count > 0) {
        const size_t safe_hit_tokens = safe_count * kv_block_size;
        if (safe_hit_tokens % linear->block_size == 0) {
          const size_t linear_index = safe_hit_tokens / linear->block_size - 1;
          if (linear_index < linear->blocks.size() &&
              linear->blocks[linear_index].is_valid()) {
            break;                     // ← 踩在有效 LINEAR 检查点上，停
          }
        }
        --safe_count;                  // ← 每次减 1 个 KV 块（128 token）
      }
    }
```

每次减 1 个 KV 块向下让步，直到 KV 深度恰好落在"有物理块的常规检查点"上。这是 **KV 对齐的让步，不是查找**——vLLM mamba 的右→左 Early-stop 扫描才是查找侧的逐块回退（§6）。

### 5.2 max_hit_tokens < tail_tokens：tail 保不住的失败画面

用户原话靶子："如果实际可命中的深度比 tail 浅，这块 tail 保不住，直接 deallocate"。代码住址在 host 预取 trim（linear:255-258）：

```cpp
// linear_state_block_manager.cpp:254-258
  if (tail_tokens != 0) {
    if (max_hit_tokens < tail_tokens) {
      deallocate(blocks);              // ← 防御护栏：tail 深度盖不满就整页放弃
      return;
    }
```

trim 的 tail 分支（composite:689-695）同款逻辑：`tail_safe_count * kv_block_size != tail_tokens` → 不用 tail，tail 块进 `to_drop` 退还。

失败画面（去掉这个护栏、硬 mount tail，三段叙事）：

1. 正确状态：KV 命中 12800 token（100 块），tail 深度 19968——两边深度不一致，trim 应回退到双方都能站住的边界。
2. 被破坏：无视 `12800 ≠ 19968`，照样把 tail 块 mount 进 LINEAR、账本记 19968。
3. 下游读到什么：LINEAR 从 S_19968 恢复，但 KV 只到 12800——**恢复边界越过实际 KV 7168 token**，注意力层读 12800..19968 这段时拿不到 KV，第一个输出 token 就错，之后每步 decode 在错误起点上复利叠加。**宁可少复用，不可错复用**——这就是护栏存在的理由。

### 5.3 mount 双动作：先清零、调用方补设

```cpp
// composite_block_manager.cpp:924-930
      for (ProbeResult& probe : trimmed.to_mount) {
        if (probe.type == BlockType::LINEAR) {
          kv_state.mount_composite_shared(probe.type, std::move(probe.blocks));
          if (probe.logical_token_count != 0) {
            kv_state.set_num_cached_blocks(probe.type, 0);      // :928 清常规账
            kv_state.set_linear_state_tail_tokens(probe.logical_token_count);  // :929 设 tail 账
          }
        }
```

`mount_composite_shared` 是"搬块"的机械动作，不知道搬的是不是 tail：它先清 tail 便签（sequence_kv_state.cpp:271），再把游标设成**向量长度**（:284）。tail 挂载向量恒为 1 块——trim 的 tail 胜出分支把常规命中块全部丢进 `to_drop`（composite:743-748），向量只剩追加的 tail 源 1 块（composite:749）；host 预取的放回路径更用 `CHECK_EQ(blocks.size(), 1u)` 断言钉死（linear:259-260）。

于是游标报 1 是**假账**：这 1 块是 8064 深度的 tail 物理源，不是第 1 个常规检查点（8192 边界）。随后调用方用 `probe.logical_token_count`（哨兵：非零 = 是 tail，全库唯一赋值点 composite:751）纠正：游标清 0 + 便签设 8064（主旅程步 9）。

失败画面（mount 后不纠正，三段叙事，沿用主旅程 prompt=8191 / tail=8064）：

1. 正确状态：tail 挂载后（游标 0，便签 8064），发布守卫按双账各查各的。
2. 被破坏：沿用向量长度的假账——向量 1 块 → 游标虚高 k=1，假账的语义读法是"本序列已拥有第 1 个常规检查点（8192 边界）"，而向量 0 号槽里躺的实物是 8064 深度的 tail 源。
3. 下游读到什么：发布守卫 `checkpoint_index <= num_cached_blocks`（linear:414）在序列真算到 8192 边界时判 1 ≤ 1 成立，跳过这次真发布——0 号槽的 tail 源（8064）顶着**第 1 个常规检查点（8192 边界）**的名分（假游标为 k 时，这个名分槽位是 blocks[k−1]；k=1 即 0 号槽）。**账面 8192 vs 实物 8064，错位 128 token**：之后每次从这页账恢复，LINEAR 状态都从浅 128 token 的起点复利叠加——不崩溃、不报错，长输出质量静默劣化（三个受害点的完整推演见第三章 §3.4）。

### 5.4 base_tokens：host 命中深度（hierarchy:1039）

```cpp
// hierarchy_block_manager_pool.cpp:1033-1043
  host_manager->allocate_shared_for_sequence(sequence,
                                             sequence->host_kv_state());  // ← host 准入 probe+trim+mount
  sequence->host_kv_state().set_prefix_cache_matched(false);             // :1038 诚实标记
  const size_t base_tokens = sequence->host_kv_state().kv_cache_tokens_num();  // :1039
  // A tail checkpoint can be deeper than the largest regular unit that fits
  // the prompt. Keep the already matched boundary in the prefetch transaction
  // so a zero regular target cannot discard its physical state.
  const size_t target_tokens = std::max(regular_target_tokens, base_tokens);  // :1043
```

`base_tokens` 直接读 host 账本的 `kv_cache_tokens_num()`——**准入刚挂载完，这个数就是 host 缓存命中的深度**。两个配套细节：

- `:1038` 把匹配标志设回 false：Store 回调验证之前，host 命中不算"正式匹配"（诚实标记）。
- `:1043` `target = max(regular, base)`：零常规目标不能丢掉已命中的 tail（例：prompt 5000 → regular_target = 0，但 tail 命中 4992 时 target 必须保住 4992）。

tail 命中时取货单恒空（hierarchy:192-194：`linear_state_tail_tokens() != 0` 直接返回空单——"A matched tail is already the deepest complete prompt boundary. Its single physical source does not occupy a regular Store unit position."）。

---

## 6. 三家对照：vLLM / SGLang 也这样管混合 attention 的 kvcache 吗

用户点名对照。一句话总答：**都有 probe 和 trim 的等价物，都没有 tail**。细节见姊妹笔记（vllm-sglang-cache-mtp），此处只给对照表：

| 维度 | xLLM 本 MR | vLLM | SGLang |
|---|---|---|---|
| 查找入口 | probe 两问：常规链 gap-tolerant 逐检查点 + tail 单键点查 | `find_longest_cache_hit`：块级链查，Phase 2 块内精查（partial hash hits） | radix 树 `match_prefix`：字典逐段下走 |
| 混合 attention 状态 | LINEAR 检查点（stride 8192）+ tail 检查点 | mamba 组独立 manager，右→左 Early-stop 扫描 [推演] | 树节点挂 mamba 状态槽，match + mamba validator |
| 回退发生在哪 | **trim 侧**按 KV 块回退对齐（composite:700-710） | 查找侧逐块回退（mamba finder 查到即停） | KV 命中向下对齐 `(hit//grid)*grid`，网格 = lcm(chunk, page) |
| 尾部不对齐 prompt | **tail 检查点**：8191 复用 8064（98.5%） | 无 tail 概念（有近亲 partial_tail [待验证]） | 无 tail；SuffixReplay 论文视其为现有系统做不到的事 |
| 防撞键在哪层 | 键层（烤进链式键） | 消费层（丢边界块重算） | 结构层（树键无法构造同键） |

xLLM 的独特处：**唯一把"块内精查"做进 LINEAR 侧尾界的**——vLLM 的 Phase 2 块内精查与 xLLM 的 tail 点查同属"块内找更深边界"的思想，但 vLLM 精查的是 KV 块内 hash 单位、丢边界块；xLLM 点查的是 LINEAR 状态的确定 tail 键、不丢。

---

## 7. 手算全程：一个 prompt 的准入全景

### 7.1 主旅程：prompt = 8191（tail 单问有货、常规链空手）

设定：hash_bs = 128，stride = 8192。请求 A（同 prompt 的上一个请求）已跑完：A 的 chunked prefill 被 cap 函数切成 {8064, 127}（scheduler_policy.cpp:514-519：tail_checkpoint 8064 > current 0 → 第一片 min(8192, 8064) = 8064）；A 在 8064 边界发布了 tail（键 hashes[62]）和 63 块 KV。请求 B 到达，每步一行：

| 步 | 动作 | 代码位置 | 数字 |
|---|---|---|---|
| 1 | host 准入 probe 问一（常规链） | linear:580 → lspc:44-48 | slice 后 8190 token → 8190/8192 = **0 个检查点** → 空手，零探索 |
| 2 | host 准入 probe 问二（tail 点查） | linear:367 | tail = (8191/128)×128 = **8064**；8064≠8191、8064%8192≠0 → 有效 |
| 3 | tail 取键 | linear:355-362 | hash_index = 8064/128−1 = **62**；链长 8191/128 = 63 格 > 62 ✓ → hashes[62] |
| 4 | find 点查 | linear:374 | A 留下的 tail 条目命中 → TailPrefixCacheMatch{块, 8064} |
| 5 | tail vs 常规比深 | composite:607 | 8064 > 0×8192 → 保留 tail_block |
| 6 | KV 叶同轮 probe | prefix_cache.cpp:64-69 | 8190/128 = 63 块逐查，全命中（miss 即停但没 miss） |
| 7 | trim tail 分支 | composite:687-696 | tail_safe_count = min(63, 8064/128=63) = 63；63×128 = 8064 == tail ✓ 且块有效 → **用 tail** |
| 8 | trim 收尾 | composite:742-752 | 常规 LINEAR 空；tail 块进 mount 向量；logical_token_count = 8064 |
| 9 | mount LINEAR | composite:926-929 | mount（清便签、游标暂=1 假账）→ 游标清 **0** + 便签 **8064** |
| 10 | mount KV | composite:932 | add_shared_blocks 63 块 → kv_cache_tokens_num = **8064** |
| 11 | 读 base_tokens | hierarchy:1039 | host 账本 = **8064**（host 命中深度） |
| 12 | 算 target | hierarchy:1027-1043 | cacheable = min(8190, KV 8064, LINEAR 0×8192=**0**) = 0 → regular_target = 0；target = max(0, 8064) = **8064**（tail 保住） |
| 13 | 取货单 | hierarchy:192-194 | 便签 8064 ≠ 0 → **空单**（tail 已是最深完整边界） |
| 14 | 正式调度 H2D | hierarchy:867-890 | 设备空 → host 8064 > 0 → 拷 63 块 KV + 1 块 LINEAR tail 上设备，设备同款 mount |
| 15 | B forward | sequence.h:210 | 8191 − 8064 = **127 token**（一个 chunk，cap: min(127, 128) = 127） |
| 16 | B 期间发布 | linear:389-393 | cached 到 8191 ≠ tail 8064 → 不发；tail 条目已由 A 发布且 B 的便签幂等双保险 |

### 7.2 副旅程：prompt = 20000（两问都有货 + 发布守卫推进）

请求 A 的 chunk 划分与发布（发布守卫的推进全程）：

| 步 | A 的动作 | 数字 |
|---|---|---|
| A1 | chunk1 → 发布常规检查点 1 | 0→8192；8192%8192=0 → index 1 > 游标 0 → 盖键 hashes[63]，游标 0→**1** |
| A2 | chunk2 → 发布常规检查点 2 | 8192→16384；守卫 2 ≤ 1？否 → 发布 hashes[127]，游标→**2** |
| A3 | chunk3（cap: min(8192, 19968−16384)）→ 发布 tail | 16384→19968；19968%8192≠0 → tail 分支；便签 0≠19968 → 盖键 hashes[155]，便签 = **19968** |
| A4 | chunk4 | 19968→20000（余 32） |

请求 B（同 prompt）host 准入：

| 步 | B 的动作 | 数字 |
|---|---|---|
| B1 | 问一常规链（gap-tolerant） | slice 后 19999 → 19999/8192 = **2 个检查点**；probe hashes[63]✓、hashes[127]✓ → 最深 16384 |
| B2 | 问二 tail 点查 + 比深 | tail = 19968 → hashes[155] → find ✓；19968 > 16384 → 保留 tail_block |
| B3 | trim tail 分支 | tail_safe_count = min(156, 156) = 156；156×128 = 19968 ✓ → 用 tail |
| B4 | 常规命中退还 | composite:743-748 → release_probes 掉引用（LRU 位置不动，浅命中已被 probe 刷新过） |
| B5 | mount | LINEAR = [tail 块]、游标 0 + 便签 19968；KV 留 156 块 |
| B6 | base_tokens / forward | hierarchy:1039 读到 **19968**；B 只 forward **32 token** |

gap-tolerant 的价值演示（把 B1 改造）：若检查点 1（hashes[63]）已被 LRU 逐出而检查点 2 还在——

```
probe hashes[63] → miss → 放无效占位，继续
probe hashes[127] → hit → last_hit = 1
结果向量 [占位, 块@16384]（resize(2) 截到最深命中）
trim 常规分支：safe_count = min(KV 156 块, 2×8192/128=128) = 128 → safe_hit_tokens = 16384
  → source_index = 16384/8192 − 1 = **1**（composite:755 的变量：结果向量里最深有效命中的槽位，
    与 §5.1 回退循环里的 linear_index 同式）→ 保住 1 号槽的块@16384 当挂载源，
  0 号槽占位块进 to_drop；KV 对齐到 16384
```

KV 链在同一场景会 miss 即停在块 63——**LINEAR 检查点的自包含性让"中间丢一个"不致命**。

### 7.3 对照例：tail 找到了但保不住（E 例，测试锚点）

block_manager_test.cpp:826-859（TailMismatchFallsBackToShorterRegularLinearCheckpoint），参数缩到 bs=4 / stride=8 / prompt=13。测试两侧都做了预插：KV 侧先用 8 token 的同前缀序列 cached_seq 走一遍 allocate+cache（:832-836），KV 缓存里只有前 8 token = 2 块；LINEAR 侧显式插入 regular 键（边界 8）和 tail 键（边界 12）（:846-847）：

| 步 | 动作 | 数字 |
|---|---|---|
| 1 | 问一常规链 | slice 后 12 → 12/8 = 1 个检查点；hashes[8/4−1=**1**] → 命中块@8 |
| 2 | 问二 tail 点查 | tail = (13/4)×4 = 12 → hashes[12/4−1=**2**] → find **命中**（测试预插了 tail 键） |
| 3 | 比深 | 12 > 8 → 保留 tail_block |
| 4 | trim tail 分支 | KV 链查 3 块、第 3 块（token 8..11）miss 即停 → 命中 2 块；tail_safe_count = min(2, 12/4=3) = 2；2×4 = **8 ≠ 12** → 整块盖满失败 → 不用 tail |
| 5 | 常规回退分支 | safe_count = min(2, 8/4=2) = 2；2×4=8 %8=0 → linear_index = 8/8−1 = 0 < 1 且有效 → **停** |
| 6 | tail 块退还 | drop_tail_candidate → deallocate |
| 7 | mount | KV 2 块 = 8 token；LINEAR 1 块@8；游标 = 1（**不清零**，logical_token_count=0 不触发 :928） |

测试断言（:854-857）：KV shared = 2、kv_cache_tokens = 8、LINEAR shared = 1、num_cached_blocks(LINEAR) = 1——与手算一致。

> **差异登记**：factbase §5.3 把此例描述为"tail 键 12/4−1=2 查不到"。回源码核验：测试在 :846-850 显式插入了 tail_hash（boundary 12），B 的 hashes[2] 与之同链同值，**find 是命中的**；真正失败的是 trim 的整块盖满条件（KV 只有 8 token 盖不满 12）。"tail 保不住"的机制在 trim 不在 find，本章按源码修正。

---

## 8. 章末自检表

| # | 自检问题 | 状态 |
|---|---|---|
| 1 | probe 两问语义讲清了吗（gap-tolerant 逐检查点 + tail 单键点查，miss 跳过继续非即停） | ✓ linear_state_prefix_cache.cpp:62-84 / linear:365-383 贴码核验 |
| 2 | 用户"两方案"正面裁决了吗（方案一方向对但为并列非回退；方案二的按块回退在 trim 不在 probe；"增加 match 可能性"兑现于 gap-tolerant） | ✓ §1.2-1.4 |
| 3 | 探测口径差裁决了吗（截 1 vs 不截，同一不变量两种写法，8064 场景构造不出来） | ✓ §2，三重兜底贴码 |
| 4 | tail_checkpoint_hash 取键公式与"内容何时放入"讲清了吗（两层键消歧 + update-then-read + 五个盖键时点） | ✓ §3 |
| 5 | 发布守卫逐行了吗（:414 幂等 / :426 推进 / :430-432 tail 幂等 / 盖键 / insert 尾块切片） | ✓ §4.2-4.4 |
| 6 | 单块 insert 发布协议核验了吗（Block 五字段表、type 隐含 / hash 盖上 / index 编在键里、无数据搬运） | ✓ §4.5，block.h:184-199 |
| 7 | trim 决策与 max_hit<tail 失败画面讲了吗（整块盖满条件 + 三段叙事：硬 mount → 恢复边界越过 KV → 错位复利） | ✓ §5.1-5.2 |
| 8 | base_tokens 与 target=max 讲了吗（:1039 host 命中深度、:1043 保 tail、tail 命中取货单空） | ✓ §5.4 |
| 9 | 三家对照给了吗（一小节表格，不展开，标注推演/待验证） | ✓ §6 |
| 10 | 手算全程带数字了吗（8191 主旅程 16 步表 + 20000 双问副旅程 + E 例对照） | ✓ §7，每步一行 |
| 11 | 每个机制有失败画面吗（去盖键 / 去幂等 / 硬 mount tail / mount 假账，均三段叙事） | ✓ §3.5 / §4.3 / §5.2 / §5.3 |
| 12 | 所有 file:line 均回读核验？ | ✓ 全部经 `git show commit-4:<path>` 逐行回读；差异 1 条已登记（§7.3） |

**本质一句话**：probe 是"拿自己确定算出的键去查别人留下的登记"，发布是"把完成边界的键盖在块上登记给别人查"——守卫管不重发，trim 管两边站同一边界，tail 因候选唯一而享受点查，gap-tolerance 因检查点自包含而值得。


# 第五章 · 三层存储与 D2H 事务：配对下沉、盖键与 EXPLICIT 发布

> 🔗 **承上启下**：最后一块拼图：这些边界键在 HBM → host → Mooncake 三层之间怎么下沉（配对下沉）、怎么防止旧键冒充新状态（盖键）、怎么显式发布（EXPLICIT）。总览里「每一个物理层级钉同一套边界」的完整实现在本章闭环。

> 核验基准：commit `commit-4`（本 MR head）。所有行号均对该 commit 用 `git show` 逐段回读核验。
> 简写约定：`hierarchy.cpp` = xllm/core/framework/block/hierarchy_block_manager_pool.cpp；`linear.cpp` = xllm/core/framework/block/linear_state_block_manager.cpp；`composite` = xllm/core/framework/block/composite_block_manager.{h,cpp}；`bmi` = xllm/core/framework/block/block_manager_impl.cpp；`pc` = xllm/core/framework/prefix_cache/prefix_cache.cpp；`skvs` = xllm/core/framework/request/sequence_kv_state.cpp；`test` = tests/core/framework/block/hierarchy_block_manager_pool_test.cpp。

## 5.0 本章导航：三问映射

本章回答用户在四轮追问中钉住的三组问题：

```
Q-a（10-12 问题2后半）  mount 用的块从哪来？用完去哪？被覆盖就用不成了吧？     → §5.7 + §5.8
Q-b（10-12 问题5后半）  "盖键"是什么？为什么 tail 块出厂无 hash？              → §5.2 + §5.6
                       "是失误只给 chunk 边界做 hash 吗"？白拷画面没看懂？
Q-c（10-13 问题4）      D2H 之后减引用对吗？异步同步都减吗？                   → §5.4 + §5.3
                       EXPLICIT 发布模式：host 显式 device 自动？
                       还是 D2H 发布过了 deallocate 不再发？
```

三问其实长在同一条生命线上：**一个块从设备下潜到 host、再被拉回来 mount 的完整事务**。§5.1 先给概念地图，之后按事务时序展开。

## 5.1 概念地图：三层介质与六个主角

### 5.1.1 三层介质与三种搬运

```
┌─────────────┐   D2H 下沉    ┌─────────────┐   G2H 远端拉取   ┌──────────────┐
│ HBM（设备）  │ ───────────→ │ host DRAM   │ ←───────────── │ Mooncake Store│
│ "宿舍"      │ ←─────────── │ "仓库"      │ ────────────→  │ "云盘"        │
└─────────────┘   H2D 恢复    └─────────────┘   取货单向它要   └──────────────┘
```

- **D2H**（Device to Host）：设备块拷去 host，本 MR 里专指"配对下沉"。
- **H2D**（Host to Device）：host 命中比设备深时，把差额拷回设备。
- **G2H**（Global to Host）：从 Mooncake Store 远端拉数据填 host 空块。

代码里的方向枚举（model_input_params.h:686-692）：`D2H2G = 4` 表示"设备→host→顺带登记 Store"，D2H 下沉实际走的就是这个复合类型（hierarchy.cpp:1144）。

### 5.1.2 六个主角（首现即教）

**Block（块句柄）**——一个物理槽位的"遥控器"。五个字段（block.h:184-199）：

- `id_`：槽位编号，例值 201。
- `size_`：本块容纳的 token 数，例值 128。
- `ref_count_`：**原子引用计数**，指向共享计数器；每复制一个 Block 别名 +1，每析构一个别名 −1。
- `manager_`：分配它的管理者，析构归零时用它路由回池。
- `hash_value_[16]`：**16 字节键槽**。注意：**没有任何构造函数初始化它**（block.cpp:27-31 的 `Block(id, manager)` 构造不碰这个字段）——块出厂时键槽是垃圾内存。这是 §5.2.4 一切故事的起点。

**盖键**——`set_hash_value`，把 16 字节 memcpy 进块上的键槽（block.h:169-171）：

```cpp
void set_hash_value(const uint8_t* hash_value) {
  memcpy(hash_value_, hash_value, XXH3_128BITS_HASH_VALUE_LEN);
}
```

行李箱挂名牌：键跟数据走，让"这块物理内存里的内容"能被"按前缀算出键的人"找到。

**OffloadBlockPair（下沉配对单）**——D2H 的事务载体（hierarchy_block_manager_pool.h:32-39）：

```cpp
struct OffloadBlockPair {
  Block src;               // 设备源块（内容在这里）
  Block dst;               // host 目的块（空壳，等着接货）
  BlockType block_type;    // 完成回调按它路由发布到哪个 host 叶
  size_t checkpoint_row;   // decode 检查点行号，预取路径用 0
};
```

谁创建：`collect_offload_pairs`；谁消费：D2H 完成回调；活多久：从配对到异步回调结算完。

**EXPLICIT 发布模式**——host 缓存的"只许点名发布"开关（composite.h:77-80）：

```cpp
enum class PrefixCachePublishMode : int8_t {
  AUTOMATIC,   // 设备用：序列驱动的发布钩子正常工作
  EXPLICIT,    // host 用：钩子全部早退，只认显式入口
};
```

**finalize_host_blocks（收货上架函数）**——D2H 完成回调的最后一步（hierarchy.cpp:39-59）：按 `publish` 标志决定"发布进 host 缓存还是只释放"。

**取货单（StoragePrefetchRequest）**——向 Store 要货的清单，只列 8192 整单元（§5.5）。

### 5.1.3 谁管哪层

- 设备侧：`CompositeBlockManager`（AUTOMATIC，block_manager_pool.cpp:99 构造时不传模式，走默认参数 composite.h:85）。
- host 侧：每个 dp_rank 一个独立 `CompositeBlockManager`（dp_rank=数据并行 rank：DP 组内的副本编号，每个 DP 副本各配一套 host 复合），**构造时显式传 EXPLICIT**（hierarchy.cpp:469-472）：

```cpp
host_block_managers_.emplace_back(std::make_unique<CompositeBlockManager>(
    std::move(per_type),
    composite->options(),
    CompositeBlockManager::PrefixCachePublishMode::EXPLICIT));
```

## 5.2 配对下沉全流程：collect_offload_pairs

### 5.2.1 全景：finish 五步里的位置

序列结束时 `HierarchyBlockManagerPool::deallocate`（hierarchy.cpp:491-510）依序做五件事：

```
① cache_full_blocks_for_sequence   最后一搏发布（设备侧 AUTOMATIC 才做事）
② collect_offload_pairs            配对下沉 ← 本章主角
③ host_manager->deallocate_for_sequence  host 账本掉引用（EXPLICIT 早退不发布）
④ composite->deallocate_for_sequence     设备账本掉引用（含 prefix cache flush）
⑤ sequence->reset()                只抹账面
```

步②的函数头（hierarchy.cpp:512-520）：遍历 host 管理器的每个叶，把"设备上已完成的块"和"host 侧新分配的空目的块"配成一对。**host 目的块从哪来**：正式调度 `allocate` 时 host 复合已经同步分配了镜像空壳（hierarchy.cpp:766），ref=1、无内容、无键。

### 5.2.2 KV 常规循环（hierarchy.cpp:631-676）

贴码（核心段）：

```cpp
const size_t completed_tokens = hbm_state.kv_cache_tokens_num();
size_t completed_blocks = completed_tokens / block_size;
// ... lookahead / needs_sequence_hash 两处修正（略）...
const size_t comparable_blocks =
    std::min({hbm_blocks->size(), host_blocks->size(), completed_blocks});
for (size_t i = 0; i < comparable_blocks; ++i) {
  Block& hbm_block = (*hbm_blocks)[i];
  Block& host_block = (*host_blocks)[i];
  const uint32_t expected_hbm_refs = entry.supports_prefix_cache ? 2 : 1;
  if (hbm_block.ref_count() != expected_hbm_refs ||
      host_block.ref_count() != 1) {
    continue;                       // ← 资格检查：设备块必须"序列+缓存"双持
  }
  host_block.set_hash_value(needs_sequence_hash
                                ? hashes[i].data
                                : hbm_block.get_immutable_hash_value());
  auto pair = std::make_shared<OffloadBlockPair>(
      OffloadBlockPair{/*src=*/hbm_block,
                       /*dst=*/std::move(host_block),
                       /*block_type=*/type,
                       /*checkpoint_row=*/0});
  offload_block_pair_queues_[dp_rank].enqueue(std::move(pair));
}
```

逐行读三件事：

- **资格检查**（:658-661）：设备块 ref 必须是 2（序列+设备缓存，prefix-capable 叶）或 1（decode SWA 从未发布；SWA=滑窗注意力 sliding window attention，此处指 DSV4 类模型的滑窗 KV cache 组，block.h:45）；host 目的块必须 ref==1（只有序列 host 账本一份，还是空壳）。
- **盖键三元**（:666-668）：host 目的块的键 = 设备源块上已盖的键（`get_immutable_hash_value` 分支），或从序列 hash 链现算（`hashes[i]` 分支，decode SWA 没盖过键时的兜底）。
- **配对**（:669-674）：src 是**拷贝**（Block 拷贝构造 inc_ref，block.cpp:37-47），dst 是 **move**（引用不变，host 向量槽位变无效占位）。pair 进每 dp_rank 一条的异步队列。

### 5.2.3 LINEAR prefill 分支（hierarchy.cpp:580-628）

LINEAR 状态块走单独通道。资格门槛（逐条独立）：

- host 已有同检查点就不再下沉——`already_on_host`：host 首块 ref≥2 说明"host 缓存+序列"双持（:604-609）。
- 源必须是 prefill 的 [restore, live] 双槽里的 **restore 槽**（index 0）：它是"已完成的检查点"，live 槽是正在写的活块（:590-593 注释）。

盖键与配对（:617-623）：

```cpp
Block& host_destination = host_blocks->front();
host_destination.set_hash_value(hbm_block.get_immutable_hash_value());
auto pair = std::make_shared<OffloadBlockPair>(
    OffloadBlockPair{/*src=*/hbm_block,
                     /*dst=*/host_destination,
                     /*block_type=*/BlockType::LINEAR,
                     /*checkpoint_row=*/0});
```

**键直接从源块抄**。源块的键哪来的？发布时盖的——`cache_read_source` 的 tail 分支在设备侧发布时就盖了源块（linear.cpp:438-440，注释原文下一节贴）。

### 5.2.4 LINEAR decode 分支（hierarchy.cpp:533-577）

decode 阶段的 LINEAR 块**从未发布进设备 prefix cache**（decode 不 probe LINEAR，composite.cpp:573-575），所以源块出厂至今**无键**。这个分支在配对边界现算键、先盖源再抄目的：

```cpp
const size_t checkpoint_index = linear_state_checkpoint_hash_index(
    checkpoint_tokens, hash_block_size);            // :551-552
sequence->update_block_hashes(hash_block_size,
                              entry.leaf->options().hasher_type());
const Slice<XXH3Key> checkpoint_hashes = sequence->block_hashes();
CHECK_GT(checkpoint_hashes.size(), checkpoint_index);
hbm_blocks->front().set_hash_value(
    checkpoint_hashes[checkpoint_index].data);      // :557-558 ← 先盖源块

auto pair = std::make_shared<OffloadBlockPair>();
pair->src = std::move(hbm_blocks->front());         // :561 move（零减引用）
pair->dst = std::move(host_blocks->front());        // :562 move
pair->dst.set_hash_value(pair->src.get_immutable_hash_value());  // :565 ← 抄给目的块
hbm_blocks->erase(hbm_blocks->begin());             // :571 空壳 erase
host_blocks->erase(host_blocks->begin());
```

配套注释（linear.cpp:497-503）说清了设计意图：

> "HierarchyBlockManagerPool owns the transfer boundary. It moves the HBM and Host front blocks into an OffloadBlockPair after both tiers preserve the same logical checkpoint. **Hashing happens once at that ownership boundary.**"

配套的所有权注释（hierarchy.cpp:569-570）：

> "The pair, rather than either rolling vector, owns both blocks until the asynchronous D2H completion releases or publishes them."

**配对=所有权搬家，零减引用**。两向量槽位被 move 走 + erase，别名没多没少，只是从"序列账本"过户到"配对单"。

### 5.2.5 盖键正解：不是"只给 chunk 边界做 hash"

用户问：**"因为有可能失误只给 chunk 边界做 hash 吗？"**——不是失误，且要拆成两问：

**第一问：hash 链是不是只算 chunk 边界？——不是。**

hash 链节节都有：第 i 格 = hash(本格 128 token + 上格摘要)，每 128 token 一个节点（block.h:163-165 注释）。tail 键 8064/128−1=62、常规键 8192/128−1=63——**键在链上处处存在，采样哪个下标由边界决定**（block_utils.h:29-36）。

**第二问：物理块上的键槽是不是出厂就空？——是，且对每类块都一样。**

`hash_value_[16]` 无构造初始化（block.h:199），出厂即垃圾内存。**键从链上"落到块上"（盖键）只发生在明确时机**，全库共三类五处：

- 发布时：LINEAR 常规（linear.cpp:423）、LINEAR tail（linear.cpp:440，注释"Stamp the source itself so D2H carries the same tail key"）。
- 预取分配时：LINEAR（linear.cpp:230）、KV（bmi.cpp:302）。
- D2H 配对时：hierarchy.cpp:557-558（decode 现算盖源）、:565/:618/:666-668（抄给目的块）。

为什么 tail/decode 块"出厂无键"是设计而非欠账：

- LINEAR 槽是 **rolling 活槽**——decode 两槽交替写（linear.cpp:519-526 的 iter_swap），prefill [restore, live] 双槽滚动。活槽内容随时被 forward 改写，给它盖键没有意义。
- 键属于"**完成可复用边界**"的快照，不属于槽位。`cache_completed_state` 的守门（linear.cpp:385-395）就是等块**恰好停在边界**才发布、才盖键。
- 常规 KV 块看起来"出厂带键"，其实是发布路径顺路盖的：insert 的 compute path 会把链键 stamp 到块上（pc.cpp:271-274 起）。发布得早 ≠ 出厂就有。

**一句话**：链上键处处有，块上键只盖边界块——"带着钥匙的内容"必须先到边界才配得上钥匙。

### 5.2.6 host 游标的诚实推进（hierarchy.cpp:678-684）

```cpp
if (queued_offload) {
  // Host advances only after this round has produced a concrete D2H pair.
  // The queue owns the destination block until transfer completion, so the
  // cursor represents a reserved logical checkpoint rather than a published
  // prefix-cache entry.
  host_state.set_kv_cache_tokens_num(hbm_state.kv_cache_tokens_num());
}
```

配对成功才推进 host 账本游标，且注释明说：这个游标是"**预定的座位**，不是已发布的条目"。这句话是 §5.3 EXPLICIT 的伏笔。

## 5.3 EXPLICIT vs AUTOMATIC：发布模式对照（Q-c 主靶）

### 5.3.1 用户两个猜测的裁决：都对，且是同一枚硬币的两面

用户问："host 要显式发布、device 自动发布？还是 D2H 已经发布过了、后边 deallocate 就不再发布了？"

**两个猜测各自命中一半，合起来正好是完整机制**：

- 前半对：模式开关就是"host 显式、device 自动"（composite.h:77-80 + hierarchy.cpp:469-472）。
- 后半也对：D2H 完成回调**就是**那次显式发布；之后 deallocate 路径上的发布钩子因 EXPLICIT 早退，不再重复发布。

### 5.3.2 模式到底关掉了什么：两处早退

**钩子一**：`cache_full_blocks_for_sequence`（finish 步①会调到它）——composite.cpp:337-341：

```cpp
void CompositeBlockManager::cache_full_blocks_for_sequence(Sequence* seq) {
  if (seq == nullptr || seq->is_graph_warmup() ||
      publish_mode_ == PrefixCachePublishMode::EXPLICIT) {
    return;                        // ← host 复合在这里整体哑火
  }
```

**钩子二**：`cache_for_sequence(seq, num_tokens)`（chunked prefill 中途发布）——composite.cpp:960-964，同样 EXPLICIT 早退。

**deallocate 的不重复发布链**：finish 步③ `host_manager->deallocate_for_sequence(sequence, host_kv_state)`（hierarchy.cpp:505）→ composite.cpp:545 先调 `cache_for_sequence(seq)` → 转发 `cache_full_blocks_for_sequence` → :339 EXPLICIT 早退。所以 host 侧 deallocate 只掉引用、不发布——**效果上"D2H 已发布过，这里不再发"**。

### 5.3.3 为什么 host 必须显式：已分配 ≠ 数据到位

linear.cpp:397-403 的守门注释是整个设计的题眼：

```cpp
void LinearStateBlockManager::cache_read_source(Sequence* seq,
                                                KVCacheState& kv_state) {
  // Host slots are D2H destinations. Only the transfer completion callback
  // may publish them; a reserved Host token cursor does not imply ready data.
  if (&kv_state == &seq->host_kv_state()) {
    return;
  }
```

物理差异一句话：**设备块随时是成品**（forward 写完即终态，写完的瞬间数据就在那）；**host 块内容来自异步拷贝**（分配出来是空柜子，§5.2.6 的"预定座位"）。若 host 也用 AUTOMATIC，finish 时钩子会把空柜子 insert 进 host 缓存——别的请求命中半截数据。

注意这是**两层防线**：EXPLICIT 枚举在 composite 层关掉序列驱动的钩子；linear.cpp:401-403 在叶层挡住分配驱动的发布（host 分配时 `allocate_prefill` 会路过 `cache_read_source`，linear.cpp:465）。两个入口都能触发发布，所以两处都要挡。

### 5.3.4 显式发布的三个入口

EXPLICIT 模式下 host 缓存仍会被写入，但只经由三条白名单路径：

- **入口一：D2H 完成回调**——`finalize_host_blocks(publish=true)` → `host_manager->cache_blocks(type, blocks)`（hierarchy.cpp:54-56）→ 叶的已盖键版 insert（bmi.cpp:202-208）。
- **入口二：预取收尾（KV 叶）**——`BlockManagerImpl::trim_prefetch_blocks` 在门禁边界已知后发布保留段（bmi.cpp:344-353）。
- **入口三：预取收尾（LINEAR 叶）**——`LinearStateBlockManager::trim_prefetch_blocks` 的稀疏向量常规发布（linear.cpp:306-319）。

三个入口的共同点：**发布时数据已经到齐，且发布者亲眼见过到齐的证据**（回调的 copy_ok / finalize 的门禁计数）。

### 5.3.5 发布模式对照表

| 维度 | 设备（AUTOMATIC） | host（EXPLICIT） |
|---|---|---|
| 内容何时就绪 | forward 写完即终态 | 异步拷贝完成才算 |
| finish 时发布 | 步① cache_full_blocks 正常发布 | :339 早退，不发布 |
| chunk 中途发布 | cache_for_sequence 正常 | :962 早退 |
| 分配路径发布 | 正常（cache_read_source） | linear.cpp:401-403 直接 return |
| 唯一发布者 | 序列驱动的钩子 | D2H 回调 / 预取收尾三入口 |
| 白名单外的发布 | —— | 全部被两层防线挡死 |

## 5.4 引用账目演化：从配对到回调（Q-c 前半）

### 5.4.1 用户问题的精确化

用户问："先 D2H，然后把转移到 host 上的 block 减少引用对吗？不管异步还是同步都会减引用对吗？"

先纠一个词：**减引用的不是"D2H 这个动作"，而是 Block 别名的析构**。`deallocate(Slice<Block>)` 本身只清 usage 统计（bmi.cpp:101-130 注释："physical ids return to the free list when the last Block alias is dropped"）；真正的 ref−1 发生在每个别名析构时（block.cpp:33-36 的 `~Block → dec_ref_count`）。

### 5.4.2 六行手算扩展版：KV 块 #201 × host 空块 #501

设定：prefix-capable KV 叶，#201 已发布（设备缓存+序列双持），#501 是 host 侧分配的空壳。

```
t0  设备发布后        #201 ref=2（序列 + 设备缓存）         #501 不存在
t1  host 分配空壳     #201 = 2                              #501 ref=1（host 账本序列持）
t2  配对（拷 src/move dst）#201 = 3（+pair 拷贝别名）        #501 = 1（过户给 pair，ref 不变）
t3  finish 步③④+reset #201 = 2（设备缓存 + pair）           #501 = 1（pair；host 槽位已 move 空）
t4  发起 D2H（move 进闭包）#201 = 2                         #501 = 1（引用不变，所有者换闭包）
t5a 回调成功          #201：deallocate+clear → 1（设备缓存） #501：insert +1 → 2 → 闭包析构 → 1（host 缓存持）
t5b 回调失败          #201 同上 → 1                          #501：不 insert 直接析构 → 0 → 回 host 空闲池
```

三个结算点对齐代码：

- t3：finish 步③（hierarchy.cpp:505）+ 步④（:508）+ reset（:509）——序列两本账上的别名析构。
- t5 设备源：回调里 `device_block_mgr_ptr->deallocate(device_blocks); device_blocks.clear();`（hierarchy.cpp:1196-1197）——clear 触发析构。
- t5a host 目的：`finalize_host_blocks` 里 insert（ref+1，hierarchy.cpp:55）后 `deallocate(blocks)`、参数向量析构（:57）——净剩 host 缓存一份。

### 5.4.3 异步/同步的裁定

用户猜"异步同步都会减"——**对，但要钉死时点**：

- 配对不提前减（hierarchy.cpp:569-570 注释）。"零减"指**结算点的净账**：decode 分支 move 零增减；KV/prefill 拷贝分支 src 瞬间 +1（t2 的 #201=3）只是拷贝瞬态，由 finish 时序列别名析构在 t3 恰好抵消——净账仍是"配对不增不减"。
- 发起零减（move 进闭包，hierarchy.cpp:1138-1139）。
- **减引用只发生在完成回调**（t5）。本 MR 的 D2H 全走异步：`engine_->transfer_kv_blocks` 返回每个 worker 一个 `folly::SemiFuture<uint32_t>`（engine.h:97），`folly::collectAll(...).thenValue(...)` 组合（hierarchy.cpp:1160-1163）。
- 若底层 RPC 同步完成，future 立即 ready，回调经 InlineExecutor 照样执行同一套减法——**结算点唯一，异步只是推迟，不改变账目**。测试佐证：`test:1839-1893` 断言回调结算前 `num_used_blocks() > 0`、结算后归零。

### 5.4.4 早减的失败画面

反事实：若配对时就减设备源引用——

```
配对时 #201 ref 2→1 →（序列别名在 reset 时再减）→ 0 → 物理 id 回空闲池
→ 空闲池立刻把它分给新请求 → 新请求 forward 往里写新数据
→ 而 D2H worker 还在读这块内存 → 两份数据同时报废，无任何报错
```

"每份别名恰好减一次、且在数据不再被读之后减"——这就是 pair 独占所有权的全部意义。

## 5.5 D2H-EXPLICIT 取货单事务（Store 侧姊妹篇）

### 5.5.1 取货单只取整单元，tail 从不经

`build_flat_kv_linear_prefetch_request`（hierarchy.cpp:186-243）三处最小重复证据：

```cpp
// A matched tail is already the deepest complete prompt boundary. Its
// single physical source does not occupy a regular Store unit position.
if (host_state.linear_state_tail_tokens() != 0) {
  return {};                        // :192-194 ← tail 命中 → 取货单恒空
}
```

- 单元从 `base_units`（:212-214，host 已有深度折算）开到 `available_units`——**本地已有不重复取**。
- 每单元的 KV 块是门禁块（`is_prefetch_gate_block_type`：KV/C4/C128 为 true，LINEAR/SWA 为 false，block.h:89-101；缩写首教：C4/C128=DSV4 分组注意力的两个压缩 KV cache 组，与 SWA 组并列为模型的三个 cache group，分别导出到 multi_block_tables[1]/[2]，block.h:37/46-47）：单元内任一 KV 块没分到 → `gate_available=false` → **break 砍单**（:233-235）。
- LINEAR 槽是可选块：单元照样建，LINEAR miss 只是少一项（:236-239）。

### 5.5.2 finalize 整单作废判据

prefetch 回调里的成败判据（hierarchy.cpp:1066-1068）：

```cpp
const bool transfer_ok =
    storage_request_ptr->units.empty() ||
    summary.gated_hits.size() == storage_request_ptr->units.size();
const bool publish =
    transfer_ok && !request->finished() && !request->cancelled();
```

- 取货单空（tail 命中/host 已够深）→ 直接成功（:1084-1087 立即 finalize）。
- **门禁块缺一整单作废**：`gated_hits` 记录连续命中的门控单元数（engine 侧 worker 在首次 gate miss 后 STOP，`test:1930-1942`）；不足全单 → `publish=false` → `finalize_prefetch` 走不发布分支：全部 take 出来 deallocate、host 状态整体 reset（hierarchy.cpp:336-342）。
- 全部门禁命中但有可选块 miss → `publish=true`，最终边界=base+最深连续可选单元（`finalize_prefetch_tokens`，hierarchy.cpp:353-383）。

### 5.5.3 D2H worker 门禁（与取货单门禁同构）

D2H 完成回调的两层门禁（hierarchy.cpp:1171-1193）：

```cpp
bool copy_ok = results.size() == expected_workers;   // 第一层：worker 数齐
// ...
if (result.value() != host_blocks.size()) {          // 第二层：每个 worker
  copy_ok = false;                                   //   报告的拷贝数==全部块数
}
```

`host_transfer_worker_count` 个 worker 各拷同一批块的不同张量分片（不同层），**每个 worker 都要报告拷完所有块**。成功才 `finalize_host_blocks(copy_ok=true, ...)`（:1199-1203）；失败"publish nothing, but both paths release every reserved Host id"（:1199-1201 注释）。

### 5.5.4 G 例：门禁的手算（test:2657-2733）

设定：`LinearTailHostOffloadBecomesVisibleAfterAllWorkersComplete`，kv_bs=4、stride=8、tail=12、2 个 worker：

```
步骤 1  配对+发起 D2H（src=设备 LINEAR 前槽，dst=host 前槽）
步骤 2  finish_worker(0, success=true)   → allocate_tail_shared 仍 miss ✅（test:2714-2716）
步骤 3  finish_worker(1, success=true)   → 可见：kv_cache_tokens_num==12、
        LINEAR 1 块、块 id==destination_id ✅（test:2717-2728）
```

tail=12 的构成手算（test:2659-2683）：

- 序列共 13 token（kTailTokens+1）；第一 chunk 按 stride=8 写 8 token（=2×kv_bs）
- 第二次 allocate 从 8 扩到 12、恰好多完成 1 个 kv_bs=4 的整块，第 13 个 token 留在 live 活槽不算完成
- 完成边界 = 8+4 = 12 = 3×kv_bs，恰停在块边界上，LINEAR 检查点快照就取在这个边界
- 而 LINEAR 背的是 GDN 循环状态——整段前缀一份快照、不按 token 切块（block.h:48-50 注释），一个检查点对应一个 LINEAR 槽位块（设备侧 [restore, live] 双槽的 restore 槽）——所以 12-token 检查点就是"LINEAR 1 块"（断言 restored_linear.size()==1，test:2717-2728）。

1 个 worker 回话≠完成——**可见性以"全 worker 齐+拷贝数对"为准**。

## 5.6 失败画面三段叙事

### 画面一：无盖键的白拷（Q-b "有数据没钥匙"）

```
正常：  配对时把源块键抄给目的块（:565/:618/:666-668）
        → D2H 拷贝"带着钥匙的内容"
        → 回调 insert(blocks) 已盖键版：token_hash_key.set(块上的键)（pc.cpp:194）
        → 登记进 host 缓存 map 的正确格子
反事实：去掉盖键
        → tail 块 D2H 到 host，键槽还是出厂垃圾值
        → insert 拿垃圾值当键登记（pc.cpp:194 照读不误）→ 幽灵条目
        → 未来请求按 hash 链算出真键（如 hashes[62]）来 find → map 里查无此键 → miss
        → host 那份状态永远查不到 → 白白拷了一份永远用不上的数据
        → 且无任何日志：预取判定 miss 是"静默的正常分支"
```

"host 侧需要键做去重对账"的准确含义：**host 缓存这本登记簿的索引就是键**。D2H 完成时登记在哪一格，由块上的键决定；未来预取查哪一格，由请求自己算的链键决定。键抄错=登记格和查询格永远对不上账。

### 画面二：EXPLICIT 缺失的半成品

```
正常：  host 复合 EXPLICIT → finish 时发布钩子早退（composite.cpp:339/:962）
        → host 块只被 D2H 回调 / 预取收尾发布
反事实：host 也用 AUTOMATIC
        → finish 步③ deallocate_for_sequence 顺路调 cache_for_sequence
        → 把 host 账本上的块 insert 进 host 缓存
        → 但配对过的块此刻还在 D2H 传输中（数据半截或全空）
        → 别的请求 probe 命中这个键 → mount → 恢复出半份线性状态
        → 整段折叠张量半新半旧，无一致性标记，静默错算
```

### 画面三：门禁缺失的半份张量

```
正常：  回调等 results.size()==expected_workers 且逐 worker 拷贝数对（:1171/:1188）
        → 全齐才 finalize_host_blocks(publish=true)
反事实：2 个 worker 回了 1 个就发布
        → worker 0 负责的张量分片已拷、worker 1 的还没拷
        → host 目的块"一半是数据、一半是空壳"
        → 发布进缓存 → 命中者恢复到混合状态 → 整块作废且无法检测
```

三段画面的公因子：**这三处不变量失效都不报错**——这正是它们必须靠机制（盖键抄写/模式开关/门禁计数）而非靠人肉纪律守住的原因。

## 5.7 mount 块的来源链闭环（Q-a）

### 5.7.1 "从哪来"：三条来源，一次汇合

mount（把 probe 命中的共享块装进序列账本）的块来源有三条，汇合点是 `allocate_shared`（hierarchy.cpp:854-892）：

```
来源①  设备 prefix cache 命中（块一直没被驱逐）        → 免搬运，直接 mount
来源②  host prefix cache 命中、比设备深                → H2D 补差额再 mount
来源③  Store 预取（第 0 层，入队前 prefetch_from_storage）
        → G2H 落 host → finalize 发布 → 回到来源②
```

`allocate_shared` 的探测顺序（代码顺序即优先级）：

```cpp
if (!hbm_state.prefix_cache_matched()) {
  composite->allocate_shared_for_sequence(sequence);      // :869 先设备
}
// ...
if (!host_state.prefix_cache_matched()) {
  host_manager->allocate_shared_for_sequence(sequence, host_state);  // :876 后 host
  host_state.set_prefix_cache_matched();
}
// ...
if (host_tokens > hbm_tokens) {                            // :886 host 更深
  sequence->set_host_cache_match(host_tokens,
                                 (host_tokens - hbm_tokens) / unit_size);
}
```

host 更深时，`collect_load_block_transfer_infos`（hierarchy.cpp:783-852）只拷差额：`begin_block = hbm_cached_tokens / block_size`（:828）——设备 8192/host 16384 → 只拷第 64..127 号块。LINEAR 的 H2D 有个细节：设备 restore 块的键从 host 源块抄（:815 `restore_block.set_hash_value(host_source->get_immutable_hash_value())`）——**回程也带钥匙**。

tail 命中的 mount 落点（composite.cpp:924-930）：

```cpp
kv_state.mount_composite_shared(probe.type, std::move(probe.blocks));
if (probe.logical_token_count != 0) {
  kv_state.set_num_cached_blocks(probe.type, 0);        // 游标清零
  kv_state.set_linear_state_tail_tokens(probe.logical_token_count);  // tail 便签
}
```

### 5.7.2 "用完去哪"：三结局

序列 finish 后块的三结局（与 §4.3 同源，此处只列 host 视角）：

- **留在缓存**：发布过的块 ref≥1，活到 LRU 驱逐。
- **回池待覆盖**：从未发布的私有块 ref→0，物理槽回空闲池。
- **D2H 下沉第二次生命**：跨层块经配对下沉，host 目的块发布进 host 缓存。

### 5.7.3 "被覆盖就用不成了吧"：对，且这正是缓存的预期生命周期

精确回答三层：

- 只要缓存还持着（ref≥1），块**不可能被覆盖**：LRU 驱逐只收 `is_shared()==false`（ref≤1）的节点（pc.cpp:249-253），命中过的块 ref≥2 被跳过。
- "被覆盖"只发生在：驱逐（map 条目抹键，pc.cpp:259-261）→ 最后别名析构 → 回池 → 再分配。**此时旧键已经从登记簿上抹掉，不存在"查到旧键拿到新数据"的错配**。
- 键是块上的字段：新主人分配后会盖自己的新键，互不干扰。

## 5.8 手算：一个 tail 状态块的三层旅程

设定（MR 级数字，与全书主线一致）：prompt=8191、hash_bs=128、stride=8192、tail=8064、tail 键下标 8064/128−1=**62**。块 id：设备槽 #77（发布后）、host 目的槽 #501。

```
【第 0 步】设备 HBM · forward 写到 8064
  介质：HBM    引用：#77=1（序列持）           键：无（活槽不盖键）
  LINEAR 双槽 [restore, live]，live 即将成为 tail 状态块
        │
        ▼
【第 1 步】设备 HBM · 发布（下一 chunk 准入触发）
  cache_completed_state 守门命中（cached==8064==tail，linear.cpp:391）
  → tail_checkpoint_hash 取键 hashes[62]（linear.cpp:355-362）
  → 盖键：source->back().set_hash_value（linear.cpp:440，
    注释"Stamp the source itself so D2H carries the same tail key"）
  → insert → 设备缓存登记
  介质：HBM    引用：#77=2（序列+设备缓存）      键：hashes[62] ✅已盖
        │
        ▼
【第 2 步】配对下沉（finish 步②，LINEAR prefill 分支 hierarchy.cpp:617-623）
  host 目的块 #501（空壳，ref=1）配对
  → 盖键：#501.set_hash_value(#77 的键)（:618，键=hashes[62] 抄过去）
  → pair 进 D2H 队列（src 拷贝 → #77=3）
  介质：HBM→队列  引用：#77=3 / #501=1（pair 持）  键：两块都是 hashes[62]
        │
        ▼
【第 3 步】finish 步③④+reset（hierarchy.cpp:505/:508/:509）
  介质：队列    引用：#77=2（设备缓存+pair）/ #501=1   键：不变
  host 账本掉引用对 EXPLICIT 早退：不发布（composite.cpp:339）
        │
        ▼
【第 4 步】发起 D2H（transfer_offload_blocks，hierarchy.cpp:1129-1161）
  BlockTransferInfo{src=#77, dst=#501, key=hashes[62], D2H2G}（:1140-1144）
  folly::collectAll 分发 host_transfer_worker_count 个 worker
  介质：传输中  引用：不变（move 进闭包）             键：不变
        │
        ▼
【第 5 步】完成回调（全 worker 齐 + 拷贝数对，hierarchy.cpp:1171/:1188）
  设备源释放：#77 deallocate+clear → 1（只剩设备缓存，LRU 里活着）
  finalize_host_blocks(publish=true)：insert(blocks) 已盖键版（pc.cpp:194 读 #501 的键）
  → #501: 1→2（闭包+host 缓存）→ 闭包析构 → 1
  介质：host DRAM  引用：#77=1 / #501=1（host 缓存持）  键：hashes[62]，登记在 host 缓存
        │
        ▼
【第 6 步】未来请求 B 入队前（prefetch_from_storage，hierarchy.cpp:985-1096）
  host 准入 probe：B 自己算 tail=8064 → 键 hashes[62] → find 命中（pc.cpp:224-231，LRU 刷新）
  → 命中即 mount：allocate_shared_for_sequence（hierarchy.cpp:1033）走 §5.7.1 同一机制
    （composite.cpp:924-930 的 mount_composite_shared），把 #501 的别名装进 B 的 host
    账本（Block 拷贝 +1）——这就是本步 #501 1→2 的动作主体
  → 取货单恒空（tail 短路，hierarchy.cpp:192-194）→ finalize 直接成功
  → final=base=8064（:1084-1087）；成功末尾置位 prefix_cache_matched（:350——此前
    事务曾故意复位它：host 层要等 Store 回调确认才算数，:1035-1038 注释）
  介质：host DRAM  引用：#501=2（host 缓存+B 的 host 账本）  键：命中验证 ✅
        │
        ▼
【第 7 步】B 正式调度（allocate → allocate_shared，hierarchy.cpp:854-892）
  设备 probe：miss（假设 #77 已被 LRU 驱逐）→ host 侧不再重探：第 6 步准入 probe 就是
    host 那一探，prefix_cache_matched 已在 finalize 成功时置位，:875-877 的 ！matched
    分支被短路，直接沿用第 6 步 mount 进账本的 tail 命中（预取失败才会 reset 后重探，:336-342）
  → collect_load_block_transfer_infos LINEAR 分支（:799-821）：
    设备 restore 块盖键 = host 源块的键（:815，回程也带钥匙）→ H2D 拷贝
  → mount 进设备账本 + tail 便签 8064（composite.cpp:926-929）
  → B 只 forward 127 token
  介质：HBM（回程后）  引用：#501≥2（host 侧持续）      键：旅程闭环，钥匙全程没离手
        │
        ▼
【第 8 步】B 用完 finish
  #501 的序列别名析构掉一份；只要 host 缓存还持（ref≥1），LRU 驱逐跳过（pc.cpp:249-253）
  "被覆盖"只在归零回池→再分配，且那时缓存条目已被驱逐抹键（pc.cpp:259-261）
  介质：host DRAM  引用：#501=1（host 缓存持）          键：hashes[62]，等下一位租客
```

小尺度代码锚点（同一机制的测试孪生）：`test:2657-2733`，kv_bs=4/stride=8/tail=12（构成手算见 §5.5.4：12=stride 8+kv_bs 4，恰为 3×kv_bs 的块边界），断言回调结算前 `allocate_tail_shared` 恒 miss、全 worker 完成后 host 命中 12 token 且块 id==destination_id。

## 5.9 章末自检表

| # | 自检问题 | 一句话答案 | 证据 |
|---|---|---|---|
| 1 | mount 的块从哪来？ | 设备缓存直取 / host 命中 H2D 差额 / Store 预取先落 host | hierarchy.cpp:854-892/:783-852/:985 |
| 2 | 用完去哪、被覆盖就用不成？ | 三结局；缓存持有期不可覆盖，回池前键已被驱逐抹掉 | pc.cpp:249-253/:259-261 |
| 3 | 盖键是什么？ | set_hash_value 把 16 字节 memcpy 进块上键槽，键跟数据走 | block.h:169-171 |
| 4 | 为什么 tail 块出厂无 hash？ | 键槽无构造初始化；键只盖"完成可复用边界"的快照，活槽不配钥匙 | block.h:199、linear.cpp:438-440 |
| 5 | "只给 chunk 边界做 hash"是失误吗？ | 不是：链上键处处有（每 128 token），块上键只盖边界块是设计 | block_utils.h:29-36、§5.2.5 |
| 6 | 白拷画面？ | 垃圾键登记成幽灵条目→未来真键 find 永远 miss→静默白拷 | pc.cpp:194/:224-231 |
| 7 | D2H 之后减引用对吗？ | 减引用=别名析构；配对/发起零减，结算点唯一=完成回调 | hierarchy.cpp:569-570/:1196-1197 |
| 8 | 异步/同步都减吗？ | 都减且同一时点（回调）；异步只是推迟 | hierarchy.cpp:1160-1163 |
| 9 | EXPLICIT 是什么？ | host 复合的发布模式：钩子早退+三入口白名单 | composite.h:77-80、composite.cpp:339/:962 |
| 10 | "host 显式 device 自动"还是"D2H 发过不再发"？ | 两个猜测各对一半，合起来=机制禁用+效果已发 | §5.3.1-5.3.4 |
| 11 | host 为什么必须显式？ | 已分配≠数据到位；空柜子 insert 进缓存=半成品命中 | linear.cpp:397-403 |
| 12 | 配对减引用吗？ | 零减：pair move 所有权搬家，注释亲口说 | hierarchy.cpp:569-570 |
| 13 | 取货单取什么？ | 只取 8192 整单元、从 base_units 起、KV 门禁 LINEAR 可选 | hierarchy.cpp:186-243 |
| 14 | tail 经取货单吗？ | 从不：tail 命中即空单（源块不占 Store 单元位） | hierarchy.cpp:192-194 |
| 15 | 整单作废判据？ | units 空或 gated_hits==units.size()；门禁缺一全作废 | hierarchy.cpp:1066-1068/:336-342 |
| 16 | D2H worker 门禁？ | worker 数齐 + 逐 worker 拷贝数==全部块数 | hierarchy.cpp:1171/:1188 |
| 17 | 回程 H2D 带钥匙吗？ | 带：设备 restore 块盖 host 源块的键 | hierarchy.cpp:815 |

## 5.10 核验声明与差异登记

**核验声明**：本章全部机制断言对 commit `commit-4` 逐段 `git show` 回读核验，关键贴码共 14 段（hierarchy.cpp 8 段、linear.cpp 3 段、composite 2 段、pc/bmi/block 各穿插）；两个测试锚点（`test:1839-1893`、`test:2657-2733`）全文通读。手算数字（8064/62/127、#201/#501 七行、G 例 12 token）均可在对应代码/测试中复算。

**与 factbase-v1 的差异登记**（以源码为准）：

1. factbase §6.2 称 `finalize_host_blocks hierarchy:44-63`——实际为 **hierarchy.cpp:39-59**（本章已用实际行号）。
2. factbase §6.2 称 "配对 hierarchy:557-575，注释 pair owns both blocks"——配对函数确从 :512 起，但 "pair owns both blocks" 注释实际在 **:569-570**，:557-558 是 decode 分支盖源块键。本章拆开引用。
3. factbase §6.2 称 G 例测试 "hierarchy:2658"——TEST 定义实际在 **test:2657**（`LinearTailHostOffloadBecomesVisibleAfterAllWorkersComplete`，:2657-2733）；另有一个同构的通用 offload 门禁测试在 **test:1839-1893**，本章一并纳入。
4. factbase §6.2 称 "三个显式发布入口：D2H 完成回调 / 预取收尾 / LINEAR trim 常规发布"——核验为 **cache_blocks 仅被 D2H 回调调用（hierarchy.cpp:55）**；预取收尾的两条发布走叶自身的 `trim_prefetch_blocks`（bmi.cpp:349-353 与 linear.cpp:315-319），不经 composite 的 cache_blocks。三入口结论成立，但路径细节如上修正。
5. factbase §6.2 六行引用表 "t2 配对 move 2/1"——核验 KV 分支 src 为**拷贝**（瞬间 #201=3），序列别名在 finish reset 时才析构回 2。本章 §5.4.2 扩展为七行，补上这一瞬态。
6. 新增 factbase 未展开的证据：EXPLICIT 模式实际只关掉**两个**序列驱动钩子（composite.cpp:339/:962），叶层 linear.cpp:401-403 是独立第二道防线；`BlockTransferInfo` 自带 16 字节键槽随传输走（model_input_params.h:691-707）。

**遗留待验证**（不承重）：host_transfer_worker 各自负责的张量分片具体切法（层维度推测，未见直接代码）；D2H 失败后是否有重试路径（当前代码只释放不重试）。



---

# 索引层（结论级速查）

## §索一 MR 总览与作者意图

**一句话总纲**：给「只能整段恢复」的线性状态（GLM-5.3 系 GDN linear attention），在每一个物理层级（HBM→host→Mooncake Store）钉死同一套「可恢复边界 + 边界键」，让 8191 这类不对齐 prompt 的前缀复用从「全部作废」变成「只重算最后一段」（8191 prompt 复用深度 0→8064，98.5%）。

**4-commit 演化**：

- **commit-1** feat 全量引入（25 文件 +864/−222，含 llm/vlm_engine 启动期校验）
- **commit-2** revert 撤 engine 校验（净 diff 于 distributed_runtime 为空；校验落 LinearStateBlockManager 构造期 `CHECK_EQ(chunk_stride % token_block_size, 0)` linear_state_block_manager.cpp:114-117——host LINEAR 叶在层级池直接构造 hierarchy:448，engine 层校验挡不住；此解释为教学推断非作者原话）
- **commit-3** bugfix 边界保持（21 文件 +563/−127：tail 贯穿 host admission/prefetch、hash 按 (hasher_type,stride) 分域、调度读构造期 stride；commit 自述在其开发环境通过 mtp_prefix_cache_test 173 / sequence_mtp_bootstrap_test 9 / scheduler_test 122）
- **commit-4** refactor 集中化 hash 映射（9 文件：`block_utils.h:29-36` 成为三消费方共用的单一事实源，删 `linear_state_hashes_` 独立链全套接口）

勘探意图验证 15/15 一致，无实质意图-代码偏差；2 条 [待验证]（stride 桥固定读 dp_rank 0 的隐式同构前提、测试通过数无法只读复跑）。

## §索二 设计评价汇总（四代书终裁 + v2 微调）

| 对象 | 结论 | 一句话理由 |
|---|---|---|
| MTP lookahead 进 hash | **好设计**（hash 链路线内） | 键覆盖=内容依赖的必然（复合材料+右移）；复用条件编码进键零运行时校验 |
| invalidate 域差异 | **好设计** | 参数化而非特判（一条循环各域按各自值回退）；回退量=覆盖量逆运算；欠债=新增域忘登记静默 0 |
| 双账本 | **合理妥协** | 量纲不同注定分账；挂载互斥有 composite:928 钉住；欠债=成长期并存无断言 |
| 三层+tail 短路"不敞亮" | **合理妥协，三七开** | 七分讲法三分实现欠债（短路散两处/回调切断控制流） |
| erase 全删 | **好设计** | 组级账本+单块元数据下沉 Block 的分工；欠债=描述-向量同步缺断言 |
| hash_type_ 成员变量 | **合理妥协** | 13 读点免传参 vs 隐藏可变状态（v2 实测 13 处，v1 记 12） |
| probe 即 LRU 刷新 | **合理妥协（行业常规）** | 浅命中浪费根因在跨叶对齐不在 LRU |

**元问题裁决**（"是巧妙还是傻逼/炫技还是不灵光"）：复杂度 9 项三分账——物理必然 5（end+1 / 发布晚一格 / fake 不进链 / 按域回退 / 剥 MTP）、路线税 2（域分桶选择器 / 全链 update-then-read 对齐）、欠债 2（无串联注释——用户三轮追问才得到答案的根因；键覆盖=内容依赖无结构化断言）。裁决：**深思熟虑的工程决策 + 文档欠债**分账，"乱打补丁"不成立、"不讲清楚"成立（v2 第二章冷读复核维持此裁决并补齐反事实四行表与参数化证据）。

## §索三 三方对照（xLLM / vLLM / SGLang）

| 维度 | xLLM | vLLM | SGLang |
|---|---|---|---|
| 键原料 | 链式摘要，MTP 域多吃 1 token | 父 hash+本块 token+extra_keys，无 lookahead | token 序列本身（radix 树） |
| 防撞键在哪层 | **键层**（烤进键） | **消费层**（丢边界块重算） | **结构层**（不同子节点无法构造同键） |
| draft KV 在共享块 | 在（复合材料） | 在（docstring 承认） | 不在（树只插确认 token） |
| 代价 | 三处对齐+读者不直观 | 每次命中少复用一块（无条件税） | 放弃 O(1) 增量链；树内存/锁 |
| 复用的东西 | target+draft 半边整体（**三家唯一**） | 到丢块为止 | 仅 target 侧 |

复用深度手算（A 33 token；B_match 全同 / B2 改 t_32）：xLLM **2 块/1 块**（实测）；vLLM **1 块/1 块**（丢块无条件税）；SGLang **全程/32 token** [推演待验证]。xLLM 有 tail、SGLang 无（SuffixReplay 论文视其为现有系统做不到的事）、vLLM 有近亲 partial tail [待验证]。细账见姊妹笔记第三章/第四章。

## §索四 Review findings（已发布 MR + 工程记录）

**已发布 6 条**（读回验证 6/6 逐字一致）：

- **P2×1**：构造期 `chunk_stride%token_block_size==0` 对齐 CHECK 使先前可运行配置启动即崩（kv_split=3 时逻辑块 384，8192%384=128≠0），报错不含实际值；建议 CHECK 补两值或向上取整
- **P3×3**：tail 快路径 CHECK 恒真自证断言缺 guarded-scenario rationale；`hash_block_size==0` 兜底三元三处复制；match() 降级路径退化为全链重算
- **Q×2**：mm 感知 hasher 改引擎级映射（可达性需作者确认）；性能契约三项可测成本无数字
- 总结论：**未发现 P0/P1**；建议 P2 修复后合入

**unverified 5 条（未发布）**：commit 声称测试数无法复跑；端到端无 benchmark；兄弟分支 废弃状态未确认；TEXT+mm 错配可达性；Mooncake 冷缓存未实测。

## §索五 学术脉络

**组内同学 的 linear-state prefix cache 血脉**（GitHub PR 时代 #1799（hash 原语）/#1802（选项）/#1846（block manager+prefix cache——`linear_state_hashes_` 独立链在此出生，本 MR 将其杀死）/#1940（恢复+容量）/#1947（batch+scheduler）/#1956（引擎接线）→#1979 VLM 启用+chunk-stride clamp→**#2161**（prefix-cache prefill attention 的 Python graph 路径修复——正是 glm5_next 唯一走的 TORCH 组图路径）→#2163；内部平台另有去 MTP 冗余省显存与本 MR 键域重做两步。其公开设计笔记：qwen35 原型、尾块缓存笔记（commit 29b510a8e）、ReplaySSM ring 审查。

**论文参照**（作者公开材料未引论文，两候选）：**SuffixReplay**（arXiv:2609.33477，TeleAI——重放派，不存检查点、留 anchor 重放近期后缀做近似重建；与 本 MR 检查点派相反；时间与问题意识高度吻合）；**Marconi**（arXiv:2411.19379，混合 LLM 前缀缓存开山，Bloom 准入）。SuffixReplay 称 SGLang 默认 8192 网格与当前 main 代码不一致 [待验证]。

## §索六 教学演化与勘误链 + v2 修正登记

**四代书演化**（元知识）：

| 代 | 行数 | 病 | 代表性裁决 |
|---|---|---|---|
| round1 | 2487 | **讲错了**（事实错误） | 5 处 hash 下标 63→62；probe 混入"按 block 回退"；finish≠erase_blocks；"退 1 token"缺 128 粒度前提 |
| round2 | 3968 | **没讲细**（跳步黑话） | 13 条裁决；黑话先定义后使用；每变量单独成段 |
| round3 | 2902 | **没讲透**（结论无实现） | 14 条裁决；引用结论必贴实现；先机制后类比 |
| round4 | 1262 | 挑战轮（物理层） | 3 处修正（复合材料+右移/静默吞吐事故/物理推进窗口但检查点只落确认边界）——三错同根因：**解释停在键层、物理层是空的** |

**v2 修正登记（v1 笔记 → v2 源码重验，十处）**：

| # | v1 口径（错/不准） | v2 裁定（源码证据） |
|---|---|---|
| 1 | "游标虚高 k=3，账面 24576 vs 实际 8064，错位 16448" | **k 恒 1、错位恒 128**：tail 挂载向量恒 1 块（composite:743-749 丢常规别名 + linear:260 CHECK_EQ 断言钉死），假账语义="已拥有第 1 个常规检查点(8192)"，8192−8064=128；16448 降级为通式代入示例（Ch3/Ch4 双章核验） |
| 2 | "(常规 3, tail 19968) 并存" | **常规 2 + 物理源 3 份**：prompt 20000 只有 2 个完整检查点 |
| 3 | TailMismatch E 例"tail 键查不到" | **find 实际命中**（测试 :846-850 显式插入 tail_hash、B 的 hashes[2] 同链同值），真失败在 trim 整块盖满条件（KV 2 块×4=8 ≠ tail 12）——"tail 保不住"的机制在 trim 不在 find（Ch4） |
| 4 | LRU 手算收尾 [N3,N4,N1,N2] | **[N3,N4,N2,N1]**（node_list 是 push_front 进 pop_front 出的 LIFO）；"浅命中块落最新端"结论不变（Ch3） |
| 5 | "MtpBlockHasher 只在 base commit 存在" | **base 与 head 都有**（block_hasher.cpp:31-49 一致）；真实出身 lookahead 原始 commit→#179 合入 commit（#179 合入）；当前 checkout HEAD 上游 main HEAD（上游另一谱系）**没有它**——本 MR 谱系与上游 main 并存两套键空间，引用"当前代码"必须钉死 commit（Ch2） |
| 6 | invalidate 调用点"commit_token:567" | 源码函数名 **update_last_step_token**（:526），invalidate 在 :570；"commit_token"是 v1 叙述性称呼（Ch2） |
| 7 | block_hashes() 读点 12 处 | **13 处**（block_manager_impl 2/composite 3/linear_state 4/sliding_window 2/hierarchy 2，git grep 实数）（Ch2） |
| 8 | "erase 全库 9 调用点" | **生产 8 + 测试 3**（两口径皆非 9）（Ch3） |
| 9 | 旧书"finish 走 erase_blocks(LINEAR)"（历代已勘误，v1 仍有残留表述） | finish 走五步 deallocate（hierarchy:491-510），全程无 erase_blocks（Ch3 §3.5 显式纠正） |
| 10 | （v2 流水线自纠）任务底稿示例"tail=9999−9999%8192" | 源码公式 **⌊prompt/128⌋×128**（完整 prompt 按 KV 块向下取整）= 9984；8192 是常规检查点位置非 tail（Ch1 章内勘误登记） |

一以贯之的教训：**教学验收标准必须是"读者能沿证据复算"，而非"技术上没说错"**；v2 补一句：**事实正确也救不了教学失败，冷读者的标准才是标准**。

## §索七 追问精华：四轮问题的原文与终答（检索索引）

> 用户原话困惑作小标题——半年后回看，"我当时为什么困惑"是最快的检索键。答案只给终态，细节见对应章。

### Round 2（10-12，七大问）

- **"prompt 只有 8192 个，恢复了怎么就错了？不管有没有 tail 都会有问题吧"** → 直觉对：8192 是整除边界天然无解，规则把它提前变成返回 0、走正确的全量重算（第一章五步链）
- **"linear_state_tail_tokens_ 皮之不存毛将焉附？finish 之后还重要吗"** → 便签随序列 reset 死；发布后活的是缓存条目（key→Block），未来序列自己重算 tail 键 find 命中（第三章）
- **"hash_type_ 固定的，双层桶难道只有一个桶吗"** → 是"最近选择器"非配置；无 MTP 1 桶、MTP 2 桶（LINEAR 特意剥 MTP 变体）；切台牌不删桶（第二章）
- **"invalidate 的 token_index−lookahead 太细节看不懂"** → MTP 块 i 的哈希多吃下一格首 token；回退量=覆盖量逆运算（第二章 token 8 例全程手算）
- **"盖键是什么？为什么块上可能没有 hash"** → set_hash_value 抄 16 字节；键只盖完成可复用边界的块是设计非失误（第五章）
- **"mount 是干嘛的，take_blocks 干什么的，块从哪来去哪"** → 四动作机械语义表；mount 用的是缓存共享块（别人还能用，ref 计数管着）；块三结局（第三章/第五章）
- **"游标不清零就算错了呢？不懂"** → mount 把游标设成向量长度，tail 挂载向量恒 1 块→游标 1 是假账"已拥有第 1 个检查点"；虚高的错位恒 128 token 复利放大（第三章 §3.4，v2 修正 #1）

### Round 3（10-13，十大问）

- **"为什么 prompt=8192 时 checkpoint_tokens==0？函数实现从没说过"** → 四路径纯函数逐行 + 双排除五步推导（第一章）
- **"常规链退了 token，tail 点查没退，8064 命中没 token forward 不还是有问题吗"** → **构造不出来**：tail 点查输入是纯函数输出，prompt=8064→tail==prompt→0→点查不发起；三重兜底（第一章）
- **"哪里讨论了这个分支是不是 tail"** → ProbeResult.logical_token_count 哨兵（非 0 即 tail 命中，全库唯一赋值 composite:751）（第三章）
- **"erase 是全删？要么不删要么删完，这么极端对吗"** → 数据结构决定（四 map 按 type 一条记录）；单块删减走 take→修剪→replace 三步流水线（第三章）
- **"EXPLICIT 是什么？device 自动 host 显式？"** → 同一枚硬币两面：模式开关+两钩子早退+叶层防线；host 块"已分配≠数据到位"防半成品（第五章）
- **"hash_type_ 当临时变量不行吗"** → 合理妥协：13 读点免传参（第二章）
- **"MTP 为什么这么特殊？hash 多算 token 块数却减，什么逻辑"** → draft 输入右移一位⇒槽位 j 依赖 t_{j+1}⇒键多吃一个 token 与格数减 lookahead 是同一物理依赖在覆盖侧/可用性侧的两次显影（第二章四步对账）
- **"区别对待是巧妙设计还是傻逼设计？炫技还是不灵光"** → 深思熟虑工程决策+文档欠债分账；"乱打补丁"不成立、"不讲清楚"成立（第二章诚实裁决）
- **"两个独立的账怎么统一用？checkpoint_index<=num_cached_blocks 用哪本"** → 常规账（量纲=检查点个数）；量纲不可通约故分账，四汇合点地图（第三章）
- **"假阳性白赚 LRU 头部不公平吧"** → 非假阳性（键真内容真）；浅命中块从没离开过 LRU；touch 语义行业常规（第三章）

### Round 4（挑战轮）

- **"前缀相同就该共享 block，凭什么管他妈下一个 token？"** → 直觉在 TEXT 域完全正确；MTP 块是复合材料，键区分下一 token 是如实区分两个内容不同的东西，不是立场（第二章）
- **"难道 vLLM 和 SGLang 也会 lookahead 吗"** → 都不做；vLLM 消费层丢块、SGLang 结构免疫（§索三）
- **"LINEAR 剥 MTP 像随便决策的人在掩饰"** → 递归状态不可能依赖未来（数学性质非实现选择）；四行反事实表：LINEAR 用 MTP 键=纯自残（第二章）
- **"不优雅、逻辑不硬、还不讲清楚，存在的意义是什么"** → 复杂度三分账；无串联注释正是三轮追问的根因（§索二）

## §索八 手算示例总表（参数→结果→所在章）

| 例 | 参数 | 结果 | 章 |
|---|---|---|---|
| A tail 边界 | 8191/128/8192 | tail=**8064** 有效 | 一 |
| A 同 | 8192 | 双条件→**0**（五步全排除链） | 一 |
| A 同 | 10000 | tail=**9984**（⌊10000/128⌋×128）vs 常规 8192，tail 深被选用，forward 16 | 一 |
| A 同 | 8193 | 规则二→0，常规链接管 | 一 |
| A KV-split | 16383/逻辑256 | **16128** | §索引 |
| B 四把钥匙 | hash_bs=256 | 常规 8192→31、16384→63；tail 7936→30、16128→62 | 二 |
| B 测试锚点 | hash_bs=128 | 8064/128−1=**62**（断言 63u 是块数非下标） | 二 |
| invalidate | 改 token 8，bs=4 | MTP stale=7→首脏块 **1**；TEXT stale=8→首脏块 2 | 二 |
| MTP 格数 | N=33/N=32，bs=16 | 两域同 2 格 / MTP 少 1 格（仅 N 整除 bs 时差 1） | 二 |
| 四行 A 表 | bs=16 改 t_16 | MTP 2 格全废/TEXT 1 格/LINEAR 保住/LINEAR 用 MTP 键纯自残 | 二 |
| B-Lifecycle | 块 77，prompt 8191 | ref 2→1（finish）→2（B 命中）→forward **127** | 三 |
| 游标假账 | tail 8064 挂载 | 游标报 1=假账"第 1 个检查点"，错位 8192−8064=**128**（v2 修正 #1） | 三 |
| 双账时间线 | 20000 成长期 | **常规 2 + 物理源 3 份** + tail 19968 并存（v2 修正 #2） | 三 |
| LRU 链 | 4 节点 | probe N1/N2→**[N3,N4,N2,N1]**（LIFO），trim 后纹丝不动（v2 修正 #4） | 三 |
| F-probe | prompt=20000 | 常规 hashes[63]/[127] 最深 16384 + tail hashes[155] 19968→trim 选 tail→forward 32 | 四 |
| 8191 主旅程 | 16 步 | probe→tail 点查 hashes[62]→mount 游标 0+便签 8064→base 8064→target=max→forward 127 | 四 |
| E 例 | bs=4/stride=8/prompt=13 | tail 键**实际命中**、真失败在 trim（8≠12 整块盖满）；游标不清零 | 四 |
| gap-tolerant | 20000 副旅程 | miss 放占位继续，"deeper hit subsumes earlier" | 四 |
| D2H 六/七行账 | #201 配 #501 | t0 2/1→配对 2/1（拷贝瞬态 +1）→回调成功 1/1 各缓存持；结算点唯一在回调 | 五 |
| G 例门禁 | 2 worker | finish_worker(0) 后仍 miss→(1) 后可见（tokens=12） | 五 |
| tail 三层旅程 | 8064/键 62 | 九步：device 发布→D2H 配对盖键→host EXPLICIT 发布→prefetch 拉回→mount→forward 127 | 五 |
| 三方复用深度 | B_match/B2 | xLLM 2 块/1 块；vLLM 1/1（无条件税）；SGLang 全程/32 [推演] | §索三 |

## §索九 来源与处置（v2 流水线声明 + 覆盖率表）

**v2 流水线**：历史讨论原文恢复 → 2A 五写手分章深潜（输入=v1 笔记事实底料+用户原话靶子+xllm 源码四 commit 链直读；贴码全部 `git show commit-4` 重摘，Ch1-Ch5 合计约 70 段）→ 2B 冷读验收四问硬门（Ch1 打回重写+二轮 PASS；Ch2-5 PASS+修复轮，共修 2 中高+约 20 中低）→ 2C 合并。v2 修正 v1 十处（§索六）。

**章节覆盖率**：

| v1 节 | v2 承接 | 深度 |
|---|---|---|
| §二 tail 检查点数学 | 第一章 | 深潜 |
| §三 键空间/MTP lookahead | 第二章 | 深潜 |
| §四 账本与生命周期 | 第三章 | 深潜 |
| §五 发布与准入 | 第四章 | 深潜 |
| §六 三层存储 prefetch | 第五章 | 深潜 |
| §一 总览 / §七 设计评价 / §八 三方对照 / §九 findings / §十 学术脉络 / §十一 勘误链 / §十二 追问 / §十三 手算表 | §索一~§索八 | 索引 |
| 四轮用户原话 | §索七 + 各章用户引子 | 全承接 |

**诚实声明**：①四本教学书在 v1 结算时已删（v1 纪律），v2 深潜层**不从书重排**而是从源码重教——教学编排（手算中间态/失败画面）为重建，事实与数字以 v1 笔记为线索、以源码核验为准（冲突按源码改，登记 §索六）；②冷读验收：五章全 PASS（Ch1 经一轮重写）；③v1 版保留在 git 历史，其"游标 k=3/16448""MtpBlockHasher 只在 base""E 例 tail 键查不到"等表述作废；④遗留 [待验证]（不承重，正文就地标注）：dp_rank0 同构前提、测试数复跑、SGLang 8192 网格、vLLM partial tail、撞键掉速幅度、三方复用深度推演、兄弟分支废弃状态、Ch5 worker 张量分片切法、D2H 失败无重试；⑤结算工作区待用户确认后清理。

---

## 附录：结算与版本记录（从卷首迁来）

> **结算来源**：四代教学书（round1 2487 / round2 3968 / round3 2902 / round4 1262 行，共 **10,619 行**）+ findings.md 审查轨（254 行）+ 本会话 4 轮追问讨论（含 10-12/10-13 两轮大追问）。
> **核验基准**：head `commit-4`（base `base-commit`，4 commits，34 文件 +1436/−277）。数值约定：KV 逻辑块 = hash 粒度 = **128**，LINEAR 检查点步距 chunk_stride = **8192**；hash 下标 0 基（8064/128−1=**62**）。

## 本版说明：为什么有 v2

本笔记 v1（345 行）与 vllm-sglang 卷 v1 同期产出、同病（索引文体：结论表多、贴码少、教学脚手架丢失）。按 v2 精编流程重做，但**原料不同**：四本教学书已在 v1 结算时删除（v1 纪律"吸收即删"——正是 v2 改掉的那条），所以 v2 的写作原料是三样：

- ① v1 笔记（345 行，作事实底料与线索）
- ② 历史讨论原文恢复
- ③ **xLLM 源码四 commit 链直读**（每处 file:line 回 `git show commit-4` 核验、贴码重新摘录——深潜层实际是"从源码重教的 本 MR"，不是原书的排版修复）

流水线：2A 五写手分章深潜 → 2B 冷读验收四问硬门（复述/零跳步跟算/先教后用/版面可扫读；Ch1 一轮打回重写后二轮过门，Ch2-5 PASS+缺陷修复轮）→ 2C 合并。**v2 重做修正 v1 十处**（登记于 §索六 v2 修正表），最重要的三处：游标虚高 k 恒 1 错位恒 128（v1 的 k=3/16448 回验不成立）、TailMismatch 例真失败在 trim 不在 find、MtpBlockHasher 在 本 MR 谱系 base/head 都在。

**两层产物**：深潜层五章（教学级，3,556 行 ≈ 原书 33%——目标读者=半年后没读过书的你）+ 索引层（§索一~§索九，结论级速查）。



> 新增卷首总览（冷启动入口）与各章「承上启下」过渡段；原卷首「本版说明/结算来源」（生产过程元叙事）迁至本附录；标题去掉生产元信息。治 v2 实测问题：冷读者打开笔记不知道这是什么、章序无弧线无过渡。
