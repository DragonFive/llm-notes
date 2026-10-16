---
title: SfaDcp:decode KV 分条注意力深潜
date: 2026-10-10
updated: 2026-10-16
tags:
  - type/lecture
  - domain/llm-inference
  - domain/systems
  - topic/context-parallel
  - topic/kv-split
  - topic/kv-cache
  - topic/attention
  - topic/npu
  - topic/pd-disagg
source_count: 3
status: stable
difficulty: advanced
---

# SfaDcp:decode KV 分条注意力深潜

> **一句话**:SfaDcpAttentionBackend 让 dcp 组内每个 rank 只存 KV latent 的 1/kv_split 条带,靠"广播小小的 query、寄回部分输出、LSE 精确合并"买回全量注意力——**KV 从不跨 rank 移动,INDEX 全量复制让"去哪找"零通信**。

## 卷首 · 全局导航

### 这页是什么

SfaDcp(decode 上下文并行)三轮教学(5 问 + 9 问 + 1 问 4 子)的结算笔记,**v2 规格重建版**:

- **深潜层**(第 1-3 章,2486 行,教学级):每个机制 = 主干叙述 + 贴码 3-8 行 + 手算全程(带中间状态,能跟着算)+ 失败画面三段(怎么错→错在哪→下游什么表现)。覆盖三轮用户追问过的全部 23 个机制——追问频率就是深度投票。
- **索引层**(卷尾,结论级):工程决策、证据边界、源码锚点,服务检索不服务学习。
- **前史**:2026-10-10 首次结算(253 行索引文体:判决表+一句话结论,0 贴码、手算只剩结论数字)→ 2026-10-13 重结算(修复虚假吸收:三轮书 6331 行经子代理逐本全文通读、两会话 466 条用户消息考古,补「讨论独有沉淀」与覆盖率表)→ 本版 2026-10-16 按 note-settle v2 深潜规格重建:深潜层从原料直读重写(一轮书自 git 历史恢复 2707 行 + 本会话考古 2925 行 + 源码 18 文件重新物化,全部行号实读核验),索引层全量继承 10-13 版沉淀。**冷读验收 23/23 通过**(22 首过 + 1 修复复验,记录见卷尾)。

### 读法

按序读(第 1 章坐标空间 → 第 2 章 indexer/mask/remap → 第 3 章交换融合/图 padding),或按「追问精华索引」直奔你的困惑;每个机制小节标题标注了它回答的是哪一轮的哪个问题(r1-Q4 / r2-Q7a / r3-(c) 式镜像编号)。统一手算数字全页共用:**page=4,dcp=2,逻辑块表 [5,7],13 token,位置 13**。

### 勘误链(四处,读旧材料以本页为准)

1. 一轮书 3.4:位置 13 被当成槽 13(三空间混淆)——二轮书已纠,本页第 1 章;
2. 二轮书:attention_mask 对 padding 行为 False——实际恒全 1,清 -1 靠 span 剥离+F.pad+validity,本页第 3 章 D5.3;
3. 二轮书手算例 C 的 [0,2,3,-1]——topk=3 恰好选中 3 个时真实行是 [0,2,3] 宽 3 无 -1(-1 是行宽示意),本页第 2 章 D2.3;
4. 二轮报告:dcp 组=TP 域内连续 kv_split 个 rank——只在 kv==tp 时字面成立,kv<tp 时组员按 dcp_stride=tp/kv 交错取样,本页第 3 章 D4.4 深水区。

## 追问精华索引(三轮 15 问 → 深潜落点)

| 轮 | 你的原话(压缩) | 判决一句话 | 落点 |
|---|---|---|---|
| r1-Q1 | indexer query 被切 TP 吗 | 不切,三投影全复制;"64 头+allreduce"是主注意力与 GSIZE 垫零的叠影 | D2.1 |
| r1-Q2 | latent block id 到 INDEX page id 是兑换公式吗 | 是:INDEX_page=E×dcp+s | D1.3 |
| r1-Q3 | 打包与 LSE 融合怎么工作 | 键全唯一保原序;LSE 三行代数精确合并 | D3.2/D4.5 |
| r1-Q4 | 全局槽/本地槽/INDEX 几何 | 四层坐标空间;三张表三消费者 | D1.1/D1.4 |
| r1-Q5 | 图模式 padding 用 0/-1 真执行吗 | 真执行,三件套,页 0..dcp-1 是保留块合法展开 | D5.1/D5.2 |
| r2-Q1 | 对 glm5.2"TP+allreduce sum"成立吗 | 不成立,glm5_2 indexer 32 头复制零归约 | D2.1 |
| r2-Q2 | kPool mask 是什么 | 每步 topk 稀疏白名单,复制计算零通信 | D2.2/D2.3 |
| r2-Q3 | E×dcp+s 兑换 | latent 每块 1 页 vs INDEX 每块 dcp 页 | D1.3 |
| r2-Q4 | 两个"5"从哪来 | 槽偏移 o(E 基)与读侧 L(b 基)的 b=s 巧合;位置 21 分道 | D1.5 |
| r2-Q5 | 高位是 E 不是 b?b 有什么用 | b=序数(查表列号),E=门牌(槽基);桥=block_table[b]=E | D1.2 |
| r2-Q6 | 必选尾是配置项吗,fill/copy/scatter 干什么 | 三配置合取;尾列 scatter 到 owned_prefix 后续进游程 | D3.3 |
| r2-Q7a | 4 个 bf16 装 fp32? | 手工浮点再编码(非裸位拆),格0=符号指数码兼失效旗 | D4.3 |
| r2-Q7b | AllToAll 怎么还在分头 | gather 轴=头维,借头还头,token 不动 | D4.1/D4.4 |
| r2-Q8 | padding 行真执行?attention_mask? | 真执行;mask 恒全 1(勘误 2),清 -1 三机制 | D5.1/D5.3 |
| r3 | mask 存序号吗/换本地 slot 吗/有效前置吗/怎么 remap | 存 0 基 kv 位置;终点是读侧 L 非写槽;三阶段前置;p→b→owner→L | D2.3/D3.1/D3.2 |

## 术语速查(跨章共享词汇,首现前可回查)

| 术语 | 一句话 |
|---|---|
| 条带 stripe | dcp 组内一个 rank 负责的那份 KV 分片(每逻辑块 1 页) |
| dcp 组 | 一起凑齐完整 KV 的 kv_split 个 rank;tp2/kv2 下=整个 TP 域(kv<tp 时组员交错取样,见 D4.4) |
| E / b | 块的门牌(块表值,槽编码基)/ 块的序数(位置除出,查表列号);桥=block_table[b]=E |
| L / owned / peer | 读侧本地索引(b 基)/ 本 rank 拥有的条带页 / 组内其他 rank |
| H_local | 每 rank 拥有的注意力头数=H//dcp(头维切片的头份额) |
| INDEX / latent cache | DSA 检索目录(每逻辑块 dcp 页)/ MLA 压缩 KV(每逻辑块每 rank 1 页) |
| mask / top-k | indexer 每步选出的稀疏白名单(存 0 基 kv 位置,-1=无效) |
| kPool / 必选尾 | 池化历史 + 最近 token 强制入选的尾部列(三配置合取才开) |
| AllToAll / 借头还头 | 组内按目的 rank 等分交换;替别人算的头打包寄回、收回自己的头 |
| LSE | log-sum-exp,softmax 归一化因子的对数,合并两半注意力的精确记账 |
| ACL 图 / 三件套 | NPU 静态图整段重放;padding 行统一喂 块 0/槽 -1/长 1 |
| DSA | DeepSeek-style Sparse Attention,GLM-5.3 的稀疏注意力层(indexer 选位置) |

---
# D1 · 坐标空间与三张表

> 深潜层章节。覆盖六条机制,编号镜像三轮用户问题号:
> 四层坐标空间(r1-Q4 / r3-(d))· b 与 E(r2-Q5)· 兑换公式(r1-Q2 / r2-Q3)·
> 三张表三消费者(r1-Q4 / r2-Q4)· 两个"5"与位置 21(r2-Q4 / r3-(d))·
> localize_slots 两调用点两输入基(r1-Q4 / r3-(b)(d))。
>
> 读者画像:半年后没读过教学书的你。所有概念首现即教,本文自闭环。
> 贴码行号全部对照本地物化 src/ 实读核验(分支 feat/glm5-next-sfa-dcp-decode 快照,见 PROVENANCE.md)。
> 二轮/三轮教学书已删,其判决与手算数字以 session-extracts 成员报告为准。

## 0. 开篇:一个 token,四个名字

### 0.1 先卸下五个包袱(每条 ≤30 字)

- 同一个 token 有四个编号,不是一个。13/61/29/5 说的都是它。
- 位置 13 不是全局槽。槽是 61,调度器按块表值编出来的。
- 读侧索引 5 不是写槽。写槽是 29,两者差一次块表查询。
- 旧材料把位置/槽/写槽混成一团,错了还不自知(恒等表特例,§2.4)。
- INDEX 写不走任何一张表。它拿全局槽 61 直通行号。

下面全文只用一套数字,先钉死:

```
page = 4        # 一页存 4 行,一行一个 token
dcp   = 2        # KV 条带切分度:一个逻辑块劈成 2 个条带,分住 2 张卡
块表   = [5, 7]    # 第 0/1 个逻辑块,物理页门牌是 5 和 7
序列   = 13 个已有 token,正在 decode 第 14 个(位置 13)
```

### 0.2 前置概念(首现即教,缺一个后面就卡)

**token 与位置**:模型一次处理一个词元(token)。位置是它在序列里的 0 基序号:第 1 个 token 位置 0,第 14 个位置 13。

**页(page)与行**:缓存池按页存放。一页 4 行,一行存一个 token 的数据。行是池子里最小的格子。

**逻辑块**:连续 8 个 token 打包成一个管理单元。为什么是 8?4(page)×2(dcp),下面讲条带时自然掉出来。

**块表(block table)**:一张小表,第 b 个逻辑块住在哪个物理页。本例 `[5, 7]`:块 0 住页 5,块 1 住页 7。b 是"第几个块",表里的值是"门牌号"。这两者的区分是本章主角之一(§2)。

**latent 池**:MLA 结构的注意力不存原始 K/V,存压缩向量。这些压缩 KV 住的池子叫 latent 池。decode 每步把新 token 的压缩 KV 写进去。

**INDEX 池**:另一个池子。稀疏注意力的"打分器"(indexer,索引器)要给每个历史 token 算一个键摘要,这些摘要单独存成一个池。打分前必须全量读它。

**SFA(稀疏快速注意力核)**:真正算注意力的 NPU 算子。它拿打分器选出的位置清单,去 latent 池取数计算。

**DCP、条带(stripe)与 rank**:decode 阶段的上下文并行。序列的 KV 按"条带"切到多张卡:逻辑块的 8 个 token,前 4 个(整页)归 rank0,后 4 个归 rank1。dcp=2 就是两张卡分摊。切的是 token 轴,不是注意力头。

**调度器与槽(slot)**:框架里发号的账房。它每步给新 token 发一个"床位号"(全局槽),告诉各家池子这一行写到哪。槽怎么编出来的,§1.2 展开。

### 0.3 概念地图:先看对象,再记名字

先不记任何函数名。看这幅布局图——三个池子怎么摆:

```
序列:    位置  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13(新)
         逻辑块 b=0(页5)  |    b=1(页7)
         条带 s=0→rank0   s=1→rank1   s=0→rank0    s=1→rank1

rank0 latent 池(只存条带0): 页5 行0-3 = 位0-3; 页7 行0-3 = 位8-11
rank1 latent 池(只存条带1): 页5 行0-3 = 位4-7; 页7 行0-3 = 位12-15
INDEX 池(两 rank 各一份全量):
         页组5 = [10,11](存位0-7); 页组7 = [14,15](存位8-15)
```

图里有三条差异,全章都在解释它们:

1. **latent 按条带切,INDEX 不切**。块 1 的 latent,rank0 只存条带 0 的 4 行,rank1 只存条带 1 的;INDEX 则两 rank 各存全部 8 行。
2. **latent 的页号 = 块表值**。块 1 的 latent 是页 7——两个 rank 都叫页 7,各存自家条带。
3. **INDEX 的页号 = 块表值×dcp+条带**。块 1 的 INDEX 是页组 [14,15],条带 0 用页 14,条带 1 用页 15。

再补一件小事:SFA 读 latent 时要按"本 rank 视角"数长度。13 个 token 拆开,rank0 看到 8、rank1 看到 5(整块 8 个 = 各 4 个,余 5 个 = rank0 拿 4、rank1 拿 1)。

### 0.4 全章主线:新 token 进村,四条路

预告主线——新 token(位置 13)进村,调度器发全局槽 61,然后四条路:

```
新 token(位置 13),调度器发全局槽 61
   ├── localize ──→ local_slots(rank0=-1, rank1=29) ──→ ① 写 latent(只写自家条带)
   ├── 原样 ──────→ 逻辑表 [5,7] ─────────────────→ ② SFA 读 latent
   ├── expand ────→ expanded [10,11,14,15] ────────→ ③ indexer 读写 INDEX
   └── 直通 ──────→ 全局槽 61 ───────────────────→ ④ 写 INDEX(第 61 行)
```

①③④是三张表/一个直通量的故事(§4);②牵出读侧坐标(§1.4、§6);61 为什么恰好是 INDEX 的行号,是 §3 的恒等式。

那张图里还有一个数你没见过:读索引 5。别急,§1.4 专门讲它。出发。

## 1. 四层坐标空间(r1-Q4 · r3-(d))

### 1.0 总表:四层空间,一个 token

先摆总账(编码公式里的符号在下面各节首现即教):

| 层 | 例值 | 编码公式 | 谁生产 | 谁消费 |
|---|---|---|---|---|
| 位置 p | 13 | p = b×(page×dcp) + o | 序列本身 | indexer 打分、读侧 remap |
| 全局槽 s_g | 61 | s_g = E×(page×dcp) + o | 调度器 slot_mapping | latent 写(经 localize)、INDEX 写(直通) |
| 本地写槽 | 29 | E×page + f | localize_slots(槽 61) | latent 写核(rank1) |
| 读侧索引 L | 5 | L = b×page + f | localize_slots(位置 13) | SFA 查块表读 latent(rank1) |

四个数、一个格。分清"哪个数在哪个空间",是本章第一刀。

### 1.1 位置 13:序列坐标

位置是最朴素的坐标:这个 token 在序列里排第 14 个,0 基记法就是 13。

它由序列本身决定,谁都不用算。它的消费者有两个:

- 打分器选 top-k 时,选出来的就是位置号(存进 mask 的是它,不是槽——r3-(a) 判决,§6 会用到);
- 读侧 remap 拿位置当输入(§6)。

位置怎么拆进块?一步除法:

```
b = 13 // (page×dcp) = 13 // 8 = 1      ← 第 1 个逻辑块
o = 13 % 8 = 5                          ← 块内偏移(第 6 个)
```

b 是"序数"(第几个块),o 再拆两步:s = o//page = 5//4 = 1(条带 1,归 rank1),f = o%page = 5%4 = 1(条带内第 2 行)。

记牢这四个小分量,后面所有公式都由它们拼出来:b=1、o=5、s=1、f=1。

### 1.2 全局槽 61:调度器发的床位号

调度器(框架的账房)每步给新 token 发一个全局槽。它不是位置,是"物理资源的行号"——按块表值编码:

```
全局槽 s_g = E×(page×dcp) + s×page + f
          = 7×8 + 1×4 + 1
          = 56 + 4 + 1 = 61
```

逐项读:E×8 跳过前面 7 个逻辑块;s×4 跳到块内条带 1 的起点;+f 到条带内第 2 个。

这里冒出一个新符号 **E**:块表值,块 1 的门牌 7(block_table[1]=7)。为什么高位是 E 不是 b?——因为槽要直接定位物理资源,门牌才是物理地址。§2 用整节讲这个区分。

反着除回去验证:

```
61 // 8 = 7   → 得到 E(不是 b!)
61 % 8 = 5    → 得到 o
```

注意:除出来的是 E=7。想从 61 得到 b=1,除法做不到——这是 §2.4 结构性证明的伏笔。

全局槽的消费者:latent 写(先经 localize 降基,§1.3)和 INDEX 写(直通行号,§4.6)。

### 1.3 本地写槽 29:rank1 的 latent 地址

latent 池按条带切了,rank1 只存自家条带。写核要的地址是"本 rank 池子里的页号+行":

```
本地写槽 = E×page + f = 7×4 + 1 = 29     → latent 页 7 行 1(rank1)
```

页号还是 7?对——**本地页=块表值**:每个 rank 的 latent 池都保留了 E 这个门牌,只是每页只装自家条带的 4 行(全局视角同一门牌装 8 行)。

rank0 呢?条带归属判定说这块不归它,它拿到的是 -1(无效标记,写核看到就跳过)。

这个降基算术的作者是 `localize_slots`。先看它的宿主(纯算术对象,不持有任何张量):

```python
class KVShardLayout:
    """Maps a logical paged-KV coordinate onto one rank's physical cache."""

    INVALID_SLOT = -1
    ...
    @property
    def logical_block_size(self) -> int:
        return self.physical_block_size * self.dcp_size
```
(kv_shard_layout.py:31-56,节选)

三个数:物理页大小 4、dcp 组大小 2、本 rank 的条带编号。`logical_block_size = 4×2 = 8`,即一个逻辑块的 token 数。SfaDcp 后端绑缓存时创建,每 rank 一个、带着自家编号。

函数本体(第一块,判归属):

```python
    def localize_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        valid_slots = logical_slots >= 0
        safe_slots = logical_slots.clamp_min(0)
        logical_offsets = torch.remainder(safe_slots, self.logical_block_size)
        owner_ranks = torch.div(
            logical_offsets,
            self.physical_block_size,
            rounding_mode="floor",
        )
        owned_slots = valid_slots & (owner_ranks == self.dcp_rank)
```
(kv_shard_layout.py:79-88)

逐行讲:输入 `logical_slots` 是槽表(本例 [61])。`valid/safe` 先把图模式 padding 的 -1 夹住,不让它参与取余。`logical_offsets = 61%8 = 5`,块内偏移。`owner_ranks = 5//4 = 1`,条带 1 归 rank1——**条带按块内偏移区间划分,不按块划分**。

第二块(算本地槽):

```python
        logical_block_ids = torch.div(
            safe_slots, self.logical_block_size, rounding_mode="floor",
        )
        local_offsets = torch.remainder(logical_offsets, self.physical_block_size)
        local_slots = logical_block_ids * self.physical_block_size + local_offsets
        return torch.where(
            owned_slots, local_slots,
            torch.full_like(local_slots, self.INVALID_SLOT),
        )
```
(kv_shard_layout.py:89-100)

逐行讲:`logical_block_ids = 61//8 = 7`——注意!这不是 b=1,是 E=7。槽 61 本来就是 E×8+5 编出来的,除回去自然得 E。`local_slots = 7×4+(5%4) = 29`。不归自家的格子,`torch.where` 填 -1。

手算全程(位置 13 的写侧):

```
输入:  全局槽 61
第 1 步: 61 % 8 = 5          → 块内偏移 o
第 2 步: 5 // 4 = 1          → owner=rank1;rank1 认领,rank0 弃
第 3 步: 61 // 8 = 7         → E
第 4 步: 7×4 + (5%4) = 29    → 本地页 7、行 1
结果:  rank1 得 29;rank0 得 -1
```

### 1.4 读侧索引 L=5:SFA 侧的本地行号

读是另一回事。SFA 在 rank1 上读 latent,它要的索引叫 L:

```
L = b×page + f = 1×4 + 1 = 5     (rank1;rank0 对这条得到 -1)
```

高位是 b 不是 E。SFA 拿到 L 后自己拆:页列 = L//4 = 1,行 = L%4 = 1,再查块表 block_table[1]=7,落到 latent 页 7 行 1。

L 怎么来的:打分器选出的位置(13)喂进同一个 `localize_slots`——同函数、不同输入、不同基。这是 §6 的主角,这里先记住结果。

### 1.5 速记:L 与本地写槽,差一次块表查询

读写同格验证(三个空间靠块表桥接的实锤):

```
写:rank1 写 latent 页 7 行 1(写槽 29)
读:L=5 → 页列 1 → 查 block_table[1]=7 → 行 1 → latent 页 7 行 1
✓ 同一个格子
```

写槽 29 ≠ 读索引 5,但经块表桥接指向同一格。速记一句话:

> **L 与本地写槽差一次块表查询:把 L 的页列 L//4 查表换成 E,再加回行,就是写槽。**

```
写槽 = block_table[L//4] × page + L%4 = 7×4+1 = 29
```

### 1.6 失败画面:写核拿全局槽当本地槽

**怎么错**:有人图省事,把 localize 这步删了,写核直接吃全局槽 61。

**错在哪**:61 是 8 行制的编码,本地池是 4 行制。写核按 4 行制拆它:61//4=15、61%4=1,认定"页 15 行 1"。

**下游表现**:页 15 属于别的逻辑块(块表值为 15 的块,若已分配)——这个 token 的 KV 污染别人家的块,SFA 下次读那块拿到错数据,**静默错算**;若页 15 未分配,越界写,内存损坏。

## 2. b 与 E:序数与门牌(r2-Q5)

### 2.0 你当时的困惑(r2-Q5 原话)

> "为什么高位是 E 不是 b……这里的逻辑块 1 是什么意思,怎么感觉 E=7 是逻辑块的意思呢,既然你说'高位是 E 不是 b',那要 b 有什么用,这个 b=1 从哪里来"

一句话判决:**b 是序数,E 是门牌;b 从位置除出来,E 从块表查出来;桥就是 block_table[b]=E。**

### 2.1 b=序数:位置除出来的查表列号

b 回答"这是序列里第几个逻辑块"。它不需要任何表,一步除法就有:

```
b = 13 // 8 = 1
```

b 的用处恰恰在"查表列号"这四个字:**没有 b,就没有 block_table[b] 这次查询**。读侧 SFA 拿到的稀疏索引,页字段就是 b;L=b×4+f 算出本地行,页列 L//4 还是 b。整条读链都活在 b 基的世界里。

### 2.2 E=门牌:块表值,写侧槽的编码基

E 回答"这个块物理上住在哪页"。它不在序列里,在块表里:

```
E = block_table[b] = block_table[1] = 7
```

门牌是分配出来的。调度器从物理池里挑空闲页给序列,挑中哪页是运行时的自由——本例挑了 5 和 7,也可能是 0 和 1(那时 E=b,见 §2.4 特例),也可能是任何别的页号。

写侧活在 E 基的世界里:全局槽 s_g = E×8+o,本地写槽 = E×4+f。高位全是门牌。

### 2.3 桥:一次查询,两个世界

```
        位置的世界(b 基)                物理的世界(E 基)
        b = 13//8 = 1      ──查表──→   E = block_table[1] = 7
        L = b×4+f = 5                    写槽 = E×4+f = 29
```

方向只有 b→E 是免费的(查表即得)。反方向 E→b 要扫整张表找"谁的门牌是 7"——块表没存逆映射。

### 2.4 恒等表特例:旧材料为什么"自洽地错"

如果块表恰好是 [0,1](E=b,单元测试里常见),三个空间的数值全部重合:

```
位置 13 → 槽 = 1×8+5 = 13(数值 = 位置)
写槽 = 1×4+1 = 5,读索引 = 1×4+1 = 5(写读同数)
```

旧教学材料的数字("槽 13、写槽 5")在这个特例下全对。但真实调度器的块表是分配出来的([5,7]),不恒等——**把特例当通例,于是自洽地错**。这也是当时两遍学不透的隐性障碍:材料里的数字互相矛盾,却都"看起来对"。

### 2.5 结构性证明:从槽只能除出 E,除不出 b

命题:给定全局槽 61 和除法/取余,能恢复 E,永远恢复不了 b。

证明分两口气。

**第一口气:能恢复 E。**槽的编码是 s_g = E×8+o,且 0≤o<8。这是标准的"高位×基数+低位"编码,除以基数 8 商就是 E、余就是 o:

```
61 // 8 = 7 = E    ✓
```

**第二口气:恢复不了 b。**b 的信息根本不在槽里。编码时用的门牌是 E,从 61 里无论怎么 div/mod,挖出来的都是 7 的各种变形。想得到 b=1,必须回答"块表里哪个下标的值是 7"——也就是求 block_table⁻¹(7)。

而块表是一张运行时分配的普通数组,没存逆映射。同一个 E=7,在序列 A 的块表里下标是 1,换个分配方案下标可以是任何数。**槽里没有的信息,除法除不出来。∎**

结构图:

```
位置 13 ──(//8)──→ b=1 ──(查表)──→ E=7 ──(×8+o)──→ 槽 61
   ↑__________________ 没有这条边 __________________↑
              (槽 → b 需要 block_table⁻¹,不存在)
```

### 2.6 失败画面:如果调度器用 b 编码槽

**怎么错**:假想调度器发槽改成 s = b×8+o = 1×8+5 = 13(拿序数当门牌)。

**错在哪(两处)**:

第一处,槽撞号。b 是序列内序数:序列 A 的块 1 和序列 B 的块 1 都编出槽 13。批处理里两条序列的新 token 抢同一个"床位"——INDEX 第 13 行被交叉覆盖,跨序列污染。

第二处,写读分家。INDEX 写侧直通行号(§4.6):写进第 13 行。但读侧按 E×dcp+s 的几何寻址(§3),位置 13 该落的行是 E×8+o = 61。写进去的 13 行没人读(孤儿行),读的 61 行没人写(旧值或垃圾)。

**下游表现**:indexer 拿错键摘要打分 → topk 选错 token → 输出错但无报错。同类的"槽必须用物理门牌编码"约束,也解释了为什么 C++ 侧的孪生函数同样以"槽高位"为页号(§6.4)。

## 3. 兑换公式 INDEX_page = E×dcp+s(r1-Q2 · r2-Q3)

### 3.0 你当时的困惑(r2-Q3 原话)

> "所以这里的意思是存在一个兑换公式,从 latent 的 block id(page id)到 indexer cache 的 page id 对应关系,就是在开启 dcp 之后,就不能用同一个 id 来表示不同 cache 的块了"

判决:对。兑换公式就是 **INDEX_page = E×dcp+s**。开 dcp 后 latent 页号和 INDEX 页号是两套编号空间,同 id 不同义。

### 3.1 空间差:每逻辑块 1 页 vs 每逻辑块 dcp 页

对照 §0.3 布局图的两个池子:

- latent:块 1 的条带数据住在**页 7**——每逻辑块 1 页(每 rank 自家那条带);
- INDEX:块 1 的摘要住在**页组 [14,15]**——每逻辑块 dcp=2 页,连号。

页组编码:

```
INDEX 物理页 P = E×dcp + s
             = 7×2 + 1 = 15      ← 块 1 条带 1 的摘要页
```

组内第 s 页专门存条带 s。E=5 的块同理:[10,11]。

### 3.2 兑换的实现:expand 一张表

`expand_indexer_block_table` 把逻辑块表 [5,7] 展开成 INDEX 页表:

```python
        shard_offsets = torch.arange(
            self.dcp_size,
            dtype=logical_block_table.dtype,
            device=logical_block_table.device,
        )
        expanded = logical_block_table.unsqueeze(-1) * self.dcp_size + shard_offsets
        expanded = torch.where(
            logical_block_table.unsqueeze(-1) >= 0,
            expanded,
            torch.full_like(expanded, -1),
        )
        return expanded.flatten(start_dim=1).contiguous()
```
(kv_shard_layout.py:121-138)

变量溯源:`logical_block_table` 是引擎块表的 int32 视图(逻辑表 [5,7]),SfaDcp 后端在 prepare 里转好。`shard_offsets = [0,1]`,dcp 个条带的编号。

逐行讲:`unsqueeze(-1)` 把每行变列向量,乘 dcp 加 [0,1]——每个门牌 E 变成 dcp 个连号页 `E×2+[0,1]`。负值保护:块表里的 -1(未分配)展开仍是 -1,不产生假页号。`flatten` 把 [[10,11],[14,15]] 摊平。

手算:

```
输入:  [5, 7]
第 1 步: 5×2+[0,1] = [10,11]
第 2 步: 7×2+[0,1] = [14,15]
结果:  [10, 11, 14, 15]
```

单元测试钉死了这个算术(含 -1 展开成 dcp 个 -1):`expand(3,7,-1) == 6,7,14,15,-1,-1`(测试文件未随 src/ 物化,转引二轮勘探报告核验)。

### 3.3 为什么必须多配页:动机链四步(先动机后公式)

**第 1 步:INDEX 必须全量读。**打分器每步 decode 都要对全部历史 token 打分选 top-k。打分 = 读每个 token 的 INDEX 行。

痛点画面:如果本 rank 只有半个序列的 INDEX 行,每步打分前都得跨 rank 拉另一半。decode 一步一个 token,每步都付通信,延迟直接吃进时延。

**第 2 步:复制比通信便宜。**既然每步都要全量,不如每个 rank 常备一份完整 INDEX——复制一次,读零通信。

**第 3 步:每 rank 都要写全部条带。**全量副本不会自己长出来。默认的 replicated 写模式:每个 rank 自己把全部条带的行都写一遍(反正要算的键摘要大家一样,复制计算零通信)。

推论:每个 rank 的 INDEX 池,必须为"每个 latent 条带"都留一页。每个逻辑块有几个条带?dcp 个。**INDEX 页数 = latent 页数 × dcp。**

**第 4 步:页组连号,一页对一个条带。**为什么不交错、不哈希?因为连号+对位能让"槽号=行号"(§3.4 恒等式),写读两侧都不用重排。

页数的官方推导链,三件套:

```cpp
// Physical index pages owned by one logical cache block. The NPU keeps the
// whole DSA indexer cache on every rank, so a logical block B maps to index
// pages [B * split, (B + 1) * split) (see platform.h). ...
inline int64_t indexer_pages_per_block() {
  if (!Platform::requires_dsa_indexer_cache_replication()) {
    return 1;
  }
  return std::max<int32_t>(util::kv_split_size_effective(), 1);
}
```
(cache_layout_builder.h:35-49,节选)

逐块讲:开头注释就是第 3、4 步的结论——NPU 每 rank 保全量 INDEX,逻辑块 B 的页是 [B×split,(B+1)×split)。函数体:NPU 平台返回 kv_split(=dcp),其它平台 1。消费它的是 INDEX 张量的形状计算和 PD 传输几何。

上游谓词 `requires_dsa_indexer_cache_replication()` 返回 `is_npu()`(platform.h,转引一轮书核验)——这套几何是 NPU 平台的显式选择,不是普适默认。

### 3.4 自然行恒等式 row(s)=s:三步初中代数

"妙"的部分:全局槽 s_g 永远恰好等于它该落的 INDEX 扁平行号。设 token 住在 b、E、s、f(本例 1、7、1、1)。

**第 1 步:调度器怎么发槽?**

```
s_g = E×(page×dcp) + s×page + f
   = 7×8 + 1×4 + 1 = 61
```

高位是 E(§2 讲过为什么:槽要直接定位物理资源)。

**第 2 步:INDEX 怎么编页编行?**

```
INDEX 物理页 P = E×dcp + s = 7×2+1 = 15     ( = expanded[3] ✓)
INDEX 行号     = P×page + f = 15×4+1 = 61
```

**第 3 步:两边相等。**

```
P×page + f
= (E×dcp + s)×page + f          ← 代入 P 的定义
= E×dcp×page + s×page + f       ← 乘开
= E×(page×dcp) + s×page + f     ← 交换律,凑成第 1 步的形式
= s_g                           ∎
```

成立条件只有两个:①槽按逻辑块尺寸编码(高位是 E);②页组基 = E×dcp(组内第 s 页对条带 s)。

注意:证明里**从没用过 E=b**。块表是 [5,7]、[0,1] 还是随机页号,恒等式都成立——这就是"自然行"的普适性。

验算(双保险):

- 正向:槽 61 → 页 15 行 1 → 扁平 61(两步各自算出);
- 反向:INDEX 第 61 行住的是谁?61//8=7(E)、(61%8)//4=1(条带)、61%4=1(偏移)——正是位置 13;
- 区间对账:页 15 的行区间 = 槽 60..63 = 位置 12..15(块 1 条带 1 的 4 个 token)✓。

「不需要任何重排」的含义:写侧槽直通行号(§4.6),读侧 expanded 与页组同一套算术——两侧零 gather/scatter。

### 3.5 门牌换算不是扩容

一个自然的担忧:页数×dcp,INDEX 是不是大了 dcp 倍?手算对照(同一 16 token 序列,page=4):

| | 逻辑块尺寸 | 逻辑块数 | INDEX 页/每 rank | INDEX 行/每 rank |
|---|---|---|---|---|
| kv_split=1 | 4 | 4 | 4×1 = 4 | 16 |
| kv_split=2 | 8 | 2 | 2×2 = 4 | 16 |

页数公式里的 ×kv_split,被逻辑块数的 ÷kv_split 精确抵消。**每 rank 的 INDEX 行数恒 = token 数**——全量副本,一人一行,本来就该这样。

所以"每逻辑块配 dcp 页"是门牌号换算,不是面积扩容。同一栋楼,门牌从"一层一户"改成"一层 dcp 户",楼还是那么大。

(总账视角:dcp=2 时全集群两份副本,总内存确实×2。但那是 §3.3 第 2 步"复制"的代价,不是"多配页"引入的新代价——两笔账要分开记。)

### 3.6 失败画面:INDEX 每逻辑块只配 1 页

**怎么错**:假想 kv_split=2 时 INDEX 仍沿用 latent 的"每逻辑块 1 页"。

**错在哪**:一页只有 4 行,装不下一个逻辑块 8 个 token 的摘要行。

**下游表现**:一半 token 的行无处落——要么越界写进隔壁块组的页(污染),要么两半互相覆盖(丢数据)。打分器读到残缺/串行的键摘要,topk 从根上选错。

## 4. 三张表,三个消费者(r1-Q4 · r2-Q4)

### 4.0 你当时的困惑(r1-Q4 / r2-Q4 原话)

> "local_slots 是给 sfadcp 用的、expanded 是给 indexer 用的吗?"(r1)
> "这两行为什么不一样,都是 index 啊,为什么会有两套输入呢,不明白啊"(r2)

判决(r1):方向对,但"两张表"的账漏了——实际是**三张表、三个消费者**,外加一个直通量。

判决(r2):INDEX 的两套输入是**写侧与读侧的分工**:写走全局槽直通(恒等式槽≡自然行),读走 expanded 表查页。

### 4.1 三张表总账

| 表/量 | 输入 | 变换 | 结果(位置 13) | 谁消费 |
|---|---|---|---|---|
| local_slots | 全局槽 61 | localize(按条带判归属) | rank0=-1 / rank1=29 | **latent 写**:只写本 rank 条带 |
| 逻辑表 | 引擎块表 | 转 int32 视图 | [5,7] | **SFA 读**:读 latent |
| expanded | 逻辑表 [5,7] | E×dcp+条带 | [10,11,14,15] | **indexer 读写** INDEX |
| (直通)全局槽 | 61 | 无变换 | 61 | **INDEX 写**:直写第 61 行 |

你的说法对在哪、漏在哪:

- 对:SfaDcp 后端确实消费 local_slots,indexer 确实消费 expanded;
- 漏一:SfaDcp 内部分"写 latent"和"读 latent"两件事,读用的不是 local_slots,是逻辑表;
- 漏二:INDEX 写用的不是这三张表里的任何数,是全局槽直通。

### 4.2 诞生:两张表在同一段代码里造出来

三张表里,逻辑表不用诞生(它就是引擎块表本身,原样往下传)。另两张诞生在 SfaDcpAttentionBackend 的备料函数 prepare 里——decode 每步先跑它:

```python
        local_slots = self._kv_layout.localize_slots(metadata.slot_mapping)
        if graph_mode:
            local_slots = copy_into_execution_buffer(
                ("DCP_LOCAL_SLOTS", tuple(local_slots.shape)),
                local_slots,
            )
        self._local_slot_mapping = local_slots
```
(sfa_dcp_backend.py:149-155)

变量溯源:`metadata.slot_mapping` 是调度器这步发的全局槽表,本例 [61]。`self._kv_layout` 是 §1.3 的 KVShardLayout 实例。`copy_into_execution_buffer` 是图模式的执行缓冲拷贝,eager 模式跳过。

输入:全局槽 [61]。变换:localize(§1.3 已手算)。输出:local_slots——rank0 视角 [-1],rank1 视角 [29]。

```python
        if self._block_table_i32 is not None:
            # ACL graph padded lanes keep logical block 0 and kv_seq_len=1 so
            # sparse MLA tiling stays valid. Expanding that row yields pages
            # 0..dcp_size-1. Do not rewrite those pages to -1: LightningIndexer
            # treats -1 as a huge page id and MTE-OOB.
            expanded = self._kv_layout.expand_indexer_block_table(self._block_table_i32)
```
(sfa_dcp_backend.py:157-162)

变量溯源:`self._block_table_i32` 是引擎块表的 int32 视图,即逻辑表 [5,7]。(这段注释是图模式 padding 行的伏笔,属另一章;本章只看 expand 这个调用。)

输入:逻辑表 [5,7]。变换:expand(§3.2 已手算)。输出:expanded [10,11,14,15]。

### 4.3 消费链 A:local_slots → latent 写

写 latent 发生在 execute_mla 内:

```python
        write_mla_paged_cache(
            attn_metadata.dcp_context.slot_mapping,
            k_latent_3d,
            k_pe_3d,
            nope_cache,
            rope_cache,
        )
```
(sfa_dcp_backend.py:258-264)

变量溯源:`dcp_context.slot_mapping` 就是前面存好的 local_slots(prepare 里 `attn_metadata.dcp_context.slot_mapping = local_slots`,sfa_dcp_backend.py:198)。`k_latent_3d / k_pe_3d` 是新 token 的压缩 KV。

write_mla_paged_cache 转调 `torch.ops.xllm_ops.reshape_paged_cache` 按槽直写。-1 槽被写核跳过。(⚠️ 证据边界:写核核体在 CANN 闭源侧;仓库内证据是调用侧注释与 localize 的 -1 语义。)

效果:rank1 写 latent 页 7 行 1;rank0 的 -1 直接跳过。**只写自家条带。**

### 4.4 消费链 B:逻辑表 → SFA 读 latent

SFA 调用点在 sfa_dcp.py 的 decode 分支:

```python
        sfa_output, softmax_max, softmax_sum = self._npu_sparse_flash_attention(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            actual_seq_lengths_query,
            dcp_context.seq_lens,        # local_seq_lens
            dcp_context.block_table,     # 未展开逻辑表(_block_table_i32)
            sparse_mode=0,
            return_lse=True,
        )
```
(sfa_dcp.py:561-571)

变量溯源:`dcp_context.block_table` 是逻辑表 [5,7](prepare 里塞进去的,sfa_dcp_backend.py:200-201)。`dcp_context.seq_lens` 是本 rank 视角长度(rank0=8、rank1=5,§0.3 已算)。`topk_indices` 已经 remap 成 L 基(§6)。

SFA 拿 L 的页字段查 block_table,落到 latent 页(§1.4 已手算:查 b=1 → 页 7)。**读用逻辑表,不是 local_slots。**

### 4.5 消费链 C:expanded → indexer 读写

indexer 拿表的方式是"换表"——SfaDcp 后端覆写 mla_index_context:

```python
    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        context = super().mla_index_context(layer)
        indexer_table = self.indexer_block_table_or_none()

        def materialize_index_cache() -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            index_cache, index_cache_scale, _ = context.materialize_index_cache()
            return index_cache, index_cache_scale, indexer_table

        return replace(
            context,
            block_table=indexer_table,
            materialize_index_cache=materialize_index_cache,
        )
```
(sfa_dcp_backend.py:209-221)

变量溯源:`indexer_block_table_or_none()`(sfa_dcp_backend.py:204-207)返回的就是 §4.2 的 expanded。这个 context 沿调用链传给打分器层。

效果:indexer 更新池(写)和打分(读)都拿 expanded 寻址 INDEX 页。**indexer 读写都用 expanded。**

深水区一句(r2-Q3 判决,sharded 写模式时):读侧还有一张 materialized 表,用"表内位置编号"——第三套编号基(npu_paged_attention.py:736-740 构造,`entry_ids×kv_split+owner_offsets` 后摊平)。默认 replicated 模式下它不是主角,知道它存在即可。

### 4.6 INDEX 写:全局槽直通

INDEX 写不走上面三张表。看落点(挑行、直写):

```python
        if has_index_shard and index_write_replicated:
            # Replicated write mode: every rank persists every stripe's page
            # at its natural row, and the natural row of global logical slot
            # s is s itself (block s // (block_size * kv_split) contributes
            # pages [e * kv_split, (e + 1) * kv_split) of block_size rows
            # each), so the global slot mapping drives the paged index write
            # directly -- exactly the kv_split_size == 1 arithmetic.
            slot_mapping = metadata.slot_mapping
```
(npu_paged_attention.py:1230-1237)

变量溯源:`metadata.slot_mapping` 是原始全局槽(61),**没有**经过 localize。注释直说:自然行的槽 s 就落在 INDEX 第 s 行。

真正的写动作:

```python
        cache_view = index_cache.view(-1, index_cache.size(-1))
        # Custom scatter sorts flattened offsets through FP32, which loses
        # integer precision for large caches with odd row widths such as 257.
        row_indices = slot_mapping.reshape(-1).clamp_min(0).to(torch.int64)
        cache_view.index_copy_(0, row_indices, values)
```
(npu_paged_attention.py:1391-1395)

`index_copy_` 按行号 61 直写扁平视图的第 61 行。"61 为什么恰好是该落的行"——§3.4 恒等式已经证完。

**两套输入为什么不同**:

- 写侧输入是"这步新 token 落哪行"。调度器发的槽已经是自然行(§3.4),直通零重排;
- 读侧输入是"历史 token 的摘要住哪页"。打分要按页寻址全量历史,页号空间是 E×dcp+s,必须查 expanded 表。

一个行级直写(scatter),一个页级查表(gather)——两套输入,一次是数据,一次是地图。

### 4.7 手算汇总:一个 token、四条路、一张总账

位置 13 的完整去向:

| 量 | rank0 | rank1 | 落点 |
|---|---|---|---|
| local_slots | -1 | 29 | latent 页 7 行 1(rank1) |
| 逻辑表(SFA 用) | [5,7] | [5,7] | 查 b=1 → latent 页 7 |
| expanded(indexer 用) | [10,11,14,15] | 同左 | INDEX 页 15 存位 12-15 |
| INDEX 写行号 | 61 | 61 | 各自 INDEX 的第 61 行 |

### 4.8 三个不变量,三个失败画面

三张表是三把钥匙,插错锁,门后都是错数据。不变量 1(latent 写必须用 local_slots)的失败画面已在 §1.6 演过,这里补另两把。

**不变量 2:INDEX 写必须用全局槽(自然行)。**

- 怎么错:INDEX 写误用 local_slots。
- 错在哪:rank0 得 -1(写核跳过,不崩);rank1 得 29 → 写进 INDEX 第 29 行。但读侧永远读第 61 行(§3.4 恒等式)。
- 下游表现:位置 13 的行是旧值或垃圾 → indexer 拿错键打分 → topk 选错 token → **输出错但无报错,最难查的一类**。

**不变量 3:SFA 读 latent 必须用逻辑表(b 基)。**

- 怎么错:SFA 误拿 expanded 当块表查。
- 错在哪:SFA 查 b=1 → expanded[1]=11 → 读 latent 页 11 行 1。页 11 是 INDEX 页号空间的页(块 5 条带 1 的 INDEX 页,§3.2 手算),在 latent 池里属于别的逻辑块。
- 下游表现:读错 KV → 错算;池子小则越界。

**核心结论(§4)**:local_slots 管写条带(-1 跳过)、逻辑表管 SFA 读、expanded 管 INDEX 读写;INDEX 写不走这三张表,全局槽直通自然行。

## 5. 两个"5"与位置 21(r2-Q4 · r3-(d))

### 5.0 你当时的困惑(r2-Q4 原话)

> "那那个 5 从哪里来的呢,对于 rank0 不是 -1 吗,5 从哪里来的呢"
> "你直接说 每个读的是 61 还是 14 15 还是 7+1 吧,不然不知道哪个是 local 哪个是全局"

先把"每个读的是哪个数"钉死(§4.7 总账的浓缩版):

```
latent 写核吃的是  29(rank1)/-1(rank0)   ← local_slots
SFA 读 latent 用的是 逻辑表 [5,7] + L=5    ← b 基
indexer 读写用的是  expanded [10,11,14,15] ← E 基页号
INDEX 写核吃的是   61                      ← 全局槽直通
```

剩下的困惑集中在一个数:怎么到处都是 5?拆开——**它们是两个不同的 5,数值巧合。**

### 5.1 第一个 5:o = 61%8(E 基槽偏移)

回看 §1.3 贴的第一块码(kv_shard_layout.py:79-88):

```python
        logical_offsets = torch.remainder(safe_slots, self.logical_block_size)
        owner_ranks = torch.div(
            logical_offsets,
            self.physical_block_size,
            rounding_mode="floor",
        )
        owned_slots = valid_slots & (owner_ranks == self.dcp_rank)
```

执行顺序是关键:`logical_offsets = 61%8 = 5` 在**第 82 行就算完了**,比属主判定(第 88 行)早。

rank0 和 rank1 跑同一段代码,都算出这个 5。rank0 最后得 -1,是结尾 `torch.where` 把"不归我"的格子改写成的——**中间量 5 在 rank0 上也存在过**。

它还有个名字:o,E 基槽的块内偏移。再拆一层:o = s×page+f = 1×4+1——高位是条带号 s。

回答原话:rank0 的 -1 是最终输出;5 是产生 -1 之前、人人共有的中间量。

### 5.2 第二个 5:L = b×4+1(b 基读侧索引)

§1.4 的读侧索引:L = b×page+f = 1×4+1 = 5。高位是块号 b。

### 5.3 巧合的条件:b=s=1

两个 5 的公式并排:

```
o = s×page + f = 1×4+1 = 5     ← E 基:块内偏移,高位=条带号
L = b×page + f = 1×4+1 = 5     ← b 基:读侧索引,高位=块号
```

低位都是 f=1,天生相同。高位一个是 s(块内第几条带)、一个是 b(序列内第几块)——**当 b=s 时,两个公式数值相等**。位置 13 恰好 b=1、s=1,撞出两个 5。

这不是设计,是本例数字的巧合。巧合就该有一个反例来破。

### 5.4 位置 21:破巧合的对照例

把序列拉长,块表扩成 [5,7,9](第 2 个逻辑块分到门牌 9)。看位置 21:

```
b = 21 // 8 = 2        ← 第 2 块(不再是 1)
o = 21 % 8  = 5        ← 块内偏移没变(s=1, f=1)
E = block_table[2] = 9
```

四个数逐一算:

```
全局槽 s_g = E×8+o   = 9×8+5 = 77     (≠ 61)
o         = s×4+f    = 1×4+1 = 5      (还是 5)
L         = b×4+f    = 2×4+1 = 9      (变成 9!)
本地写槽    = E×4+f    = 9×4+1 = 37     (rank1)
```

**L=9 ≠ o=5**——b 从 1 变 2,读侧索引跟着变;o 只看块内,纹丝不动。两个"5"正式分道。

顺带验证恒等式在位置 21 依然成立(它对任意 E 都成立,§3.4):

```
INDEX 页 P = E×dcp+s = 9×2+1 = 19     (= expanded[5],表 [10,11,14,15,18,19])
INDEX 行 = P×4+f = 19×4+1 = 77        (= 槽 77 ✓)
```

### 5.5 失败画面:把 o 当 L 用

**怎么错**:看到两个 5 数值相同,有人把"E 基槽偏移 o"直接当"b 基读侧索引 L"喂给 SFA。

**错在哪**:在位置 13 上碰巧不出错(b=s=1)。挪到位置 21:o=5 但 L=9。SFA 拿 5 当 L:页列 5//4=1、行 1 → 查 block_table[1]=7 → 读 latent 页 7 行 1。

**下游表现**:页 7 行 1 是**位置 13 的格子**,而位置 21 的正确落点是页 9 行 1(L=9 → 页列 2 → block_table[2]=9)。张冠李戴,静默错算——巧合数字骗过了测试,换一行就翻车。

### 5.6 一句话收束

**o 是"E 基的块内偏移"(属主判定前的中间量),L 是"b 基的读侧索引"(SFA 的行号);b=s=1 时数值巧合,位置 21(L=9≠o=5)把巧合破掉。**

## 6. 同一个 localize_slots,两个调用点,两种输入基(r1-Q4 · r3-(b)(d))

### 6.0 你当时的困惑(r3 原话)

> "这个 mask 里之前存的是序列中选中 token 的序号吗,需要换成本地的 slot 号吗……怎么 remap 成本地的呢"

前半问的答案(r3-(a) 判决):mask 存的是 **0 基 kv 位置**。源码 docstring 原文:"Returns absolute kv-position top-k indices [B, S_q, topk] (int64, -1 = invalid)"(glm5_next.py:1102-1104)。不是全局槽 61,也不是本地槽。

后半问(换成什么、怎么换)的答案藏着一个漂亮的巧合:**同一个 localize_slots 函数,两个调用点,两种输入基。**

### 6.1 写侧调用:喂槽 61(E 基)

§4.2 已贴:sfa_dcp_backend.py:149,`localize_slots(metadata.slot_mapping)`。

输入是全局槽 61。函数除法取余后,输出 29——E 基的物理页空间(§1.3 手算)。

### 6.2 读侧调用:喂位置 13(b 基)

读侧的调用链长一些,三跳:

```
sfa_dcp.py:559   topk_indices = self._remap_sparse_indices(topk_indices)
      ↓
sfa_dcp.py:329   _remap_sparse_indices(融合核或 torch 回退两条路)
      ↓
sfa_dcp_ref.py:22-40   remap_sparse_indices(参考实现)
```

参考实现的本体:

```python
def remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    """Pack this rank's owned sparse slots to the front of the SFA width.
    ...
    """
    last_dim = int(topk_indices.shape[-1])
    if last_dim < index_topk:
        raise RuntimeError(...)
    return layout.pack_owned_slots(topk_indices)
```
(sfa_dcp_ref.py:22-40,节选)

最后一行是关键:`pack_owned_slots(topk_indices)`。而 pack 的第一行就是 localize(kv_shard_layout.py:110,`localized = self.localize_slots(logical_slots)`)。

pack 做的另一半事(有效前置打包)属于"游程契约",不在本章展开。这里只看:**localize 吃到的输入是 topk_indices,即位置(13),不是槽。**

位置 13 的高位除出来是 b=1(§1.1),输出自然落在 b 基:L = 1×4+1 = 5。

### 6.3 函数不查表:输入什么基,输出什么基

回看 §1.3 贴的函数体:localize_slots 全程只有 `div` 和 `remainder`,**不查任何表**。

```
写侧:输入 61(E 编码)→ 高位除出 E=7  → 输出 7×4+1=29(E 基,物理页空间)
读侧:输入 13(b 编码)→ 高位除出 b=1  → 输出 1×4+1=5(b 基,块表下标空间)
```

同一个函数、同一套算术,输入空间换了一套,输出空间跟着换。这不是 bug 是设计:条带归属判定(o//4=owner)对两种输入同样成立,因为位置和槽的低位编码同构(都是"块尺寸 8 取余再切半")。

**读侧从不经过全局槽 61**(三轮主编裁决 R2:二轮书"位置→全局槽→owner"是概念性表述,读侧没有 61 这个中间步)。61 只活在写侧:调度器发出、INDEX 直通、latent 写前降基。

### 6.4 C++ 孪生:同公式再证一次

C++ 侧有一个孪生函数 `map_cache_slots_to_kv_shard`(npu_cp_plan.cpp:914 起),由 CP 计划的 `NpuCpPlan::prepare_cache_slots` 调用(npu_cp_plan.cpp:1167 起),做同样的降基:

```cpp
  const int32_t logical_block_size = block_size * kv_split_size;
  torch::Tensor logical_block_offsets =
      torch::remainder(recovered_logical_slots, logical_block_size);
  torch::Tensor slot_kv_split_ranks =
      torch::floor_divide(logical_block_offsets, block_size);
  ...
  torch::Tensor logical_block_ids =
      torch::floor_divide(logical_slots, logical_block_size);
  torch::Tensor physical_block_offsets = logical_slots % block_size;
  torch::Tensor local_physical_slots =
      logical_block_ids * block_size + physical_block_offsets;
```
(npu_cp_plan.cpp:914-958,节选)

逐块讲:同一套算术——槽 % 逻辑块尺寸得偏移,偏移 // 页尺寸得属主,槽 // 逻辑块尺寸得块号,块号×页尺寸+行内偏移得本地槽。

**「本地页 = 块表值」不变量**(r3-(d) 判决的原文):输入是全局槽时,`logical_block_ids` 就是 E,本地槽 = E×block_size+行——降基只压缩偏移(8 行制→4 行制),页号原样保留门牌。全局视角同一门牌 8 行(整块),本地视角 4 行(自家条带)。

### 6.5 失败画面:不 remap,直读

**怎么错**:假想删掉 remap,SFA 把位置直接当本地行号 v 用。

**错在哪**:rank1 的 latent 本地布局是"页列 v//4、行 v%4、查 block_table[页列]"。拿位置 5 直读:v=5 → 页列 1、行 1 → 查表得页 7 行 1 = **槽 29,是位置 13 的格子**。拿位置 13 直读:v=13 → 页列 3——块表 [5,7] 只有 2 列,**越界**。

**下游表现**:要位置 5 的 KV,拿到位置 13 的(张冠李戴);要位置 13 的,读到越界处的陈旧页垃圾。静默错算,不崩溃。

## 7. 收官:本质就一句话

窗户纸就一句话:

> **三张表是三个消费者各自的母语读本:写核说 E 基(槽/写槽)、SFA 说 b 基(L+逻辑表)、indexer 说页组基(expanded);三本字典之间只隔一次块表查询,而 INDEX 的自然行几何让全局槽免查字典直通行号。**

拆成三口气说破:

**第一口气:三个消费者,三副眼镜。**latent 写核只认"本 rank 页号+行"(还要 -1 当跳过信号);SFA 读核拿到的页字段是 b,要查逻辑表;indexer 面对的池子页数是 latent 的 dcp 倍,要查 expanded。谁的眼镜塞错了表,数据就落错格(§4.8)。

**第二口气:两套编码共用同一个 E。**槽把 E 编在高位(E×8+o),INDEX 行也把 E 编在高位((E×dcp+s)×4+f),系数 8=2×4 恰好咬合——两个编码自己撞成同一个数。不是巧合不是魔法,是 §3.4 的两个约定共用同一个 E。

**第三口气:localize 是一台无表除法器。**写侧喂槽、读侧喂位置,它都只做 div/mod。输入高位是什么基,输出就是什么基——所以写槽 29 与读索引 5 长在两棵树上,靠 block_table[1]=7 这座桥指向同一格。

冷读自测:合上这页,你能不能只凭"page=4/dcp=2/表[5,7]/位置 13"把 61、29、5、9(位置 21 的 L)四个数全部手算出来?能,这一章就长在你身上了。

## 附:术语消歧表

| 术语 | 量纲/含义 | 值从哪来 | 控制什么 | 易混对象(区别) |
|---|---|---|---|---|
| 位置 position | 0 基 token 序号 | 序列本身 | topk 输出(mask 内容)、读侧 remap 输入 | 全局槽(13 ≠ 61) |
| 逻辑块 logical block | token 数,尺寸 = page×dcp = 8 | 序列按 8 切 | 块表下标 b、INDEX 页组 | 物理页(页号空间) |
| 物理页 physical page | 页号 | 池分配:latent 用 E,INDEX 用 E×dcp+s | 池内寻址 | 逻辑块(下标空间) |
| 块表 block table | 页号数组 | 调度器分配 | b→E 的桥 | expanded(展开后的 INDEX 页表) |
| b(序数) | 块下标 | 位置除出(b=p//8) | 读侧查表列号 | E(门牌,值从表里查) |
| E(门牌) | 块表值 | block_table[b] | 写侧槽编码基(槽=E×8+o) | b(序数,值从位置除出) |
| 全局槽 global slot | 槽号 = E×8+o | 调度器 slot_mapping | latent 写(经 localize)、INDEX 写(直通) | 位置;本地槽 |
| 条带 stripe | 块内整页段,[s×4,(s+1)×4) | 块内偏移区间 | latent 归属、INDEX 组内页 | 头分片(切头,不切 token) |
| 本地写槽 | E×page+行 = 29 | localize_slots(槽 61) | latent 池写核 | 读侧索引(29 ≠ 5) |
| 读侧索引 L | b×page+行 = 5 | localize_slots(位置 13) | SFA 查块表读 latent | 写侧槽;槽偏移 o(两个"5") |
| o(槽偏移) | 块内偏移 = s×4+f | 槽或位置 %8 | 属主判定的原料 | L(b 基,位置 21 上 9≠5) |
| local_slots | 每 token 的本地写槽表 | prepare 里 localize 产物 | latent 写 | 逻辑表/expanded(另两张表) |
| expanded 表 | INDEX 页号序列 | expand_indexer_block_table(逻辑表) | indexer 读写 INDEX | 逻辑表(SFA 用);local_slots |
| dcp_rank | 0..dcp-1 | KV 条带轴上的编号 | 条带归属 | tp_rank(注意力头切分轴) |

## 自检

写作输入与诚实声明:本章深潜写作直读 round1-book.md 第 625-1398 行原文(774 行);二/三轮书已删,其判决与手算数字以 session-extracts.md 成员报告为准(MEMBER REPORT #101/#104/#113、#143/#148、ASSISTANT #135/#165);全部贴码行号对照本地物化 src/ 实读核验(kv_shard_layout.py / sfa_dcp_backend.py / npu_paged_attention.py / sfa_dcp.py / sfa_dcp_ref.py / npu_cp_plan.cpp / cache_layout_builder.h / glm5_next.py)。

| # | 教学提示词硬约束 | ✓/✗ | 证据位置 |
|---|---|---|---|
| 1 | 零裸术语,前置首现即教 | ✓ | §0.2 八件前置;E/b/o/s/f/L 首现处均先给人话定义(§1.1/§1.2/§2.1/§2.2) |
| 2 | 先全景后展开 | ✓ | §0.3 三池布局图、§0.4 四条路主线图,先于一切机制展开 |
| 3 | 概念地图先于调用链 | ✓ | §0.3 图里无函数名;§1.0 四层总表先于 localize 贴码 |
| 4 | 段落 ≤3 行,拒绝分号长句 | ✓ | 全章;所有长推导竖排进代码块,不揉进段落 |
| 5 | 贴码 5-15 行,逐块讲 | ✓ | 15 个贴码块全部 4-14 行,每块配变量溯源+逐行讲(输入/变换/输出) |
| 6 | 每个机制配可手算示例 | ✓ | §1.3(61→29)、§3.2([5,7]→[10,11,14,15])、§3.4(恒等式 61 验算)、§5.4(位置 21)、§6.2(13→5) |
| 7 | 手算分步成行,带中间状态 | ✓ | §1.3/§3.2/§3.4/§5.4 均为"输入→逐步→结果"代码块 |
| 8 | 不变量翻译成失败画面三段(怎么错→错在哪→下游表现) | ✓ | §1.6/§2.6/§3.6/§4.8(×2)/§5.5/§6.5 共 7 处,均三段式 |
| 9 | 证据规范:file:line+原文贴码+变量溯源+禁断章取义 | ✓ | 15 处贴码全部实读物化 src 核验;每块交代所属函数与调用环节 |
| 10 | 推导零跳步 | ✓ | §3.4 三步代数逐步乘开;§2.5 两口气证明显式写出"槽里没有的信息除不出来" |
| 11 | 编号镜像用户问题号 | ✓ | 章首导语映射;各节标题带 r1-Q4/r2-Q3/Q4/Q5/r3-(b)(d);§2.0/§3.0/§4.0/§5.0/§6.0 引用户原话 |
| 12 | 同名概念强制消歧 | ✓ | 附:术语消歧表 14 行(位置/槽/块/页/b/E/o/L/两张表) |
| 13 | 核心结论真是一句话 | ✓ | §4.8 尾、§5.6、§7(收官一句话+三口气) |
| 14 | **冷读者能否跟着手算出 61/29/5/9 这四个数** | ✓ | 61=7×8+5(§1.2);29=7×4+1(§1.3);5=o=1×4+1 与 L=1×4+1(§5.1-5.3);9=位置 21 的 L=2×4+1(§5.4);§7 留冷读自测题 |
| 15 | 诚实纪律:行号实读核验;书外内容标转引;证据边界标注 | ✓ | 行号全部实读核验;§3.2 测试与 §3.3 platform.h 标注"转引一轮书/二轮报告核验";⚠️ 证据边界 1 处(§4.3 CANN 写核闭源) |

> 自检结论:15/15 通过,无 ✗ 项。唯一证据边界(写核 -1 跳过的核体行为在 CANN 闭源侧)已显式标注,未当作已证事实陈述。

---

# D2 indexer 层与 mask + D3 remap 与打包

> 深潜层章节(教学级)。覆盖「谁在打分、mask 长什么样、mask 怎么变成本地读索引」这条生命线。
> 事实与行号全部对照本工作区 src/ 物化源码实读核验;
> 二/三轮教学书已删,其判决与手算数字以 session-extracts.md 成员报告为准(诚实声明)。

## 0. 导航:本章回答哪些追问

三轮教学讨论中追问频率最高的八个机制,按提问轮次镜像编号:

```
r1-Q1   indexer 的 query 是全量还是被 TP 切了?
r2-Q1   「glm5.2 是 64 头 + allreduce sum」这个记忆对吗?
r2-Q2   kPool mask 在哪里、起什么作用、是随机的吗?
r2-Q6   「必选尾」是配置项吗?_fill_wide_remap 在干什么?
r3-(a)  mask 里存的是选中 token 的序号吗?坐标是什么?
r3-(c)  是有效放前面、无效放后面吗?「前后」分几层含义?
r3-(d)  怎么 remap 成本地的?算术链全程是什么?
(r1-Q3(a) 的排序问题并入 r3-(c),同一机制)
```

**统一数字**(全章不变,拿笔就能跟着算):

```
page = 4      每个物理页装 4 行(教学值;真实 128)
dcp  = 2      dcp 组两个 rank:rank0、rank1
逻辑块表 = [5, 7]   序列第 0/1 个逻辑块放在页 5/页 7
序列已有 13 个 token;新 token 是第 14 个 = 位置 13(0 基)
位置 13 的全局槽 = 61;rank1 的写槽 = 29;读侧索引 L = 5
逻辑块大小 = page × dcp = 8(一个逻辑块装 8 个 token)
```

## 0.1 前置概念(首现即教,后文直接使用)

写给半年后没读过书的你,每个一句话:

- **TP(张量并行)**:模型太大时把权重按列切成 N 份,每卡一份;切列即切注意力头。
- **rank**:一张 NPU 卡上一个独立计算的进程。本章例 2 个:rank0、rank1。
- **DCP(dcp 组)**:`kv_split=2` 时两个持不同 KV 条带的 rank 结成的小组。TP 管「你有哪几个头」,DCP 管「你保管每个块的第几条带」——两个正交的轴。
- **条带**:逻辑块(8 token)按 page=4 切两半;rank0 持条带 0(块内偏移 0..3),rank1 持条带 1(偏移 4..7)。
- **latent(MLA 压缩 KV)**:每 token 的 K/V 压成一条低维向量存缓存;被 DCP 切条带的就是它。
- **indexer(索引器)**:主注意力开算前的轻量打分器,选出最重要的 top-k 个历史位置;只产出「位置清单」,不产出注意力结果。
- **INDEX cache**:每历史 token 的「索引指纹」缓存(一条 128 维 K 向量加门分数),indexer 打分读它。
- **kPool mask / topk 行**:indexer 每步算出的稀疏选择掩码——「每 token 可见哪些历史位置」的白名单(D2.3 拆开讲)。
- **SFA(sparse flash attention)**:稀疏闪注意力算子,只对行里列出的位置算注意力;读行方式是**碰到第一个 -1 就停**(首停契约)。

## 0.2 四层坐标空间(全章的地图)

| 坐标空间 | 记号 | 例值(位置 13) | 含义 | 谁用它 |
|---|---|---|---|---|
| **位置** | p | 13 | 0 基 kv 位置 | indexer 打分、mask 存储 |
| **读侧索引** | L | 5 | 本 rank 条带内行号:`b×page+f` | SFA 读 latent |
| **写侧本地槽** | — | 29 | 写核落点:`E×page+f` | 写 KV 的核 |
| **全局槽** | — | 61 | 调度器发的行号:`E×8+o` | INDEX 写(自然行) |

四个数不是一个东西。桥接(全章反复用):

```
位置 13 → b=13//8=1,查表 block_table[1]=7 → 逻辑块 1,门牌 E=7
E=7 → 写侧算术 E×8+o(o=5)     → 全局槽 61(INDEX 第 61 行)
E=7 → 写侧算术 E×4+(o%4)      → rank1 写槽 29(本地页 7 行 1)
位置 13 → 读侧算术 b×4+(p%4)   → 读侧索引 L=5
L=5 → SFA 查 block_table[5//4]=7,行=5%4=1 → 本地页 7 行 1(与写槽同格!)
```

读侧 L=5 与写侧槽 29 **指向同一个物理格**——两条算术路不同、落点相同,写读闭环(D3.1 手算)。

四个小字母钉死:**b**=逻辑块序数(`p//8`);**E**=块表门牌(`block_table[b]`);
**s**=条带号(`(p%8)//4`);**f**=条带内偏移(`p%4`)。
位置 13:b=1、E=7、s=1、f=1。

## D2 indexer 层与 mask

### D2.1 (r1-Q1) indexer 的 query 是全量还是被 TP 切了?

**问题重述**:引文说 indexer「用 expanded 表读【本 rank 的 INDEX 副本】给历史打分」——query 是全量还是被 TP 切了?

**一句话判决:全量。indexer 三投影在 TP 下不切,每 rank 持有完整权重。**

| 对象 | TP 下切不切 | 怎么处理 |
|---|---|---|
| 主注意力 query 头(q_b_proj) | **切**(按头) | 每 rank 一半头 |
| latent KV | **切**(按条带) | 每 rank 一条带 |
| indexer 三投影(wq_b/wk/weights_proj) | **不切** | 每 rank 完整副本 |
| INDEX cache | **不切** | 每 rank 逐位相同的全量副本 |
| hidden_states | **不切** | TP all-reduce 后两 rank 相同 |

记法:**切两样(头、条带),不切三样(indexer 权重、INDEX、hidden)。**

#### 贴码:加载时谁切谁不切

属于 `Glm5NextForCausalLM._load_dsa_attn`(glm5_next.py:3046 起),
模型加载时逐投影声明切分方式:

```python
        # q_b_proj: column-parallel QLinear (shard dim 0).
        self._load_qlinear(L, attn, "q_b_proj", 0)
        if not self.cfg.indexer_shared(i):
            idx = attn + "indexer."
            # Indexer projections are plain nn.Linear and stay replicated: the
            # sparse kPool mask is shared across TP ranks (each rank attends its
            # head-subset to the SAME mask), so wq_b/wk/weights_proj hold the
            # full index-head tensors on every rank.
            for w in ("wq_b.weight", "wk.weight", "weights_proj.weight"):
                L.load_fp(idx + w)
```
(glm5_next.py:3050-3065,节选;`L` 是权重加载器,`attn`/`i` 是层名与层号)

- `q_b_proj` 传切分参数 0:**主注意力 query 按头切**——被切的那一样;
- indexer 三投影 `load_fp` **不带切分参数 = 整块复制到每个 rank**;
- 注释说死原因:kPool mask 在 TP 各 rank 间共享,每 rank 的头子集对着**同一个(SAME)mask**。

补一刀为什么「切不了」:头结构长在 query 侧(`wq_b` 摊开全部 index heads,
`wk` 输出只有一条共享 K、单头,`weights_proj` 每头一个标量门)——
TP 对单头张量没有头可切,只能复制。

#### 手算:latent 世界被切,INDEX 世界没切

序列 13 个 token,新 token 是位置 13。

**latent 侧**:总长 13 = 1 整块 + 5 零头 → rank0 分到 1×4+clamp(5−0,0,4)=**8** 个,
rank1 分到 1×4+clamp(5−4,0,4)=**5** 个(8+5=13 ✓)——latent 被切。

**INDEX 侧**(位置 13 的指纹在哪):

> 输入 → 位置 13:块 1(E=7)、条带 1、条带内偏移 1。
> 计算 → INDEX 页 = E×dcp+s = 7×2+1 = **15**;行 = 15×4+1 = **61**。
> 结果 → **两个 rank 都有第 61 行,逐位相同**——INDEX 没被切。

位置 13 的 latent 只在 rank1;它的索引指纹,两个 rank 都有。

#### 一致性:复制计算,零通信

三个输入两 rank 逐位相同 → 各自独立重算,打分结果天然逐位相同。
select_topk 调用路径上**没有任何 all_reduce / all_gather**;decode 元数据
也不挂分片标记——indexer 的打分上下文里没有 rank 项。

> ⚠️ 证据边界:逐位一致由「相同输入 + 确定性 kernel」推出,非显式同步;
> 无跨 rank 逐位比对测试;端到端佐证是实测 max|dlogprob|=0.0。

#### 失败画面:如果切了(假想配置)

假设 indexer query 被切:rank0 用头子集 A 打分,rank1 用头子集 B——
打分公式不同 → topk 清单不同 → 两 rank 读**不同位置集合** →
融合时加权对象错位,拼出的不是任何正确结果,且**静默错算不崩溃**。

诚实收窄:这幅画面只对「**切头且不归约**」的假想配置成立——
完整裁决在 D2.2 末尾。

### D2.2 (r2-Q1) 「glm5.2 里 64 个头、结果求 sum、做了 allreduce」——是叠影

**问题重述**:你记得 glm5.2 的 query 有 64 头、TP 切、结果 allreduce sum——glm5.3-flash 不是这样吗?

**一句话判决:记忆是两件事的叠影。64 头+all_reduce 属主注意力;
indexer 实为 32 头、三投影全复制、单核打分+topk 零归约。**

| 事物 | 头数 | TP 处理 | 归约 |
|---|---|---|---|
| 主注意力 | 64(默认) | **切头**(q_b_proj 列切) | o_proj 后 **all_reduce** |
| indexer | **32**(index_n_heads) | 三投影**全复制** | **无** |
| indexer kernel | 垫到 64(GSIZE) | 垫零头(不是切) | 核内求和,零跨 rank 通信 |

#### 证据 1:主注意力才有 64 头 + all_reduce

主注意力头数默认 64(glm5_2.py:191 `n_heads` 缺省 64);TP 切头后每 rank 32 头,
o_proj 行并行必须归约——forward 尾部:

```python
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            output_dtype = o.dtype
            o = o.to(torch.float32)
            distributed.all_reduce_(o)
            o = o.to(output_dtype)
```
(glm5_2.py:556-561;`v_full` 是各头输出拼接,`o` 是本层注意力结果)

你记的「64 头求 sum + allreduce」就是这里——**属于主注意力,不属于 indexer**。

#### 证据 2:indexer 实为 32 头,三投影全复制

配置 `index_n_heads: int = 32`(glm5_2.py:115);加载时三投影整块复制:

```python
            if not self.model.layers[i].self_attn.is_shared:
                idx = attn + "indexer."
                loader.load_w8a8_projection(idx, "wq_b")
                loader.copy_replicated(idx + "wk.weight")
                loader.copy_replicated(idx + "k_norm.weight")
                loader.copy_replicated(idx + "k_norm.bias")
                loader.copy_replicated(idx + "weights_proj.weight")
```
(glm5_2.py:896-902;`loader` 是 W8A8 权重加载器)

`wq_b` 不带切分参数 → 全量复制;`wk` 等逐字面复制。**两代模型在此同款。**

#### 证据 3:kernel GSIZE=64 垫零头(叠影第二来源)

aclnnQuantLightningIndexer 的 AICPU tiling 硬性要求 `头数_q/头数_k = 64`;
GLM-5.2 是 32/1,于是垫零:

```python
# The AICPU tiling of ``aclnnQuantLightningIndexer`` requires
# ``num_heads_q / num_heads_k == 64``. GLM-5.2 uses ``index_n_heads=32`` and
# ``num_heads_k=1``, so we pad Q / q_scale / weights along the head axis to
# 64 heads (padded weights are zero, keeping ``score = sum_h w[h]*q[h]*k``
# mathematically identical).
```
(glm5_2.py:43-47;垫零函数 :588-619,常量 `_QLI_KERNEL_GSIZE = 64` 在 :85)

**手算:垫零为什么不改分数。** 打分公式 `score = Σ_h w[h]·(q[h]·k)`:

> 输入 → 真 32 头,垫 32 个零头凑满 64。
> 计算 → 垫的头 h≥32:q[h]=0 → q[h]·k=0;w[h]=0 → 项 = 0。
> 结果 → 64 项和 = 32 真项 + 32 零项 = 原分数,逐位不变。

你记的「64 头」是 kernel 的形状要求(垫出来的),不是模型头数(32),更没被切。

#### 证据 4:打分 + topk 单核完成,零归约

```python
            topk = kernels.quant_lightning_indexer(
                q, index_cache,
                (weights_padded * weight_scale).to(torch.float16),
                q_scale, index_cache_scale, qli_metadata,
                actual_seq_q, actual_seq_kv, block_table,
                self.topk, cmp_ratio,
            )
```
(glm5_2.py:680-692,节选;`index_cache` 是本 rank 的 INDEX 全量副本)

输入全量 q(垫零后)+ INDEX 副本;核内打分+topk 一次完成;
输出位置清单——**没有 all_reduce,没有跨 rank 通信**。

#### 失败画面收窄(D2.1 末尾那幅画面的最终裁决)

「切头不归约 → topk 不同 → 融合拼错」只在假想配置下成立
(真把 indexer 头切开、又不在 topk 前归约)。glm5_2/glm5_next 都没这么做。
那幅画面的价值是解释**为什么必须复制**:切了头要么归约(每步同步点),
要么各选各的(静默错算);复制是第三条路——显存换零通信。

**核心结论:indexer 32 头全复制零归约;「64 头+allreduce sum」是主注意力(64 头切+o_proj all_reduce)与 kernel 垫零(GSIZE=64)的叠影。**

### D2.3 (r2-Q2 + r3-(a)) kPool mask 是什么、存什么坐标、-1 从哪来

**问题重述**:kPool mask 在哪里、起什么作用、是随机的吗?GLM-5.2 有没有?
mask 里存的是选中 token 的序号吗?行多宽?-1 怎么冒出来的?

**一句话判决:kPool mask = indexer 每步算出的 topk 稀疏选择掩码;
不随机、不是新 cache,共享靠复制计算;行里存 0 基 kv 位置。**

#### 它是什么:一张「每 token 可见谁」的白名单

decode 每步、每层,indexer 产出一张表,行=token,列=可见的历史位置:
如 token0 行 = [17, 3, 42, -1](可见位置 17/3/42)。

谁产生:indexer 的 select_topk(每步现算);谁消费:remap(D3)→ SFA;
活多久:本层本 step。**不随机**——是打分+topk 的确定性输出;
不是新 cache,不是固定模式。代码里叫 `topk_indices`——名字 indices,语义 mask。

#### 贴码 1:构造点文档串,坐标语义说死

```python
        Returns absolute kv-position top-k indices
        ``[B, S_q, topk]`` (int64, -1 = invalid), matching the reference
        indexer output.
```
(glm5_next.py:1102-1104;select_topk 是 Glm5NextIndexer 的打分选位方法)

- 形状 `[B, S_q, topk]`:batch × query 数 × 行宽;
- **存 absolute kv-position(0 基位置)**——不是全局槽(61),不是读侧索引 L(5),是「13 那个空间」;
- `-1 = invalid`:无效槽统一记号。

旁证:因果过滤直接拿位置与 q_pos 比大小(glm5_next.py:1282-1289);
mask 值从诞生到 remap 前**从不查块表**——查表是 D3 的事。

#### 贴码 2:行宽与 -1 补齐

```python
        output_width = self.topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        if topk_indices.shape[-1] < output_width:
            topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
```
(glm5_next.py:1293-1296;`self.topk` 默认 2048,:219)

行宽 = index_topk(2048),**开必选尾再加 kpool−1**(D3.3);
序列太短凑不满时 `F.pad(-1)` 补尾。

#### -1 的五个来源(排查时按此对号)

| # | 来源 | 发生在 | 代码锚点 |
|---|---|---|---|
| 1 | 不足宽补齐 F.pad | 池数 < 行宽 | glm5_next.py:1294-1295 |
| 2 | 无效候选 masked_fill | 池含无效成员/因果过滤 | glm5_next.py:1276-1279、1289 |
| 3 | padding 行整行清 -1 | 图模式假 token 行 | glm5_next.py:1297 |
| 4 | 图 token 维 padding 剥离后回补 | span 剥离 keep_rows 再 pad 回 | glm5_next.py:1342-1353 |
| 5 | remap 后 peer 沉底 | 不是本 rank 的位置置 -1 | kv_shard_layout.py:96-100 |

来源 5 是 D3 主角。来源 4:压缩尾路径先按真实 span 剥掉 padding 行
(`_prefix_token_tensor` 截前缀),算完再 pad 回捕获桶宽——静态图形状不变,假行全 -1。

> ⚠️ 证据边界:旧一轮书「attention_mask 对 padding 行为 False」不被源码支持——
> decode 的 attention_mask 恒全 1;清 -1 靠 span 剥离+F.pad+有效性过滤(二轮勘误,采信)。

#### 构造期顺序:有效项按分数降序

```python
            selected = pool_scores.topk(select_k, dim=-1).indices
```
(glm5_next.py:1261;torch.topk 默认 sorted=True)

返回 indices 按分数从大到小,之后构造期无按位置重排——
**行内有效项次序 = 分数次序**。

> ⚠️ 证据边界:NPU 融合 indexer(aclnn 核内)输出顺序闭源待验证;
> 但注意力对选中**集合**求加权和,置换不改结果(D3.2)。

「共享」的实现:不是谁算完发给大家,而是**每人自己算一遍,结果逐位相同**
(输入三复制,D2.1/D2.2 已证)。**glm5_2 没有 kPool**(每 token 一行 K,cmp_ratio=1);
kPool 压缩池是 glm5_next 新增,两代共用「topk 白名单」机制本身。

#### 勘误:toy 例 [0,2,3,-1] 与源码不符

二轮书例 C 曾写「topk=3、历史 4、选中 {0,2,3} → mask 行 = [0,2,3,-1]」。
**那个 -1 是行宽示意,不是源码行为。** 按源码重算:

> 输入 → topk=3,kpool=1,选中 3 个:{0,2,3}。
> 计算 → 行宽 = 3+(1−1) = 3;topk 输出宽 = select_k×rate = 3;F.pad 条件 3<3 不成立。
> 结果 → 真实行 = **[0,2,3],宽 3,没有 -1**。

何时才有 -1:真实配置 topk=2048 时同一场景是 [0,2,3]+2045 个 -1;
toy 想见 -1 要改成 topk=4 选中 3。三轮主编裁决(方案甲):二轮正本不动、
勘注明示;本笔记按源码讲。

#### 失败画面:把 mask 的值当槽号用

mask 存**位置**;若当本地槽直接喂 SFA,rank1 上值 5 会解到位置 13 的格子、
值 13 会越界读垃圾——静默错算。数字版全程在 D3.1 末尾。

## D3 remap 与打包

### D3.0 概念地图:一张 mask 的生命线

```
【decode 一步 · 一个 DSA 层 · rank0 与 rank1 并行】

① indexer 选位置(D2 已详述)
      输入:本 rank 的 INDEX 全量副本 + expanded 表
      输出:topk 行(0 基 kv 位置,-1=无效)
      两 rank 的 topk 行逐位相同(复制计算)

② remap:坐标翻译 + 打包 ← D3.1/D3.2 的主角
      变换:localize → 我拥有的变读侧索引 L,peer 的 -1;
            排序键打包 → 有效槽连成一段前缀
      输出:稀疏索引行(读侧本地索引,有效前置,-1 尾)

③ 本地稀疏注意力 SFA
      只对行里列出的位置算注意力;沿行走到第一个 -1 停
      输出部分输出 + LSE 统计量(融合属下一章,不展开)
```

总画面:**两 rank 拿着同一份 mask,各自 remap 出互补的不同本地行;
全程没有为「位置列表」发过一个字节的通信。**

### D3.1 (r3-(d)) remap 算术全程:位置 p 怎么变成读侧索引 L

**问题重述**:mask 存位置、SFA 读本地格子——中间坐标翻译链的每步算术是什么?

**一句话判决:p → b=p//8 → owner=(p%8)//4 → 本 rank 得 L=b×4+(p%4),
peer 得 -1;SFA 再查 block_table[L//4]=E、行=L%4。**

#### 先看对象:localize_slots

`KVShardLayout` 类的方法(kv_shard_layout.py)——条带布局的几何函数库,
纯算术、无通信、无数据搬运,只有索引翻译。
它被两个调用点共用:**写侧**喂全局槽(E 基),**读侧(remap)喂位置(b 基)**——
同公式,不同输入空间。

#### 贴码:翻译公式

```python
    def localize_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        valid_slots = logical_slots >= 0
        logical_offsets = torch.remainder(logical_slots.clamp_min(0), self.logical_block_size)
        owner_ranks = torch.div(logical_offsets, self.physical_block_size,
                                rounding_mode="floor")
        owned_slots = valid_slots & (owner_ranks == self.dcp_rank)
```
(kv_shard_layout.py:79-88,节选;后段 :89-100 算本地槽并按属主筛选)

```python
        local_offsets = torch.remainder(logical_offsets, self.physical_block_size)
        local_slots = logical_block_ids * self.physical_block_size + local_offsets
        return torch.where(owned_slots, local_slots,
                           torch.full_like(local_slots, self.INVALID_SLOT))
```
(kv_shard_layout.py:94-100;INVALID_SLOT=-1 在 :34)

page=4、dcp=2 时(logical_block_size=8):`o = p%8` 块内偏移;
`owner = o//4` 条带号(**0..3 归 rank0,4..7 归 rank1**);
`L = (p//8)×4 + o%4`;不是我的 → **-1**。

#### 手算全程:位置 13 的七步链(rank1 侧)

> 第 1 步 · 输入:mask 行里的一项,位置 p = 13。
> 第 2 步 · 块序数:b = 13//8 = **1**。
> 第 3 步 · 块内偏移:o = 13%8 = **5**。
> 第 4 步 · 属主:owner = 5//4 = **1** → rank1 的格子。
> 　　　　　(rank0 算出 owner=1≠0 → 直接 -1,后面几步不算。)
> 第 5 步 · 条带内偏移:f = 5%4 = **1**。
> 第 6 步 · 读侧索引:L = b×4+f = 1×4+1 = **5**。
> 第 7 步 · SFA 查表:页列 = L//4 = 1 → block_table[1] = **E=7**;
> 　　　　　行 = L%4 = **1** → 物理格 = 本地页 7 行 1。

连读:**13 → b=1 → o=5 → owner=1 → f=1 → L=5 → 页 7 行 1。**

**对照算(位置 8,rank0 侧)**:b=1、o=0、owner=0、f=0 → L=**4**;
SFA:页列 1 → 页 7,行 0。rank1 对位置 8 得 -1。同一位置,两 rank 答案永远互补。

**位置 21 分道例(破「两个 5」巧合)**:位置 13 的 o=5 与 L=5 数值相同,
是 **b=s=1 的巧合**(L=b×4+f,o=s×4+f,b=s 时相等),不是恒等式。
表扩为 [5,7,9] 算位置 21:b=2、o=5、owner=1、f=1 → L=**9**≠o=5,分道。

#### 写读闭环:L=5 和写槽 29 同一格

写侧对照:调度器为位置 13 发**全局槽 61**(E 基:61=E×8+o=7×8+5),
写侧同样调 localize_slots,喂 61:

> 输入 → 槽 61。计算 → b'=61//8=**7**(=E);o=5;owner=1;写槽=7×4+1=**29**。
> 结果 → rank1 把 latent 写进本地页 7 行 1。

```
写侧:槽 61 → localize(61) → 写槽 29 = 本地页 7 行 1
读侧:位置 13 → remap → L=5 → SFA 查 block_table[1]=7 → 本地页 7 行 1
```

**两条算术路不同,落点同一格**——写进去的正是读出来的。
不变量:**本地页号 = 块表值 E**;写侧槽是 E 基(除回去直接是 E=本地页),
读侧 L 是 b 基,block_table[b]=E 把它接回同一页。

> 提醒(三轮裁决):讲读侧 remap **勿引入全局槽 61 当中间步**——
> 读侧从位置直接到 L,从不物化 61;「位置→全局槽→owner」只是概念性表述。

#### 失败画面:不 remap,把位置直接喂 SFA

SFA 对行里的值一律按 **L 语义**解读(页列=值//4,行=值%4,再查表)。
rank1 忘了 remap 时:

```
张冠李戴(值 5):想读位置 5 → SFA 解读 L=5 → 页列 1 → 页 7 行 1
  → 那格装的是位置 13!(写侧刚写进去)→ 加权对象整个错位

越界读垃圾(值 13):想读位置 13 → SFA 解读 L=13 → 页列 13//4=3
  → block_table[3] ← 表只有 2 项 [5,7]!→ 越界 → 陈旧页号/垃圾
```

两件事都**静默发生**:无断言、无 NaN、不崩溃——
每 token 输出悄悄错掉,极难排查。

**核心结论:remap = 除法定块、除法定属主、乘加得 L,peer 一律 -1;
SFA 靠 block_table[L//4] 接回物理页。不翻译表,位置 5 顶替位置 13 的格子,
位置 13 越界读垃圾。**

### D3.2 (r3-(c) 阶段②) 打包:pack_owned_slots 的一个加法

**问题重述**:remap 后行里「我的」(L)和「不是我的」(-1)混着——
有效放前、无效放后吗?前后谁控制?两 rank 要协商吗?

**一句话判决:是——但「前后」分三阶段含义不同;顺序由排序键控制,
键全唯一 → 恒保原序;两 rank 零通信零协调。**

r3-(c) 的完整答案,三阶段先钉死(②本节,③是 D3.3):

| 阶段 | 发生在 | 「前」= | 「后」= | 机制 |
|---|---|---|---|---|
| ① 构造期 | select_topk | 选中的(分数降序) | 未选/无效 | torch.topk sorted=True |
| ② remap 期 | pack_owned_slots | 本 rank owned | peer 的(-1 沉底) | 键 = 列号+(无效?width) |
| ③ 宽行期 | _fill_wide_remap | 有效游程 | -1 尾 | 尾列 scatter 续游程 |

打包是硬需求(不变量):**SFA 走到第一个 -1 就停**,有效槽必须连成前缀。

#### 贴码:键设计

```python
    def pack_owned_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        """Map logical slots onto this rank and pack every owned slot to the front.

        SFA walks ``sparse_indices`` until the first ``INVALID_SLOT``, so all slots
        this rank owns must form one leading run. A kPool indexer appends tail
        columns next to the top-k ones; packing the whole row keeps that tail in
        the same run instead of leaving it behind the ``-1`` padding.
        """
```
(kv_shard_layout.py:102-109;调用方是 remap,sfa_dcp.py:345-349)

```python
        localized = self.localize_slots(logical_slots)
        width = int(localized.shape[-1])
        original_order = torch.arange(width, dtype=torch.float32,
                                      device=localized.device).expand_as(localized)
        pack_keys = original_order + (localized < 0).to(torch.float32) * width
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(localized, dim=-1, index=pack_order.to(torch.int32))
```
(kv_shard_layout.py:110-119)

逐行:先 localize(D3.1);`original_order` 是列号表;
`pack_keys` **核心一行——有效列键=列号,无效列键=列号+width**;
升序 sort 得列顺序;gather 按顺序取值。

键设计三层:有效键落 [0,width)、无效键落 [width,2×width)、
两区间不相交 → **键两两不同**。后果:排序结果唯一(不依赖 stable 语义,
根本没有相同键),有效列相对序 = 列号序 = **原序**。
原序 = 构造期的分数降序(D2.3)——pack 不发明顺序,只把无效沉底。

#### 手算:toy {0,2,3} 两 rank 各自打包(中间状态全给)

topk=3、kpool=1、历史 4 位置、选中 {0,2,3}(D2.3 勘误后:行宽 3 无 -1):

| 步 | rank0 | rank1 |
|---|---|---|
| localize | [0, 2, 3](p=0/2/3 全是块 0 条带 0,owner=0) | [-1, -1, -1](owner=0≠1) |
| 键(width=3) | [0, 1, 2](全有效) | [3, 4, 5](全无效,+3) |
| 打包后 | **[0, 2, 3]** | **[-1, -1, -1]** |

选中集全落在块 0 条带 0——rank0 全包,rank1 整行 -1:
**互补的退化形态**,两行合起来是同一选中集的两个视图。

#### 扩展变体 {0,2,3,5}:看一次真拆分

加位置 5(块 0 条带 1,rank1 的),行宽 4:

| 步 | rank0 | rank1 |
|---|---|---|
| localize | [0, 2, 3, -1](p=5 owner=1≠0) | [-1, -1, -1, **1**](p=5:f=5%4=1 → L=0×4+1) |
| 键(width=4) | [0, 1, 2, 7] | [4, 5, 6, 3] |
| 打包后 | **[0, 2, 3, -1]** | **[1, -1, -1, -1]** |

三件事一眼可见:长度不同(3 对 1)、内容不同(0,2,3 对 1)、
合起来恰是 {0,2,3,5} 的**互补拆分**——位置 5 的格子只在 rank1。
**全程零跨 rank 协调**,两 rank 互不知对方的行;后续 LSE 融合也只交换
部分输出,从不对齐位置列表。

#### 失败画面:不打包,-1 夹中间

```
rank1 不打包的行: [-1, -1, -1, 1]
SFA 走到第 0 列 → -1 → 停 → 第 3 列的 L=1(位置 5)没被读
→ 位置 5 的 V 不参与注意力 → 输出错,无报错
```

洞是常态:构造期因果过滤会挖洞(无效候选 -1),remap 又把 peer 变 -1——
打包是刚需。

**核心结论:打包 = 一个加法(无效键+width)让排序自动完成;键全唯一保原序;
两 rank 各自打包零协调,产出互补的不同本地行。**

### D3.3 (r2-Q6 + r3-(c) 阶段③) 必选尾与宽行 _fill_wide_remap

**问题重述**:「必选尾」是配置项吗?压缩池不是一对多吗,为什么 owned_prefix
是一个数?tail_offsets 是什么?fill_(-1) 铺底、前缀拷入、尾列 scatter 在干什么?

**一句话判决:必选尾是三配置合取;owned_prefix 数的是列不是池;
_fill_wide_remap 把尾列 scatter 到前缀有效游程后面,续成一段。**

#### 前置:必选尾为什么存在

kPool 每 rate 个 token 压成一池行;最新 token 可能还没攒满一个池、
**没进压缩池**——打分只在完整池上打,最新 token 永远选不中。
不强制带上,最新 token(刚生成的前文)永远进不了注意力。
开必选尾时,select_topk 在 top-k 列后**追加最近若干位置**。

```python
    index_kpool: int = 1
    index_kpool_compress: bool = False
    index_kpool_always_select_tail: bool = False
```
(glm5_next.py:220-222;行宽公式 :1293)

`index_kpool`(=rate)几个 token 压一池;`index_kpool_compress` 走压缩池;
`index_kpool_always_select_tail` **必选尾开关**(行宽 = topk+kpool−1)。
「一对多」澄清:一对多是**池与成员**(一池行覆盖 rate 个 token);
owned_prefix 数的是**行里的列**——两个维度,不冲突。

#### 贴码 1:构造期 append_causal_tail 自带稳定重排

```python
    tail_start = torch.div(positions + 1, rate, rounding_mode="floor") * rate
    tail = tail_start[..., None] + offsets
    tail = tail.masked_fill((tail > positions[..., None]) | (positions[..., None] < 0), -1)
    combined = torch.cat([selected_indices, tail], dim=-1)
    # SFA stops consuming a row at its first -1. Causal filtering can leave
    # holes in the scored pool prefix, so stably pack every valid slot before
    # the invalid suffix after appending the unscored tail.
```
(glm5_next_kpool.py:308-315;函数 append_causal_tail :297-320)

`tail_start` 是 query 位置后第一个未满池起点;`tail` 是最近 rate−1 个位置,
因果不可见的填 -1;拼接后**立刻稳定打包**(键设计与 D3.2 同款,:316-320)——
构造期就保证有效前置。

#### 贴码 2:消费侧 _fill_wide_remap

remap 快路(融合核,D3.4)只吃**恰好 2048 宽**前缀——宽出来的尾列归它管:

```python
    """Join the packed top-k prefix and the kPool tail into one valid run.

    SFA walks ``sparse_indices`` until the first ``INVALID_SLOT``, so the owned
    tail slots have to continue the owned prefix run. Parking them behind the
    prefix's ``-1`` padding drops the newest force-selected tokens from every
    decode step.
    """
    index_topk = int(packed_prefix.shape[-1])
    owned_prefix = (packed_prefix >= 0).sum(dim=-1, keepdim=True)
    tail_offsets = owned_prefix + torch.arange(tail_slots.shape[-1], device=out.device)
    out.fill_(layout.INVALID_SLOT)
    out[..., :index_topk].copy_(packed_prefix)
    out.scatter_(-1, tail_offsets, layout.pack_owned_slots(tail_slots))
```
(sfa_dcp.py:95-110,docstring 与本体;调用方 remap,sfa_dcp.py:377-382)

逐行:

- `owned_prefix`:**每行**数前缀里有效列个数——keepdim 保留行维,
  是 [行数,1] 张量,「每行一个数」不是全表一个数;
- `tail_offsets = owned_prefix + arange(尾宽)`:尾列落点,从有效个数
  的下一格往后排;
- `fill_(-1)` 铺底 → 前缀拷入 → 尾列先 localize+打包再**按落点 scatter**。
  尾列不占固定列位,占「前缀有效游程结束处往后」的动态位置。

#### 手算:_fill_wide_remap 微例

设 topk=3、kpool=2(尾宽 1)、rank1 侧,前缀已打包 = [1,-1,-1],尾列 = [5](位置 5):

> 第 1 步 · owned_prefix:前缀里 ≥0 的列数 = 1。
> 第 2 步 · tail_offsets:落点 = 1+[0] = [1] → 尾列写第 1 列。
> 第 3 步 · 尾列打包:pack_owned_slots([5]) → p=5 是 rank1 的,L=1 → [1]。
> 第 4 步 · 组装:铺底 [-1×4] → 拷前缀 [1,-1,-1,-1] → scatter 第 1 列写 1。
> 结果 → out = **[1, 1, -1, -1]**:两个有效连成一段前缀。

验证游程:第 0 列(前缀)、第 1 列(尾列)有效,第 2 列 -1 停。

#### 失败画面:尾列排在 -1 后,最新 token 每步被丢

```
反事实(尾列留原列位): [1, -1, -1, 1]
SFA 走到第 1 列 → -1 → 停 → 尾列的 L=1(位置 5,最新强制选中)没被读
→ decode 每步丢掉最新必选 token → 生成质量缓慢劣化,无报错
```

docstring 原话翻译:把尾列停在 -1 后面,**每步 decode 都丢掉
最新的强制选中 token**。

**核心结论:必选尾 = 三配置合取 + 构造期稳定前置;消费侧用 owned_prefix
当 scatter 起点把尾列续进有效游程——SFA 首停契约是这一切存在的唯一理由。**

### D3.4 (r3-(d) 收尾) 融合核约束与静默回退

**问题重述**:D3.1/D3.2 讲的是 torch 慢路;TileLang 融合核快路何时走、何时不走?

**一句话判决:topk 恰 2048、token≤256、page/dcp 均 2 的幂、int32 连续 NPU
张量,全满足才走融合核;否则静默回退 torch 路(同语义,慢)。**

```python
    return (
        index_topk == _REMAP_TOPK
        and last_dim == _REMAP_TOPK
        and 0 < num_tokens <= _REMAP_MAX_TOKENS
        and _is_power_of_two(physical_block_size)
        and _is_power_of_two(dcp_size)
        and topk_indices.dtype == torch.int32
        and topk_indices.is_contiguous()
        and _is_npu_tensor(topk_indices)
        and _fused_remap_available()
    )
```
(sfa_dcp.py:75-85,函数 `_can_use_fused_remap` :66-85;
常量 `_REMAP_TOPK=2048`、`_REMAP_MAX_TOKENS=256` 在 :29-30)

- `index_topk == 2048`:配置 topk 必须恰等于核编译宽度;
- `last_dim == 2048`:实际行宽也得 2048——**宽行走不了快路**,
  尾列切出去单独走 _fill_wide_remap(sfa_dcp.py:377-382);
- `0 < num_tokens <= 256`:本步 token 上限;
- 2 的幂 + int32 + 连续 + NPU:核的形状与布局要求。

glm5_next 默认(2048/page=128/dcp=2)恰好全命中——常规 decode 走快路。
门控不满足时**不抛异常、不打日志,静默**走 torch 参考
(sfa_dcp.py:334-351):`remap_sparse_indices`(sfa_dcp_ref.py:23-40),
本体就是 `layout.pack_owned_slots`——D3.2 同款。语义两路一致
(有效前置+-1 尾+保原序),差只在速度。

最容易踩的边界:**spec-verify 宽批**——投机解码验证一步 token 数 =
批大小×(spec+1),超 256 就回退;glm5_next 默认全满足,生产上只有这一处
掉速(会话研究报告列为「残余风险 3」)。

```
风险画面:spec-verify 一步 300 token(>256)
→ 门控不满足 → 静默回退 torch 路
→ 正确性无损,但 remap 从单核融合变多次算子往返
→ 每步延迟上升;无告警无日志,只有 profile 看得见
```

> ⚠️ 证据边界:回退无告警的判断来自门控代码(无日志语句);
> 掉速幅度无公开 benchmark,不引数字。

**核心结论:融合核是 shape 特化快路(2048/256/2 的幂),不满足静默回退
同语义慢路;spec-verify 宽批是唯一生产掉速点。**

## 本质一句话

```
indexer 是「全量复制的打分员」:两 rank 各拿全量权重和全量 INDEX,
各自重算出同一份白名单(mask,0 基位置);remap 是「每人翻自己的译表」:
位置 ÷8 定块、%8÷4 定属主,我的变 L、不是我的 -1,一个加法把无效
沉底打包成前缀;SFA 拿着互补的本地行各读各的条带——白名单共享靠
复制,位置落地靠算术,全程零通信。
```

捅破窗户纸:**mask 是「看哪」的名单,L 是「在哪」的地址,remap 就是
名单到地址的一次邮政编码翻译——两个 rank 翻同一本码簿,
却各自只投递自己片区的信。**

## 自检

对照教学提示词硬约束与本任务规格逐条检查:

| # | 检查项 | 状态 | 证据位置 |
|---|---|---|---|
| 1 | 机制 1:glm5_2 vs glm5_next 对照(32 头/三投影复制/单核零归约/GSIZE=64 垫零/叠影裁决/失败画面收窄) | ✓ | D2.1+D2.2(贴码 glm5_next.py:3050-3065、glm5_2.py:896-902/556-561/43-47/680-692,均实读核验) |
| 2 | 机制 2:kPool mask=每步 topk 白名单,非随机非新 cache,共享靠复制,glm5_2 无 kPool | ✓ | D2.3「它是什么」+「共享的实现」段 |
| 3 | 机制 3:mask 坐标语义(0 基 kv 位置/行宽 2048+尾/-1 五来源/构造期降序) | ✓ | D2.3 贴码 1/2+五来源表+topk sorted |
| 4 | 机制 4:勘误(toy [0,2,3] 宽 3 无 -1) | ✓ | D2.3「勘误」节(三轮裁决方案甲) |
| 5 | 机制 5:remap 算术全程(七步链/SFA 查表/写读闭环/失败画面双例) | ✓ | D3.1(位置 13 七步+位置 8+位置 21+闭环+两幅失败画面) |
| 6 | 机制 6:pack_owned_slots(键全唯一保原序/零协调/toy 两例互补) | ✓ | D3.2(键设计三层+{0,2,3} 与 {0,2,3,5} 手算表) |
| 7 | 机制 7:_fill_wide_remap(owned_prefix/tail_offsets/微例/必选尾三配置) | ✓ | D3.3(三配置+append_causal_tail+微例+失败画面) |
| 8 | 机制 8:融合核约束(2048/256/2 的幂/int32)与静默回退、spec-verify 宽批 | ✓ | D3.4(门控贴码+风险画面) |
| 9 | 每机制三段齐:主干+贴码+手算+失败画面 | ✓ | D2.1-D3.4 各节;D2.3 失败画面指向 D3.1(同机制不重复展开) |
| 10 | 贴码 3-8 行为主、标 文件:行号、行号实读核验 | ✓ | 全部代码块 ≤15 行;glm5_2.py/glm5_next.py/glm5_next_kpool.py/sfa_dcp.py/sfa_dcp_ref.py/kv_shard_layout.py 逐一 read 核验 |
| 11 | 前置概念首现即教 | ✓ | 0.1 九条 + b/E/s/f(0.2)+ rate/尾宽(D3.3)+ GSIZE(D2.2) |
| 12 | 统一数字:page=4/dcp=2/表[5,7]/13token/位置13/槽61/写槽29/读索引5 | ✓ | 0.2 坐标表+全章演算;位置 21 分道例显式标注表扩为 [5,7,9] |
| 13 | 编号镜像 r1-Q1/r2-Q1,Q2,Q6/r3-(a)(c)(d) | ✓ | 各节标题括注;r1-Q3(a) 并入 r3-(c) 已在导航声明 |
| 14 | 段落≤3 行、单句≤40 字、演算分步成行 | ✓ | 全章;手算块用「输入→计算→结果」三行式/分步表 |
| 15 | ⚠️ 证据边界显式标注 | ✓ | 4 处:逐位一致(D2.1)/融合 indexer 顺序(D2.3)/attention_mask 勘误(D2.3)/回退无告警(D3.4) |
| 16 | 诚实:二/三轮书已删,判决以 session-extracts 成员报告为准 | ✓ | 卷首声明+勘误节标注「三轮主编裁决」 |
| 17 | **冷读者验收 A:能否跟着算出 toy 例两 rank 的行** | ✓ | D3.2 两张手算表:localize/键/打包三行中间状态齐——{0,2,3} → rank0 [0,2,3]、rank1 全 -1;{0,2,3,5} → rank0 [0,2,3,-1]、rank1 [1,-1,-1,-1] |
| 18 | **冷读者验收 B:能否跟着算出位置 13 的 L=5** | ✓ | D3.1 七步链:b=13//8=1 → o=5 → owner=1 → f=1 → L=1×4+1=5 → 页 7 行 1;附位置 8(→4)与位置 21(→9)对照 |

全部 ✓,无 ✗ 项。

---

# D4 交换与融合 + D5 图模式 padding(深潜)

> 定位:SfaDcp 结算笔记深潜层第 4/5 章。覆盖机制总览、query all-gather、
> AllToAll 载荷与 LSE 编码、头拓扑、valid_lse 融合、图模式 padding 三件套。
> 编号镜像一轮 Q3(b) 与二轮 Q7a/Q7b/Q8。读者画像:半年后没读过教学书的你,
> 前置概念首现即教。

## 统一数字(全章手算共用)

- page=4(每物理页 4 行)、dcp=2(dcp 组两个 rank);
- 逻辑块表 [5,7];序列已有 13 个 token,decode 第 14 个(位置 13,0 基);
- 同一 token 四个数:位置 13 / 全局槽 61 / 写槽 29 / 读索引 5;
- local_seq_lens:rank0=8、rank1=5;expanded=[10,11,14,15];
- LSE 演示:lse_0=12、lse_1=13;载荷走查 (dcp,H_local,T,D+4)=(2,2,2,6)。
- 真实参数(page=128、topk=2048、H_full=64)只作对照,不进演算。

> 证据纪律:行号全部对照 `src/` 物化文件实读核验(分支
> feat/glm5-next-sfa-dcp-decode,快照 3bdf81ef2 / tip a2edee8498)。
> 二/三轮教学书已删,判决与手算以 session-extracts 成员报告为准,标
> [二轮报告]/[三轮报告];CANN 闭源行为标 ⚠️ 证据边界。

---

## 机制总览:decode 一步的八段旅程

前置概念(首现即教):

- **dcp 组**:kv_split=2 把一份 KV latent 沿 token 轴切 2 条(stripe,条带),
  TP 域内两 rank 一组、各存一条;NCCL 进程组名 `"dcp"`。
- **H_local / H**:TP 切头,每 rank 持 H_local 个头;H=全头数(本例 4/2)。
- **LSE(log-sum-exp)**:`lse = m + log Σ exp(x−m)`,一个标量装下
  「softmax 分母多大」,是后面融合的货币。
- **AllToAll**:组内交换——载荷切 dcp 片,第 r 片发 rank r,同时收齐
  每个 rank 发给我的片。区别于 all-gather(人人收全份)。

```
decode 一步(一个 DSA 层,dcp=2,两 rank 并行)
  prepare:调度器发全局槽 → 三张表诞生(local_slots / 逻辑表 / expanded)
  ① 拿索引上下文  mla_index_context:块表换成 expanded(页号空间)
  ② indexer 选位置:读全量 INDEX 副本打分 → topk(两 rank 逐位相同)
  ③ 写 KV:新 token 的 latent 只写本 rank 条带(peer 槽位 -1)
  ④ query all-gather(异步发起):[T,H_local,D] → [T,H,D]  ════ 通信边界 1 ════
  ⑤ remap:全局位置 → 本地读索引,自己的挪前部,别人的置 -1
  ⑥ 等待 query 到齐:handle.wait(),每 rank 手握全部头的 query
  ⑦ 本地 SFA:只对本 rank 条带算稀疏注意力 → 部分输出 [T,H,D] + LSE [T,H]
  ⑧ 打包 [dcp,H_local,T,D+4] → AllToAll → LSE 融合      ════ 通信边界 2 ════
     → 本 rank 头的最终输出 [T,H_local,D] → 交回 o_proj
```

主线一句话:**KV latent 从头到尾没跨 rank;跨边界的只有 query(④)和
「部分输出 + LSE」(⑧),大小都 ∝ token 数 × 头数,与上下文长度无关。**

主流程代码(decode 分支):

```python
        gather_context = dcp_context.gather_context                   # :556
        topk_indices = self._remap_sparse_indices(topk_indices)       # :559 ⑤
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)       # :560 ⑥
        sfa_output, softmax_max, softmax_sum = self._npu_sparse_flash_attention(  # :561 ⑦
            ql_nope, q_pe, kv_cache, topk_indices, ...,
            dcp_context.seq_lens,           # 本地长度(rank0=8 / rank1=5)
            sparse_mode=0, return_lse=True) # :569-570
        softmax_lse = softmax_max + torch.log(softmax_sum)            # :572
        return self._merge_dcp_outputs(sfa_output, softmax_lse)       # :573 ⑧
```

(sfa_dcp.py:556-573 节选。`dcp_context` 由 prepare 造好;`kv_cache`
是本 rank 条带——它从头到尾没离开本 rank。)

### 为什么是 query 走、KV 不走:两种通信账

softmax 分母要对全部历史位置求和,rank0 只有半条 KV。直觉方案「收齐 KV」,
先算账:

- **方案 A,每步 all-gather KV**:大小 ∝ 上下文。设全上下文 256K,切 2
  条带、每条带 128K:128K × kv_lora_rank=512 × bf16 ≈ 每条带 128 MB;
  收齐 2 条带 = 256 MB/层;每生成 1 token × 60 层 → 每步搬约 15 GB。
  带宽打爆,**切了等于没切**。
- **方案 B,广播 query**(实际选择):大小 ∝ 本步 token × 头。decode T=1,
  真实配置 64 头 × 576 维 × bf16 ≈ 每层几十 KB,**与上下文长度无关**。

失败画面(选 A 的死法):

```
每步每层 all-gather 256 MB → 60 层 → 每 token 搬 15 GB
→ 带宽打爆,decode 延迟从毫秒级劣化到秒级 → 服务不可用
```

问题本质:找通信量与「每步新增数据」成正比、不与「历史数据」成正比的
分布式注意力方案。答案 = 广播小 query + 寄回小结果 + INDEX 复制让
「去哪找」零通信。

### prefill 的反向经济学

同一个类装着两套经济学。prefill 批(`num_prefills>0`)走反向路:

```python
        if self._has_prefill(attn_metadata):
            gathered_kv_cache = self._finish_dcp_gather(gather_context)  # KV 已收齐
            return self._npu_sparse_flash_attention(
                ql_nope, q_pe, gathered_kv_cache, topk_indices, ...,
                sparse_mode=3,        # 全注意力,不需要稀疏选择
                return_lse=False)     # 手握全量 KV,无需 LSE 合并
```

(sfa_dcp.py:537-554 节选。)反向 gather 在 `_record_dcp_kv_gather_context`
(:253-272):把本 rank 拥有的 latent 块 `index_select` 出来,沿**块轴
(dim=0)all-gather** 给全组。

为什么 prefill 敢搬 KV:**摊薄**。KV gather 一次摊给 T 个 token 的
query(T=整个 prompt);收齐后手握全量 KV 直接算完。

decode 一次只有 1 个 token 无法摊销,才反向设计。一句话:
**prefill 搬大仓库算一次,decode 借小账本算每次**。

---

## D4 · 交换与融合

### D4.1 query all-gather:借头还头(r2-Q7b 上半)

**是什么**:gather 的轴是**头维 dim=1**,不是 token 维。每 rank 出
`[T, H_local, D]`,收齐变 `[T, H_local×dcp, D] = [T, H, D]`;token 不动。

```python
    def _start_dcp_query_gather(self, ql_nope, q_pe):       # :438
        if not has_rope_dim(q_pe):
            return self._start_dcp_gather(
                ql_nope.contiguous(),
                dim=1,                                      # :446 ← 轴=头维
                split_sizes=(ql_nope.shape[-1],))
```

(sfa_dcp.py:438-448。`ql_nope` 是 query 的 NoPE 段 `[T,H_local,D]`;有
RoPE 的模型先 cat 一次传、到端拆回(:456-461)。gather 实现 :315-327:
NCCL 只能拼 dim 0,permute 头维到前再换回——语义仍是沿头维拼接。)

**为什么发起得早**:gather 异步,发出去就返回;⑤ remap 是纯算术,
趁通信在飞算掉;⑥ 才 wait。通信与计算重叠,decode 延迟敏感,白捡的半步。

```python
    def _record_query_gather_context(self, ql_nope, q_pe, attn_metadata):  # :463
        if self._has_prefill(attn_metadata):
            return                                           # :469-470 prefill 不广播
        attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(...)
```

(sfa_dcp.py:463-471。)

**手算**:

```
输入:rank0 出 [1,2,D](头 h0,h1);rank1 出 [1,2,D](头 h2,h3)
变换:沿 dim=1 all-gather,两份头维拼接
结果:两边都收齐 [1,4,D] —— 全部 4 个头的 query
```

为什么 gather 头不 gather token:⑦ 的 SFA 要对**全部头**算注意力(每 rank
要交还「所有条带 × 全部头」的部分结果给头的主人);token 只有本步新增 1 个,
rank 间没有对方的 token 可借——**有的借的只有头**。

失败画面(gather 轴选错):沿 token 维(dim=0)gather,拼出 [2T,H_local,D]
——token 翻倍、头还是缺的;⑦ 拿不到全头 query,⑧ 的打包契约头数对不上,
整条链形状校验爆炸。

「借头还头」:**④ 借入全组的头(替他们算我条带上的部分结果),
⑧ 还回每人自己头的部分结果。借什么还什么,单位都是头。**

### D4.2 载荷 (dcp,H_local,T,D+4) 逐维推导(r2-Q7a)

⑧ 的载荷是 4 维张量,形状由这个函数算出:

```python
def packed_send_shape(num_tokens, num_heads, head_dim, dcp_size, scatter_dim, dtype):
    local_scatter_size = (num_tokens if scatter_dim == 0 else num_heads) // dcp_size
    replicated_size = num_heads if scatter_dim == 0 else num_tokens
    return (dcp_size, local_scatter_size, replicated_size, head_dim + _lse_pack_dim(dtype))
```

(dcp_packed_a2a.py:354-364。`num_tokens/num_heads/head_dim` 来自 SFA 输出
形状(sfa_dcp.py:400);`scatter_dim=1` 调用方写死——头轴散射。)
`_lse_pack_dim`(:346-351):输出是 bf16/fp16 时 LSE 占 4 格;fp32 占 1 格。

我们的调用下 `local_scatter_size = H//dcp = H_local`、`replicated_size = T`:

| 维 | 大小 | 语义 |
|---|---|---|
| dim0 | dcp | **每目的 rank 一片**——片 r 发组内 rank r(all_to_all_single 沿 dim0 等分) |
| dim1 | H_local | 头份额轴(H_full//dcp);片内按**收方**的头排布 |
| dim2 | T | **复制轴**——每个 token 在 dcp 片里各一份(每个头的主人都收到全部 token) |
| dim3 | D+4 | 前 D 格输出向量;后 4 格编码后的 LSE |

**为什么 D+4**:SFA 吐的 LSE 是 fp32(24 位有效数字),载荷是 bf16
(8 位尾数)——一格装不下。解法不是换精度,是把 fp32 的 LSE **手工再编码**
成 4 个各 ≤255 的小整数分装 4 格(D4.3)。小整数 bf16 精确可表示。

**手算(形状账)**:

```
输入:⑦ 部分输出 [T=2, H=4, D=2] + LSE [2,4]
计算:dim0=dcp=2;dim1=4//2=2;dim2=T=2;dim3=2+4=6
结果:send 形状 (2,2,2,6) = 48 格 = 2片×2头×2token×(2数据+4LSE)
```

失败画面(dim0 放错):`all_to_all_single` 沿 dim0 切片、片 r 发 rank r。
若 dim0 放 T 而非 dcp,切出的片不再对齐「一个目的 rank 一片」——收方按
(片, 头份额, token) 解读,拼出的 (token,头) 张冠李戴。**静默错配不崩溃**。

### D4.3 LSE 的 4 格手工浮点再编码(r2-Q7a)

**量纲先钉死**:LSE 是 **per-(token, 头) 标量**——每个 (t,h) 各一个 lse,
载荷 dim3 后 4 格属于且仅属于当前行 (头, token)。

**是什么:不是裸位拆,是手工浮点再编码。**
「4 个 bf16 装 1 个 fp32」若理解成「32 个 bit 切 4 段」就错了——bf16 是
浮点只有 8 位尾数。

正确做法:按数学意义拆开,重编成 4 个各 ≤255 的小整数
(0..255 < 2^8,bf16 精确域)。

编码侧(打包核,dcp_packed_a2a.py:86-100 节选):

```python
            lse_exponent = tl.floor(tl.log2(safe_abs_lse))          # 取指数
            significand = abs_lse * tl.exp2(23.0 - lse_exponent)    # 24 位整数 ∈ [2^23, 2^24)
            significand_hi = tl.floor(significand / 65536.0)         # 高段(÷2^16)
            significand_remainder = significand - significand_hi * 65536.0
            significand_mid = tl.floor(significand_remainder / 256.0)  # 中段(÷2^8)
            significand_lo = significand_remainder - significand_mid * 256.0  # 低段
            exponent_code = lse_exponent + 128.0                     # 格0:指数+128
            exponent_code = tl.where(lse < 0.0, -exponent_code, exponent_code)  # 负号取反
            exponent_code = tl.where(finite_lse, exponent_code, 0.0) # 非有限→0=失效旗
```

(:82-85 的 `finite_lse` 先判废 NaN/±inf。合法码绝对值 ∈ [2,255],
永不取 0——格 0 的 0 兼职「失效旗」。)

解码侧(融合核,dcp_packed_a2a.py:196-201):

```python
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
```

**手算:lse=12 编码(六步)**:

```
1) lse=12.0:finite=真,abs=12,nonzero=真
2) lse_exponent = floor(log2(12)) = 3        (2^3=8 ≤ 12 < 16 ✓)
3) significand = 12 × 2^(23−3) = 12 × 2^20 = 12,582,912   (∈ [2^23,2^24) ✓)
4) hi = 12,582,912 ÷ 65,536 = 192(整除);mid = 0;lo = 0
5) exponent_code = 3 + 128 = 131(正数不取反)
6) 四格 = [131, 192, 0, 0],每格 ≤ 255 → bf16 精确可存
```

**解码验算(四步回 12)**:

```
1) [131,192,0,0]:131≠0 → valid;sign=+1;|131|=131
2) safe_exponent = 131−128 = 3
3) significand = 192×65,536+0+0 = 12,582,912
4) lse = 12,582,912 × 2^(3−23) = 12,582,912 ÷ 2^20 = 12 ✓ 往返零误差
```

为什么无损:全部中间量是整数,存取都在小整数精确域,无任何一步舍入——
这是「再编码」与「类型强转」的本质区别。

失败画面(直接 cast fp32→bf16):bf16 只 8 位尾数,lse=12.0000001 与
12.0000002 强转后可能同值 → 融合权重 w_r 偏差(lse 差 0.1 → 权重比差
约 10%)→ 概率比失真,静默错算。一致性实验 的 0.0 依赖这条无损通道。

### D4.4 AllToAll 头拓扑:rank_idx = head_idx // H_local(r2-Q7b)

先破误区:④ 不是「allgather 了所有 token」——token 谁也没有对方的,
借的是**头**。⑦ 之后每 rank 手握「全部头 × 本 rank 条带」,而最终只需要
自己的 H_local 个头——多余的 1/2 **还给头的主人**(多余占比=1−1/dcp:
本例 (H−H_local)/H=2/4=1/2;dcp=4 时才是 3/4)。

打包核路由(dcp_packed_a2a.py:54-65 节选):

```python
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)
        if SCATTER_TOKENS:
            rank_idx = token_idx // local_scatter_size
        else:
            rank_idx = head_idx // local_scatter_size     # ← 我们走这支
            scatter_idx = head_idx % local_scatter_size
            replicated_idx = token_idx
```

(`linear_idx` 是 (token,头) 展平行号。`rank_idx = h // H_local`:
头 h 放进发给其主人的片;token 轴整行复制进每一片。)

AllToAll 与融合接线(sfa_dcp.py:419-436 节选):

```python
        pack_dcp_output_lse(output, lse_th, dcp_size, scatter_dim=1, send=send)  # :419
        dist.all_to_all_single(recv, send, group=self.dcp_group.device_group)    # :426
        h_local = num_heads // dcp_size                                          # :427
        return fused_dcp_lse_combine(recv, head_dim, scatter_dim=1, output=merged)  # :436
```

(`send/recv` 是静态执行缓冲,形状同 (dcp,H_local,T,D+4);`h_local` 从输出
形状反推,不依赖构造参数——glm5_next 传全局头数、glm5_2 传本地头数的
历史债在此免疫。)

**dcp 组是谁**。构造在 C++ 启动时(py_causal_lm.cpp:307-330 节选):

```cpp
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();  // :307
    if (!is_deepseek_v4 && cp_size_ == 1 && kv_split_size > 1) {            // :308
      CHECK_EQ(tp_size_ % kv_split_size, 0);                                // :309
      const int32_t dcp_stride = tp_size_ / kv_split_size;                  // :310
      const int32_t dcp_rank = parallel_args.kv_split_rank();               // :311
      init_process_group("dcp", ..., dcp_rank, kv_split_size, ...);         // :317
```

主例 tp=2/kv=2(一致性实验 实测拓扑):dcp_stride=1,**dcp 组 = 整个 TP 域**:

```
TP 域(一个 dp 副本)          dcp 组 = {r0, r1}
┌─────────┬─────────┐
│  rank0  │  rank1  │    rank0: tp_rank=0, dcp_rank=0 → 条带0, 头 h0,h1
│ 头h0,h1 │ 头h2,h3 │    rank1: tp_rank=1, dcp_rank=1 → 条带1, 头 h2,h3
│ 条带0   │ 条带1   │    两轴正交:tp_rank 管头,dcp_rank 管条带
└─────────┴─────────┘
```

深水区(kv<tp,如 tp4/kv2):dcp_stride=2,组号公式(:313-316)的组员是
**每隔 stride 取一个**({tp0,tp2}、{tp1,tp3} 交错),连续两 rank 反而是
同条带的 TP 复制。

> 诚实标注:二轮报告「dcp 组=TP 域内连续 kv_split rank」只在 kv==tp
> 字面成立;布局结论引用须钉死代码版本。

**手算:(2,2,2,6) 载荷的一生(四站全链)**

设定:dcp=2,H_local=2,T=2,D=2 → H_full=4;rank0 拥有 h0,h1,rank1 拥有 h2,h3。

第 1 站 · ⑦ SFA 之后(每 rank 各自):

```
输入:全头 query [2,4,2] + 本 rank 条带 KV
输出:partial_out [2,4,2] + lse [2,4]   (每个 (t,h) 一个标量)
```

第 2 站 · 打包(路由 rank_idx = h // 2):

```
rank0:片0=头h0,h1 的行→发给rank0(自留);片1=头h2,h3→发给rank1
rank1:片0(头h0,h1)→发给rank0;片1(头h2,h3)→自留
片内一行 (头h, token t) 的 6 格 = [o0, o1, e码, 尾hi, 尾mid, 尾lo]
```

第 3 站 · AllToAll:

```
rank0 收到:recv[0]=rank0 对头h0,h1 的部分结果(条带0)
           recv[1]=rank1 对头h0,h1 的部分结果(条带1)
→ 恰好「我的头 × 所有条带」,零冗余(h2,h3 的数据一行没收)
```

第 4 站 · 融合(每个 (t,h∈{h0,h1}) 做 D4.5 四步数学):

```
输入:recv [2,2,2,6] → 输出:merged [T=2, H_local=2, D=2] → 交回 o_proj
```

通信量核对:每 rank 收发各 48 格,恰好是它需要的;对比 all-gather 全部
部分输出(收 [dcp,T,H,D+4]=96 格,1/2 是别人的头——48/96 格,即
1−1/dcp(dcp=4 时才是 3/4)——收了也扔)。
一句话:**借头还头,移动小账本,不动大仓库。**

失败画面(散射轴选错):pack 走 `SCATTER_TOKENS` 分支(按 token 切片)
→ rank0 收到「token0 的所有头」而非「我的头的所有 token」→ 融合核按
(头份额,token) 解读,语义对调。**形状合法、数值静默错位——比崩溃更糟。**

### D4.5 valid_lse 防御与 LSE 精确合并(r1-Q3b)

**问题重述**(一轮 Q3(b)):两 rank 各算一份只覆盖自己条带的注意力输出,
怎么合成全局正确的一份?只融 max 和 sum 够吗?

先钉死四个误区:

- 融合主体是**完整输出向量**,不是 max/sum 两个数;
- max 只发生在 **LSE 标量**上,用来生成权重;
- 被交换的是「部分输出 + LSE」整体;
- 融合粒度是**每个 (token, 头)**。

**是什么:四步数学**(每个 (t,h) 各一遍):

```
max = max(lse_0, lse_1)
w_r = exp(lse_r − max)
merged = w_0·O_0 + w_1·O_1        ← 向量逐维加权,不是两个数相加
输出 = merged / (w_0 + w_1)
```

**为什么精确(三行代数,实数域)**。O_r 是局部归一输出:
`O_r = (Σ_{i∈r} exp(x_i−m_r)·v_i)/S_r`,`lse_r = m_r + log S_r`,M = max_r lse_r:

```
① w_r = exp(lse_r−M) = exp(m_r−M)·S_r
② w_r·O_r = exp(m_r−M)·S_r·(Σ_{i∈r} exp(x_i−m_r)·v_i)/S_r = Σ_{i∈r} exp(x_i−M)·v_i
③ Σ_r w_r·O_r = Σ_{全部选中 i} exp(x_i−M)·v_i ;  Σ_r w_r = Σ_i exp(x_i−M)
```

分子分母同缩基准 M → merged = 全局 softmax。∎
两半部分 softmax **精确合并**,不需要把 KV 或分数拼回去。

**手算(统一数字,某一对 (t,h))**:

```
设定:rank0: m_0=10, S_0=e² → lse_0=12;rank1: m_1=9, S_1=e⁴ → lse_1=13
1) max = max(12,13) = 13
2) w_0 = exp(12−13) = e⁻¹ ≈ 0.368;w_1 = exp(0) = 1
3) merged = 0.368·O_0 + 1·O_1   (D 维逐维)
4) 输出 = merged / 1.368 → rank1 占 ≈73%,rank0 占 ≈27%
```

窗户纸:**w_r 不是「rank 的重要性」,是这个 rank 拥有的位置在全局
softmax 质量里的份额**。验算:rank0 折到基准 13 的总质量
= exp(m_0−13)·S_0 = e⁻³·e² = e⁻¹ = w_0 ✓。

三种错误融合的失败画面:

```
画面一 · 直接平均:正确 73/27,平均各 50 → 输出偏,偏差随 lse 差指数放大
画面二 · 只融 max/sum:O_0/O_1 各只对自己位置归一,手里没有全局正确的向量
        —— 流程走不通,不是精度问题
画面三 · 按位置对齐:位置列表从未交换,两 rank 有效集合不同长 → 无可对齐的轴
```

**valid_lse 防御:全 -1 行为什么不污染**。⑤ remap 后本 rank 一个选中位置
都没有的行整行 -1;SFA 首停第 0 列,该 (t,h) 的部分结果是空,LSE 可能是
NaN 或垃圾。融合核三重防御(dcp_packed_a2a.py:196-214 节选):

```python
                packed_valid = exponent_code != 0.0        # 格0=0 → 失效旗
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)     # :203
            partial_output = tl.load(...).to(tl.float32)                      # :204-208
            partial_output = tl.where(valid_lse, partial_output, 0.0)         # :209
            merged += partial_output * weight                                 # :210
            weight_sum += weight                                              # :211
        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)             # :213
        merged /= denominator                                                  # :214
```

- `weight=0`:无效份权重归零(不进分子);
- `partial_output=0`:无效份向量归零(载荷是垃圾也不进分子);
- `denominator` 保底 1:所有份都无效(全 -1 行)→ 输出零向量而非 0/0=NaN。

失效旗的来源即 D4.3:编码侧对非有限 LSE 把格 0 置 0(:100)。
手算走查(某 (t,h) 上 rank1 全 -1):

```
输入:recv[0] 正常(lse_0=12);recv[1] 格0=0(失效旗)
计算:valid_1=假 → w_1=0、partial_1=0;w_0=exp(12−12)=1
结果:输出 = (1·O_0+0)/(1+0) = O_0 —— 空份完全出局,不污染
```

端到端佐证:一致性实验 真机实测(P@cp4/tp2/kv2 + D@cp1/tp2/dp4/kv2),
greedy 文本逐位一致、48/48 logprobs 相同、**max|dlogprob|=0.0**
[主会话/二轮报告转引]。

> ⚠️ 证据边界:SFA 算子(CANN 闭源)对 attention_out 的归一约定无法直读;
> 融合公式仅对「局部归一输出」精确,端到端 0.0 偏差反推约定必然如此。
> SFA「首停」的核内行为同闭源,由 xllm 侧 docstring 与打包契约交叉印证。

出身注:模块移植自 vllm-ascend 的 sfa_cp.py(PR 14457,模块头注 :14-17);
数学是社区验证过的成熟方案,本仓库自研部分是接线与防御。

**核心结论**(r1-Q3b):不止 max/sum——交换完整输出向量+LSE,
LSE 取 max 定权重,向量加权求和后归一,每 (token,头) 精确一次。

---

## D5 · 图模式 padding

### D5.1 三件套:块 0 / 槽 -1 / 长 1(r2-Q8a-c)

前置概念(首现即教):

- **ACL graph(图模式)**:把 decode 一步的整串算子**录制**成执行图,
  每步「重放」,省去逐步调度开销。硬约束:**重放时所有输入张量形状
  必须与录制时一模一样**。
- **padding 行**:真实 batch(2)补齐到捕获尺寸(4)时虚构的 batch 行;
  结果最终丢弃,但**计算过程必须安全跑完**。
- **eager 模式**:不录制,真实 batch 直接跑,没有 padding 行。
  两种模式跑**同一批 kernel**;-1/0 之分不在模式,在张量种类(D5.2)。

图捕获时(一次性)构造静态缓冲(decode_acl_graph.py:1409-1414 节选):

```python
        static_block_table = torch.zeros(
            padded_batch_size, self._max_blocks_per_sequence,
            dtype=torch.int32, device=device)      # 块表全 0(占位)
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(padded_batch_size, dtype=torch.int32, device=device),
            ...
            kv_seq_lens_host_values=[1] * padded_batch_size,   # :1480 长度全 1
```

(这只是**录制时的占位值**,每次重放都被下面的回填覆盖。)

每次 decode 重放前回填(decode_acl_graph.py:1727-1741):

```python
            static_metadata.block_table[:batch_size, :copy_cols].copy_(src_bt[:batch_size, :copy_cols])
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()      # :1735 块表打 0
        # Padded lanes must remain valid inputs for sparse MLA tiling.  Their
        # token and slot mapping are dummy values, so one KV token is safe.
        if padded_batch_size > batch_size:
            static_metadata.slot_mapping[batch_size:].fill_(-1)       # :1740 槽打 -1
            entry.kv_seq_lens_delta[batch_size:].fill_(1)             # :1741 长度打 1
```

逐块读:真实行原样拷入 `[:batch_size]`(eager 消费的就是这份);padding 行落
**三件套:块 0 / 槽 -1 / 长 1**。

注释点破意图:padding 行必须是稀疏 MLA tiling 的合法输入,一个 KV token
安全。同款处理还有两处:DSA 管理表(:1858-1860,`permanently reserved
block 0` → `target.zero_()`)与 host 侧 KV 长度(:1940-1947,补 [1]×n)。

**手算:三件套在统一数字下的行为**

图捕获 batch=4、真实 2、page=4、dcp=2;行 0,1 真实,行 2,3 padding。

第一算 · expanded(padding 行,喂 indexer):

```
输入:块表 padding 行 = [0,0]
计算:页 = E×dcp+s:0×2+0=0;0×2+1=1(每列同理)
结果:expanded 行 = [0,1,0,1] —— 全合法页号(细节 D5.2)
```

第二算 · 写路径(喂写 KV 的核):

```
输入:slot_mapping 行 = -1
计算:localize_slots 对负槽判无效(kv_shard_layout.py:80),原样 -1
结果:local_slots=-1 → 写核跳过,块 0 不被虚构数据污染
```

第三算 · local_seq_lens(长度 1 摊两 rank,kv_shard_layout.py:69-77):

```
输入:kv_seq_lens=1;逻辑块尺寸 = page×dcp = 8
计算:full=1//8=0 整块;rem=1
      rank0: 0×4+clamp(1−0,0,4)=1;rank1: 0×4+clamp(1−4,0,4)=0
结果:rank0 的 SFA 看到 1 行(tiling 有效);rank1 看到 0 行
```

**padding 行真执行,浪费 ∝ 行占比**(答 r2-Q8 原话「是真执行吗,不浪费算力吗」):

- 真执行:录制好的算子流不认识「虚构」,padding 行照常走完 ② 打分→
  ⑤ remap→⑦ SFA→⑧ 打包全链(topk 被清 -1,D5.3;SFA 空转;
  融合时 valid_lse 出局);
- 浪费 ≈ 行占比:捕获 4、真实 2 → 50% 行陪跑(真实 5/捕获 8 → 37.5%);
- **kv_len=1 把每行陪跑代价压到最小**:注意力量 ∝ query 数 × KV 长度,
  KV 长度取 1(而非 max 上下文)→ 每行只算 1 个 KV 位置;
  桶分档(batch 分桶捕获)从另一头压行占比。

失败画面(三件套各拆一件):

```
拆块表(0→-1):expanded 出现 -1 → CANN indexer 当无符号页号 0xFFFFFFFF
  ×页大小寻址 → MTE 越界崩溃,或静默错 top-k(更糟)。展开 D5.2。
拆槽(-1→0):写核把虚构 token 写进全局槽 0(属真实序列)
  → 污染真实数据,静默错算。
拆长度(1→0):SFA 的稀疏 MLA tiling 遇 kv_len=0 → 除零/空区间
  → 核报错或图捕获失败。
```

三件套 = 三类消费者契约的最大公约数(下节一张表收口)。

### D5.2 页 0..dcp-1 与三类消费者契约(r2-Q8b)

**「0..dcp-1 都是占位符吗」——不是常量,是保留块 0 经合法公式展开的结果**:

```python
        if self._block_table_i32 is not None:
            # ACL graph padded lanes keep logical block 0 and kv_seq_len=1 so
            # sparse MLA tiling stays valid. Expanding that row yields pages
            # 0..dcp_size-1. Do not rewrite those pages to -1: LightningIndexer
            # treats -1 as a huge page id and MTE-OOB.
            expanded = self._kv_layout.expand_indexer_block_table(self._block_table_i32)
```

(sfa_dcp_backend.py:157-162。)`expand_indexer_block_table`
(kv_shard_layout.py:121-138)把块表值 E 展开成 [E×dcp, E×dcp+1, ...]。

手算:padding 行块 0 → 页 0×2+s = **[0,1]**——与真实行 [5,7]→[10,11,14,15]
同一条公式,非特例。

注释即设计者写下的失败画面:`Do not rewrite those pages to -1:
LightningIndexer treats -1 as a huge page id and MTE-OOB`。

**块 0 为什么读着安全**:永久保留块(:1859 注释 `permanently reserved
block 0`),真实序列也用它;padding 行去**读**无害,**写**被槽 -1 挡死。
一读一写,两头堵死。

**-1/0 分界线:张量种类,不是模式**。三类消费者契约:

| 张量种类 | 例子 | 消费者 | 对 -1 的行为 | 结论 |
|---|---|---|---|---|
| slot 列表 | slot_mapping、local_slots、remap 后索引 | 写 KV 核 / SFA | 跳过 / 首停 | **-1 安全** |
| 页表 | expanded、INDEX 读表 | CANN indexer 核(pool_key_indexer/LightningIndexer 族) | 当无符号页号寻址 | **-1 致命,必须全合法** |

eager 与 graph 同一批算子壳(indexer 侧纯转发 `EXEC_NPU_CMD(aclnnLightningIndexer,
...)`,lightning_indexer.cpp:93-108 [一轮书转引]);分界在**收件人**。

失败画面(expanded 写 -1 五步):

```
1) padding 行块表 0 改 -1 → expanded 行 = [-1,-1,-1,-1]
2) 交给 indexer 核 → 拿 -1 当页号:int32 下 -1 = 0xFFFFFFFF(约 43 亿)
3) 乘页大小寻址 → 远超 INDEX 张量边界
4) MTE(Memory Transfer Engine,昇腾访存引擎)越界
5) 崩溃;侥幸不崩则静默错 top-k(glm5_next.py 注释原话
   `silently returns wrong top-k sets`)
```

> ⚠️ 证据边界:「-1→巨大页号→MTE 越界」发生在闭源 CANN 算子内;仓库内
> 证据 = 两处注释 + 两处 clamp + 测试断言,无法直接验证核内寻址。

这不是 padding 专属防御——**真实行**的 expanded 也可能含 -1(逻辑块没被
页背书),处理是 clamp:

```python
            # The DCP-expanded indexer table carries -1 for logical blocks that are not
            # backed by a page. The dense read path clamps those entries
            # (glm5_next_kpool.py), while forwarding -1 to the kernel makes it address a
            # huge page id and silently returns wrong top-k sets.
            block_table=fused_block_table.clamp(min=0).to(dtype=torch.int32).contiguous(),
```

(glm5_next.py `pool_key_indexer` 调用点 [一轮书转引];torch 读路径同款
`clamp(min=0)`。宁可读块 0 的无效数据,也不让 -1 见核。)

**核心结论**(D5.1+D5.2):三件套 = 块 0(喂 indexer,合法页)/ 槽 -1
(喂写核,跳过)/ 长 1(喂 SFA,tiling 成立);分界永远是
「**这张表的读者拿到 -1 会做什么**」。

### D5.3 勘误:attention_mask 恒全 1,清 -1 的真实机制(r2-Q8d)

**勘误声明**(对一轮书第 4 章的正式修正 [二轮报告]):一轮书曾写
「attention_mask 对 padding 行为 False(引擎按真实长度构造掩码)」——
**不被源码支持**。

```python
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden.device)
```

(glm5_next.py:2775-2776。decode 的 mask **缺省恒全 1**。)
图模式更直接——重放只传两参:

```python
    def _forward_static(self, entry: _DecodeGraphEntry) -> ModelExecutionOutput:
        if entry.static_input_embedding is None:
            return self.model(entry.static_input_ids, entry.static_positions)   # :2065-2066
```

(decode_acl_graph.py:2064-2066。`attention_mask=None` → 缺省全 1;
eager 与 graph 同——mask 从没按真实长度构造过。)

那 padding 行的 topk 怎么变成 -1?**真实机制三件,与 mask 无关**:

机制一 · **span 剥离(padding 行根本不进打分)**:

```python
        if kpool_query_lens is not None and sum(kpool_query_lens) < num_tokens:
            # Spans list complete requests only. Bucket rows after that prefix
            # are padding: keep them out of the tail update, then pad top-k
            # back so attention still sees the captured bucket.
            keep_rows = sum(kpool_query_lens)
            hidden_states = _prefix_token_tensor(hidden_states, num_tokens, keep_rows)
            ...                                  # key/weights/qr/positions 同裁
            num_tokens = keep_rows
```

(glm5_next.py:1342-1354。`kpool_query_lens` 是请求 span 列表,只列完整
请求;span 前缀之后的桶行就是 padding,**打分之前就被裁掉**。)

机制二 · **F.pad(-1) 把 topk 行补回桶尺寸**:

```python
        if num_tokens < full_num_tokens:
            topk_indices = F.pad(topk_indices, (0, 0, 0, 0, 0, full_num_tokens - num_tokens), value=-1)
```

(glm5_next.py:1569-1570。padding 行以 **-1 行**身份补回——静态图要求
topk 行数=捕获尺寸。**这才是 padding 行 topk 全 -1 的出生地**。
同族:行宽不足 output_width 时 F.pad(-1) 补宽,:1294-1295。)

机制三 · **validity 过滤(真实行的 -1)**:

```python
            topk_indices = topk_indices.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
            ...
            pos_visible = (safe_pos <= q_pos[:, :, None]) & kv_ok & attention_mask[:, :, None]
            topk_indices = topk_indices.masked_fill(~pos_visible, -1)
```

(glm5_next.py:1276-1278/:1288-1289。`masked_fill` 从零定义:把条件为
真的格子整格改写成给定值。拦无效候选的是 `selected_valid`/`kv_ok`;
:1289 的 attention_mask 参与的是因果可见性,decode 时全 1、不拦人。)

**remap 从零定义**(padding 行 topk 的下游,三轮主线 [三轮报告]):
把 topk 里的**全局位置 p** 翻译成本 rank 的**读侧索引**:

```
p → b = p//8 → owner = (p%8)//4
  → 本 rank:L = b×4 + (p%4);peer:整格 -1
  → 打包有效前置(kv_shard_layout.py:102-119,排序键=列号+无效列加 width,键全唯一)
```

(padding 行 topk 整行 -1 → localize 判无效 → 保持 -1 → SFA 首停第 0 列
→ 空结果 → ⑧ 打包后 LSE 无效 → D4.5 valid_lse 出局。**六步闭环,
无一步依赖 attention_mask。**)

失败画面(信了旧书的说法):

```
1) 以为 padding 行 topk 被 attention_mask 屏蔽 → 去找「按真实长度构造掩码」
   的代码 → 找不到(不存在)
2) 若真按 mask 全 1 放行且无 span 剥离:padding 行会真打分
   → 读块 0 展开的页 0,1 的 INDEX 数据 → topk 选出垃圾位置
3) 实际没炸,是因为 span 剥离(:1342-1354)在 mask 之前裁掉了 padding 行
   —— 防线在别处,旧书把功劳记错了人
```

> ⚠️ 证据边界:融合 indexer 核对压缩尾路径 padding 行的核内行为闭源
> [待验证][二轮报告];本节三机制的 xllm 侧代码均实读核验。
> 二轮成员报告的行号(2774-2775/1348-1357 等)与本次实读有 ±1~6 漂移,
> 一律以实读为准。

---

## 收束:一章一句话

**借头、还头、LSE 合账,通信只随新增数据走;padding 行按「消费者的
契约」喂块 0/槽 -1/长 1——mask 从不挡人,挡人的是 span 剥离与失效旗。**

## 自检

| # | 检查项 | 状态 | 证据位置 |
|---|---|---|---|
| 1 | 八段旅程图 + KV 不跨 rank 主线 + 两通信边界 | ✓ | 机制总览;sfa_dcp.py:556-573 |
| 2 | Q-gather 不 KV-gather:通信 ∝ token×头 vs ∝ 上下文 + 失败画面 | ✓ | 两种通信账(128MB→15GB) |
| 3 | prefill 反向经济学:KV gather 摊薄/sparse_mode=3/return_lse=False | ✓ | sfa_dcp.py:537-554/:253-272 |
| 4 | query all-gather 轴=头维 dim=1,token 不动;借头还头一句话 | ✓ | D4.1;sfa_dcp.py:438-448/:315-327 |
| 5 | 载荷 (dcp,H_local,T,D+4) 逐维推导表 | ✓ | D4.2;dcp_packed_a2a.py:354-364 |
| 6 | 4 格再编码:格0=sign×(exp+128) 兼失效旗、尾数三段、每格≤255 | ✓ | D4.3;dcp_packed_a2a.py:86-100/:196-201 |
| 7 | **冷读者能否跟着算出 lse=12 的 4 格编码** | ✓ | D4.3 编码六步(→[131,192,0,0])+解码四步验算回 12 |
| 8 | **冷读者能否跟着算出 (2,2,2,6) 载荷切分** | ✓ | D4.4 四站走查(48 格账) |
| 9 | AllToAll 路由 rank_idx=head_idx//H_local;TP×DCP 网格(kv==tp 主例+kv<tp 诚实标注) | ✓ | D4.4;dcp_packed_a2a.py:54-65 + py_causal_lm.cpp:307-330 |
| 10 | valid_lse 三重防御(旗立&非NaN&非±inf→weight=0/partial=0/分母保底1) | ✓ | D4.5;dcp_packed_a2a.py:196-214 |
| 11 | LSE 合并三行代数 + 一致性实验 max\|dlogprob\|=0.0 | ✓ | D4.5;一致性实验 标注转引 |
| 12 | 三件套构造与回填贴码;真执行、浪费∝行占比、kv_len=1 压最小 | ✓ | D5.1;decode_acl_graph.py:1727-1741;50%/37.5% 例 |
| 13 | 页 0..dcp-1=保留块 0 经 E×dcp+s 展开(非占位符) | ✓ | D5.2;sfa_dcp_backend.py:157-162 + [0,1] 手算 |
| 14 | 三类消费者 -1 契约表 + expanded 写 -1→MTE 越界五步失败画面 | ✓ | D5.2;⚠️ CANN 闭源边界 |
| 15 | 勘误:mask 恒全 1(图/eager 同);清 -1 靠 span 剥离+F.pad+validity;masked_fill/remap 从零定义 | ✓ | D5.3;glm5_next.py:2775-2776/:1342-1354/:1569-1570/:1276-1289 |
| 16 | 每机制四件套:主干+贴码+手算(中间状态)+失败画面 | ✓ | 9 机制逐一核对 |
| 17 | 前置概念首现即教(dcp 组/H_local/LSE/AllToAll/ACL 图/padding 行/masked_fill/remap) | ✓ | 各节首段 |
| 18 | 统一数字 page=4/dcp=2;编号镜像 r1-Q3b/r2-Q7a/Q7b/Q8 | ✓ | 统一数字框;节标题 |
| 19 | 段落≤3 行;贴码块 3-14 行;演算分步成行 | ✓ | 全章 |
| 20 | 诚实:行号实读、二/三轮标来源、⚠️ 边界 4 处、拓扑表述精确化 | ✓ | 卷首声明;D4.4/D4.5/D5.2/D5.3 |

> 遗留说明(给主编):① lightning_indexer.cpp:93-108 与 glm5_next.py
> clamp 调用点为一轮书转引(src/ 未物化该 cpp);② 一致性实验 数字与
> `silently returns wrong top-k sets` 注释为成员报告/主会话转引;
> ③ py_causal_lm.cpp 任务书标 :305-330,实读块起于 :307,以实读为准;
> ④ 二轮报告行号漂移已在 D5.3 末尾披露。

---

## 索引层(外围,结论级)

### 工程决策与验证

- **为什么 Q-gather 不 KV-gather**:每步通信 ∝ token×头(几十 KB)而非上下文(GB 级);方案对比:A all-gather KV(带宽打爆,稀疏压缩红利被复制放大 dcp 倍;256K 上下文×512×bf16×60 层≈每步 15GB,通信账演算见第 3 章总览)/B 分数 all_reduce 归约再 topk(每层每步对整张分数图同步)/C 复制各自重算(显存换通信,INDEX 每 token 仅 128 维 vs latent 512+ 维)。选 C,同选 AllToAll 头轴散射+单载荷(一次通信搬完、零冗余;复盘:decode 小包多次场景,通信次数是延迟主导项)。
- **为什么 INDEX 复制**:indexer 打分要全量历史;复制换读零通信,且与 prefill PCP 复制写同构,PD 传输两边通用同一资源描述(cache_layout_builder.cpp:118-149 describe_replicated_index_pages:一段连续 dcp 页无空洞无跳排);NPU 平台 `requires_dsa_indexer_cache_replication()` 强制。sharded 写模式的坑:每 rank 只写自家条带进页组第 0 页,组内其余页陈旧,RANK_LOCAL 1:1 搬运会把陈旧页当有效页搬过去——replicated 模式任何单个 writer 都能供出全部有效页。
- **M9 开关**(讨论催生):`XLLM_CP_INDEX_WRITE_MODE`(replicated 默认/sharded 支线)——用户判"无显存收益+引入 allgather 是纯纯负向优化"方向纠偏:latent 才是省显存的地方,INDEX 本来就每 rank 全量复制;sharded 只对 owner-sharded 读路径正确,是半成品默认关。
- **拓扑锚定 cp=1**:master 门控对 cp>1 要求"kv_split>1 必须 PD+PREFILL 角色",cp=1 提前返回不做此检查(设计而非漏洞)——正确性由后端选择保证(executor.py:126-176:NPU+cp1+dcp 组>1+MLA → SfaDcp),准入配对测试钉死准入-选择配对。
- **PD 宽度协商**(kv_cache_transfer.cpp:97-145):等宽→RANK_LOCAL(1:1 块映射;src kv_split>1 且 sharded 写模式→reject,stale INDEX 页风险);D 侧 kv1→STRIDED;其余 UNRECONCILABLE。一致性实验 链接日志 `Successfully linked instance, prefill_kv_split_size: 2`。
- **验证神谕**:等宽 D(kv2 SfaDcp) vs 交错 D(kv1) 对照——greedy 逐位一致、48/48 logprobs、max|dlogprob|=0.0;P@103 cp4/tp2/kv2 → D@99 cp1/tp2/dp4/kv2(24 卡真机)。
- 遗留:overlap×SfaDcp 组合当时未合测;spec-verify 宽批掉速回退。

### 讨论独有沉淀(会话考古,书未覆盖;2026-10-13 重结算挖掘,本版全量继承)

- **pr370 同构 vs CP owner-shard 的本质冲突**:等宽 D(kv2) 走 SfaDcp 需要 INDEX 全副本由双 owner 各供一页,而 reshard planner 的 per-resource overlap 每资源只能路由一页——"数学上表达不了";候选修法"+rank"写行是 vacuous fix(owner≥1 会推自己从未写过的行,测试全绿但症状依旧)。
- **M7 传输损坏根因链**(故障分析范例):manifest 与字节级 plan 均正确,损坏在 block-id 映射层——width-blind 的 rank_local_mapping 不看 D 声明的 kv_split_size,异构 2→1 对错走 1:1 映射,stripe-1 owner 用陈旧内存覆盖有效 prompt KV(逐层 last-writer-wins),token 0 精确(token 1+ 全错)。
- **prefix admission 的生产意义**(三轮施压后的认知修正):prefix 门是 **host offload 的前置条件**——validate_host_cache_options 要求 prefix 开启→不放行 prefix 则 `--host_blocks_factor>1` 设不了→host offload(Mooncake store 分层)对 glm5_next CP 永远关死;"不是省 prefill 算力那点收益,是打开 host 分层"。dengyingxu1 主线(#462)=prefix 放行→host offload→线性检查点进 Mooncake 分层;本会话 device 层 PD 复用是支线。glm5.3-flash 需重做 prefix 的根因:混合线性架构的前缀复用=KV 块+LINEAR 检查点双轨,而 LINEAR 检查点有发布时机断点(发布循环跳过 LINEAR、cache_read_source 依赖"下一次 allocate_prefill"、PD 交接后不存在"下一次"——同时解释 M4 零复用)。
- **PD 三路线裁决**:A(D 侧 bookkeeping)/B(P 检查点存活)/C(trim 放宽),用户拍板 B;四路上游横比(vLLM delay_free→evictable 池/SGLang release(checkpoint=True) 入树)证实 P 侧 cached-but-evictable 是生态共识。
- **D 侧传输跳过"半个机制"**:remote_shared_num→advance_transfer_block_idx 链路存在(D try_allocate→probe 前缀→命中变 shared_blocks_num→P 从 cursor 起推)但端到端未测;P 自身命中只进记账不缩传输;正确第一步是先测量不是写代码。
- **scheduler overlap 真相**:调度器是 C++ 不是 Python;greedy argmax over CP-replicated logits 跨 rank 字节一致;non-greedy 每 rank 本地采样、RNG lockstep 只在 init 种子,分歧真实存在。

### 证据边界(⚠️ 四条,书内显式标注;散落深潜层各处,此处汇总)

①SFA"首个 -1 停"与写核"-1 跳过"在 CANN 侧闭源(证据=四处 docstring+_fill_wide_remap 防御代码的存在动机);②LightningIndexer/pool_key_indexer 把 -1 当巨大页号→MTE 越界闭源(证据=两注释+两 clamp+测试断言);③"逐位一致"由相同输入+复制权重+确定性 kernel 推出,无显式跨 rank 比对测试(端到端佐证 一致性实验 max|dlogprob|=0.0);④SFA 对 attention_out 的归一约定无法直接读(由融合代数+端到端 0.0 偏差反推)。另有:融合核 padding 行核内处理[待验证]、融合 indexer 输出顺序闭源、padding 浪费无 benchmark、个别行号 ±1 漂移(以分支物化文件实读为准)。

### 关键源码锚点(分支 feat/glm5-next-sfa-dcp-decode @ a2edee8498)

| 机制 | 位置 |
|---|---|
| 后端类/prepare/execute_mla | sfa_dcp_backend.py:15/:134-202/:223-276 |
| 执行者(remap/merge/gather) | sfa_dcp.py:29/:66-110/:315-383/:385-471/:526-573 |
| 几何算术(localize/pack/expand) | kv_shard_layout.py:69-138 |
| select_topk 坐标/行宽/F.pad | glm5_next.py:1090-1105/:1261/:1293-1296/:1342-1357 |
| indexer 复制注释/打分循环 | glm5_next.py:3058-3065/:1221-1236;glm5_2.py:44/:896-902/:680-692/:557-561 |
| 4 格编码/valid_lse/shape/路由 | dcp_packed_a2a.py:62-69/:82-116/:142-214/:354-364 |
| dcp 组创建 | py_causal_lm.cpp:307-330 |
| 页表不变量(本地页=块号)/C++ 孪生 | npu_cp_plan.cpp:914-953 |
| INDEX 写直通/materialized 表 | npu_paged_attention.py:1230-1237/:1391-1395/:730-751 |
| 图 padding 三件套/真执行 | decode_acl_graph.py:1406-1480/:1727-1741/:2065-2066 |
| 后端选择 | model_executor/executor.py:126-176 |
| 宽度协商 | framework/kv_cache_transfer/kv_cache_transfer.cpp:97-145 |
| INDEX 页数/资源描述 | cache_layout_builder.h:35-49;cache_layout_builder.cpp:118-149;kv_cache_shape.cpp:455-481 |

## 覆盖率表(书章节 + 追问子问 → 深潜/索引裁决)

| 输入 | 深潜承接 | 理由 |
|---|---|---|
| 一轮书 Ch1 谁在算(indexer 复制) | D2.1-D2.2 | 用户 r1-Q1/r2-Q1 两轮追问 |
| 一轮书 Ch2 INDEX 几何与三张表 | D1 全章 | r1-Q2/Q4、r2-Q3/Q4/Q5、r3 反复回踩 |
| 一轮书 Ch3 打包交换与 LSE | D3.2 + D4 全章 | r1-Q3、r2-Q6/Q7a/Q7b |
| 一轮书 Ch4 图 padding | D5 全章 | r1-Q5、r2-Q8(a-f) |
| 一轮书卷首旅程/工程决策 | D4.0 总览 + 索引层 | 总览深潜;选型细节结论级(未被追问展开) |
| 二轮书(已删,经成员报告) | D2.3/D3.1/D3.3/D4.4/D5.3 各判决点 | 九问判决全数落位对应深潜小节 |
| 三轮书(已删,经成员报告) | D2.3/D3.1/D3.2 | mask 四子问题判决落位 |
| 三轮用户子问 (a)(b)(c)(d) | D2.3(a)、D3.1(b)(d)、D3.2+D2.3(c) | 追问精华索引逐条可查 |
| 10-13 重结算:会话考古讨论独有沉淀(M9/M7/prefix/PD 裁决等) | 索引层「讨论独有沉淀」节 | 讨论内容非书内容,结论级即可 |

体量对照:深潜层 2486 行 / 一轮书核心章节(Ch1-4)约 2530 行 ≈ 98%,远超 v2 下限(核心章节 30-50%);索引层(含讨论独有沉淀)~160 行。

## 冷读验收记录(v2 硬门)

- **方式**:三个未读原料、未参与写作的干净视角代理,分章逐机制三问(①不看别处能否复述机制主干②手算能否跟着算出书中数字③术语是否先教后用),判决分段增量落盘(coldread-d1/d2d3/d4d5.md,工作区)。
- **结果**:**23/23 通过**——D1 6/6、D2D3 8/8 首过;D4D5 8/9,机制 5(AllToAll 头拓扑)②手算不通过(两处"3/4"比例与统一数字 dcp=2 矛盾,应为 1/2=1−1/dcp),按红线打回写手角色修复(3/4→1/2+口径钉死+顺修 D+4 记号),修复后干净代理定向复验通过。
- **非阻断观察**(冷读者记录在案,未触及机制核心):外围术语首现未教已由卷首「术语速查」结构性兜底;D2D3 的 rate 首现晚于勘误手算(D3.3 正式定义);"七条件"实为 9 个 and 项等。
- 本节为验收记录摘要;详细判决文件随工作区处置,核心结论已在本节与勘误链留存。

## 来源与处置(诚实声明)

- **深潜层写作输入(本版 10-16)**:一轮书(2707 行,自 ob_note git 历史 commit 5fed7d9 恢复,分章直读)+ 本会话考古(session-extracts.md 2925 行:原始教学 3 万字/三轮成员报告/用户追问原话)+ 源码 18 文件(分支 feat/glm5-next-sfa-dcp-decode 重新物化,全部贴码行号实读核验)。每章由独立写作代理从原料直读写出(非摘要转写),分步落盘。
- **索引层继承(10-13 重结算,commit 9fc33f7)**:三轮书 6331 行经子代理逐本全文通读(5q/9q/round3 分 8/7/3 段,产出 312/276/281 行高保真摘要)+ 两会话 466 条用户消息全集考古——「讨论独有沉淀」「工程决策」「证据边界」「源码锚点」四节由该版全量继承并入本页索引层。
- **二/三轮书正本已删**(commit ec92de6,10-13 重结算后用户拍板):本版深潜层经成员报告判决+会话考古+源码重验重建;如与原书有出入以源码为准。
- **版本链**:v1 结算(10-10,commit a6e2d5a,253 行索引文体)→ 重结算(10-13,9fc33f7,修复虚假吸收+讨论独有沉淀)→ v2 深潜重建(本版,2637+ 行,冷读验收 23/23)。
- **v2 重建工作区**:`~/.dsh/teaching-pilots/sfadcp-v2rebuild/`(原料恢复/三章草稿/冷读判决),验收通过后按用户指令处置。
- 关联:2026-10-03_glm53_flash_pcp_mr418_421_deep_dive(四连前置 MR)、2026-10-07_xllm_mr370_kv_split_pd_reshard_deep_dive(PD 传输与 SFA 三件套)、2026-10-08_xllm_kv_split_rank_layout_four_laws(rank 布局四律,D4.4 深水区互证)、2026-10-08_xllm_mr425_overlap_kvsplit_admission_deep_dive(宽度准入)。
