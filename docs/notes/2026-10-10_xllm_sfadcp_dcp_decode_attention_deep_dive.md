---
title: SfaDcp:GLM-5.3-Flash decode KV 分条注意力三轮教学结算
date: 2026-10-10
updated: 2026-10-13
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
source_count: 4
status: stable
difficulty: advanced
---

# SfaDcp:GLM-5.3-Flash decode KV 分条注意力三轮教学结算

> 一句话:**SfaDcpAttentionBackend 让 dcp 组内每个 rank 只存 KV latent 的 1/kv_split 条带,靠"广播小小的 query、寄回部分输出、LSE 精确合并"买回全量注意力——KV 从不跨 rank 移动,INDEX 全量复制让"去哪找"零通信**。数据按"消费者的契约"摆放、复制和填充:INDEX 为共享而复制,页表为寻址而合法,slot 为跳过而 -1。

> 本页为**重结算版**:2026-10-12 首次结算只抽样读了 82 行书就写下"已全吸收"(151 行薄笔记,note-settle 虚假吸收教训的直接案例);本版按修补后的 skill 三道硬门重做——三轮书 6331 行经子代理逐本全文通读、两个会话 466 条用户消息考古完毕、章节覆盖率表见文末。

## 场景:三轮教学与勘误链

主题源自会话「pcp-小步」的 M11 系列:给 glm5_next 启用 D(kv_split) decode,真机 24 卡验证 `max|dlogprob|=0.0`(P@103 cp4/tp2/kv2 → D@99 cp1/tp2/dp4/kv2,分支 `feat/glm5-next-sfa-dcp-decode`,快照 3bdf81ef2/tip a2edee8498)。围绕它做了三轮教学 + 一轮单会话首轮教学:

1. **一轮(5 问,2707 行)**:query 全量/INDEX 自然行/打包与 LSE 融合/三张表/图 padding;
2. **二轮(9 问,2746 行)**:glm5_2 indexer 对照、kPool mask、E/b 兑换、两个"5"、必选尾、载荷 D+4、AllToAll 头拓扑、padding 真执行;
3. **三轮(1 问 4 子,878 行)**:mask 的坐标空间与 remap 链。

**勘误链(读旧材料以本页为准,四条主裁决)**:
1. **一轮书 3.4 三空间压成一个数**(上一轮实错):把位置 13 当槽 13 用、把读侧索引 5 说成"写槽"。正确:表 [5,7] 时第 14 个 token 的位置=13、全局槽=61(7×8+5)、写槽=29(rank1 本地页 7 行 1)、读索引=5——四个数四个空间。这是用户 Q2 两轮"似懂非懂"的病根;恒等表特例 [0,1](E=b)下三空间数值重合,旧材料把特例当通例,"自洽地错"。
2. **二轮书"attention_mask 对 padding 行为 False"不被源码支持**:decode(图/eager 同)的 attention_mask **恒全 1**(glm5_next.py:2774-2775 缺省 ones;图模式只传两参 decode_acl_graph.py:2065-2066)。真正清 -1 的是三机制:span 剥离+F.pad(-1)(:1348-1357/:1293-1296)+ causal/validity 过滤(:1246-1252/:1276-1277)+ 融合核核内(闭源待验证)。
3. **二轮手算例 C 的 `[0,2,3,-1]` 与源码不符**:topk=3、历史 4、恰好选中 3 个时真实行是 `[0,2,3]`、宽 3、无 -1(F.pad 不触发);引文 -1 是行宽示意(真实配置 topk=2048 时同场景=[0,2,3,-1×2045])。
4. **二轮 pack"位置→全局槽→owner"表述**:读侧 remap 是纯位置算术,从不物化全局槽 61(61 只在写侧 prepare 出现)。

## 结论先行(十五问判决索引)

| 问 | 判决 |
|---|---|
| indexer query 切 TP 吗 | **不切**。glm5_2/glm5_next 的 indexer 三投影(wq_b/wk/weights_proj)TP 全复制,打分+topk 单核零归约;"glm5.2 64 头+allreduce sum"是**主注意力**(64 头、TP 切、o_proj all_reduce)与 kernel `GSIZE=64` 垫零头的**叠影**;"切头→topk 不同→拼错"的失败画面只对"切而不归约"的假想配置成立 |
| kPool mask | = indexer 每步算出的 **topk 稀疏选择掩码**(select_topk 构造,SFA 消费),非随机、非新 cache;"共享"靠复制计算零通信;glm5_2 无 kPool(每 token 一行 K,cmp_ratio=1) |
| mask 存什么坐标 | **0 基 kv 位置**(docstring "absolute kv-position",glm5_next.py:1090-1105);行宽稳态 = index_topk = 2048,开必选尾再加 kpool−1;三旁证:因果过滤直接比大小/池成员下标原样进 topk/mask 值从不查块表 |
| 兑换公式 | `INDEX_page = E×dcp + s`;开 dcp 后 latent 页号与 INDEX 页号是两套空间,同 id 不同义 |
| 两套 INDEX 输入 | 写侧全局槽直通(恒等式 槽≡自然行),读侧 expanded 表查页;读核访存粒度是页不是行 |
| 两个"5" | 槽 61 的块内偏移 `o=61%8=5`(E 基,**属主判定前的 rank 无关中间量**——rank0 得 -1 前 5 已算出)与读侧索引 `L=b×4+f=5`(b 基)的 **b=s 数值巧合**;位置 21(b=2)分道:L=9≠o=5 |
| b 与 E | b=序数(位置除出,读侧查表列号),E=门牌(块表值,写侧槽编码基),桥=block_table[b]=E;从槽只能除出 E 除不出 b(61//8=7 是 E 不是 b,结构性不对称) |
| 有效前置 | 分**三阶段**:构造期(选中天然在前,按分数降序——torch.topk 默认 sorted=True)/remap 期(owned 前置、peer 沉底)/宽行期(尾列并入游程);契约根源 = SFA 碰到行内第一个 -1 就停 |
| remap 怎么做 | 纯位置算术 `p→b=p//8→owner=(p%8)//4→本rank L=b×4+(p%4)/peer −1` + 打包前置 + SFA 查表落格(L//4 查块表、L%4 取行);读侧不经过全局槽;两 rank 得**互补的不同**本地行 |
| 载荷 (dcp,H_local,T,D+4) | 第一维=每目的 rank 一片;4 格不是裸位拆,是**手工浮点再编码**:格0=符号指数码(exp+128,兼失效旗)、格1-3=尾数拆段 16+8+8,每格 ≤255 的 bf16 精确小整数(lse=12→[131,192,0,0] 编解码双向可手算);LSE 是 per-(token,head) 标量 |
| AllToAll 为何分头 | gather 轴=**头维 dim=1**(token 不动);AllToAll 按头份额"借头还头"——query 沿头轴 gather 进来,部分结果沿头轴 scatter 还回去;交换发生在 **dcp 组**(TP 域内连续 kv_split 个 rank)内部,不是全局 TP×DP 大网;移动小账本(query/部分输出),不动大仓库(KV) |
| 图 padding | 同批 kernel(eager/graph 重放同一批,无模式切换),真执行、浪费∝padding 行占比(例 37.5%)、行代价被 kv_len=1 压最小;页 0..dcp-1=保留块 0 经 E×dcp+s 的合法展开(-1 会被 CANN indexer 核当页号→MTE 越界);attention_mask 恒全 1,清 -1 靠 span 剥离+F.pad+validity 过滤 |

## 概念地图:三层分离与坐标空间

三层不许混叫一个"cache":**模型语义**(MLA latent/DSA indexer K/kPool 尾)、**框架对象**(block table、slot_mapping、dcp 组、Attention backend)、**物理布局**(页/行/条带)。三层对象总表(谁产生→谁消费→活多久):MLA KV latent(前向投影→写 cache/SFA 读,常驻);indexer 的 K+门分数(每 token 一条 head_dim 维共享头 K,便宜到可复制);topk 稀疏索引(单步即焚);部分输出+LSE(SFA return_lse→融合核,单步即焚);block table(每序列一行,第 b 列=第 b 个逻辑块页号 E[b]);dcp 进程组(TP 域内连续 kv_split 个 rank 的 NCCL 组,py_causal_lm.cpp:305-330)。

### 四层坐标空间(全书手算统一参数:page=4, dcp=2, 逻辑块表 [5,7], 13 token)

| 空间 | 例子(第 14 个 token) | 编码公式 | 谁在用 |
|---|---|---|---|
| 位置 position | 13 | 序列内 0 基序号 | indexer 打分、**mask/topk 存的就是这个** |
| 全局槽 global slot | 61 | E×8+5 | 调度器 slot_mapping、INDEX 写直通(写侧量) |
| 本地写槽 | 29(rank1) | E×4+行 | latent 写入 localize_slots(61)(写侧量) |
| 读侧本地索引 L | 5(rank1;rank0 得 −1) | b×4+f | **remap 的产物**、SFA 稀疏索引列坐标 |

一行记住:**L 与本地写槽差一次块表查询**——L=5 → 页列 5//4=1 → block_table[1]=7 → 才到槽 29。「5 那个空间」和「29 那个空间」不是一层。位置 21 对照(表扩为 [5,7,9]):b=2、E=9、槽 77、L=9≠o=5、物理格 37——两个"5"正式分家(o 钉在分配视图上,b 变它不动;L 钉在序列视图上,b 变它跟着变)。

### 谁持有什么(TP=2, kv_split=2 的 decode 实例)

```
rank 0:query 头 [0..H/2) │ KV latent 条带0(每逻辑块第0页) │ INDEX 全量副本
rank 1:query 头 [H/2..H) │ KV latent 条带1(每逻辑块第1页) │ INDEX 全量副本
        └────── dcp 组 = TP 域内连续 kv_split 个 rank(py_causal_lm.cpp:305-330)──────┘
```

- **latent**:每逻辑块每 rank 只有一页,本地页号=块表值 E(npu_cp_plan.cpp:914-953);rank0 存页 5 行 0-3=位 0-3、页 7 行 0-3=位 8-11;rank1 存位 4-7、12-15——**条带按块内偏移区间划分,不按块划分**;
- **INDEX**:每逻辑块 dcp 页 `[E×dcp, (E+1)×dcp)`(cache_layout_builder.h:44-49),自然行恒等式 `row(s)=s`(INDEX 总行数恒=token 数——多配页是"门牌换算"非扩容:页数的 ×kv_split 被逻辑块数的 ÷kv_split 精确抵消);
- INDEX 复制强制:NPU 平台 `requires_dsa_indexer_cache_replication()`(platform.h:76-77);与 prefill PCP 复制写(replicate_pool_write_block_table,kv_shard_layout.py:182-219,docstring 公式就是 E×dcp+s 一字不差)同一套算术,PD 传输同一份资源描述两边通用。

## 机制主线:decode 一步(每个 DSA 层)

```
① mla_index_context:block_table 换成 expanded 表([5,7]→[10,11,14,15])
② indexer(select_qli):读 INDEX 副本打分 → topk(0 基位置,宽度 2048)
③ execute_mla:write_mla_paged_cache(local_slots)只写本 rank 条带
④ query all-gather(异步,头维 dim=1):[T,H_local,D]→[T,H,D] 双方齐
⑤ remap:topk 位置 → L(owned 前置/peer −1)
⑥ 等 query 到齐 → ⑦ SFA:本 rank 条带 + L 查表,sparse_mode=0,return_lse
⑧ pack [dcp,H_local,T,D+4] → AllToAll → fused LSE 合并 → [T,H_local,D]
```

**KV 从头到尾没跨 rank**;跨边界的只有 query(④)和部分输出+LSE(⑧),大小 ∝ token×头,与上下文长度无关。主调用链 sfa_dcp.py:556-573:`_remap_sparse_indices`(:559)→ `_finish_dcp_gather`(:560)→ `_npu_sparse_flash_attention(..., return_lse=True)`(:561-571)→ `softmax_lse = softmax_max + log(softmax_sum)`(:572)→ `_merge_dcp_outputs`(:573)。

**prefill 反向经济学**(sfa_dcp.py:537-554,不 remap):num_prefills>0 → KV 块 index_select 后沿块轴 all-gather 成完整副本、位置直接当稀疏索引(配展宽块表)、sparse_mode=3 全注意力、零输出交换——prefill 一次 T 个 token 摊薄 gather;decode 一次 1 个 token 摊不动,才广播 query。**remap 是 decode"不搬 KV、只换坐标"的专属方案**。

### indexer 层细节

**glm5_2 对照(拆"64 头+allreduce"叠影)**——三件事拆解表:

| # | 事 | 头数 | 切不切 | 求和发生在哪 | 通信 |
|---|---|---|---|---|---|
| ① | 主注意力(glm5_2) | n_heads=**64** | 切(TP 头切分) | o_proj 输出 all_reduce(glm5_2.py:557-561) | 有 |
| ② | indexer(glm5_2) | index_n_heads=**32**(:115) | 不切(三投影 copy_replicated,:896-902) | 无跨 rank 求和 | 零 |
| ③ | indexer 打分核内部 | 垫到 64 | 单 rank 内部 | 核内 Σ(垫零头,:43-47/:85) | 无 |

垫零头不动分数:score=Σ_{h=0}^{63} w_h·(q_h·k),h≥32 时 q_h=0 且 w_h=0 → 每项贡献 0×0=0。假想"切而不归约"失败画面:2 rank、4 候选 {A,B,C,D}、topk=2 → rank0 选 {A,C}、rank1 选 {B,C} → topk 清单分叉 → SFA 读不同位置子集 → 融合加权的对象对不上 → **无崩溃、无 NaN、悄悄算错**——但这只对假想配置成立,两代实际都走复制零通信。

- glm5_next:indexer 投影同样复制(glm5_next.py:3058-3065 注释 "each rank attends its head-subset to the SAME mask");topk 一致性=复制计算(相同输入×复制权重),零通信;INDEX head 维=1(kv_cache_shape.cpp:455-481 字面量 1)——被缓存的 K 每 token 只有一条共享 head_dim 向量,TP 无头可切只能复制。
- kPool mask:select_topk 产出三道关卡(:1245-1252 候选池可见性三重与+masked_fill -inf / :1256-1289 topk 后 masked_fill -1 / :1297 行级兜底);-1 五个来源:不足宽 F.pad(:1294-1295)/无效候选 masked_fill/padding 行整行(:1297)/图 token 维 padding(:1569-1571)/remap 后 peer 沉底。

### INDEX 几何与三张表(写读零重排的账)

**四条路总表**(位置 13,三张表三消费者+一个直通量):

| 表/量 | 输入 | 变换 | 结果 | 消费者 |
|---|---|---|---|---|
| local_slots | 全局槽 61 | 按条带判归属 | rank0=-1 / rank1=29 | latent 写(只写本 rank 条带) |
| 逻辑表 | 引擎块表 | int32 视图 | [5,7] | SFA 读 latent |
| expanded | 逻辑表 [5,7] | E×dcp+条带 | [10,11,14,15] | indexer 读写 INDEX |
| (直通)全局槽 | 61 | 无 | 61 | INDEX 写:直写第 61 行 |

**自然行恒等式代数证明**(零跳步):①调度器发槽 s_g=E×(page×dcp)+s×page+f=7×8+4+1=61(高位是 E,因为槽按块表值编码);②INDEX 编页 P=E×dcp+s=15,行=P×page+f=61;③两式恒等:E×dcp×page+s×page+f=E×(page×dcp)+s×page+f ∎。成立条件只有两个(槽按逻辑块尺寸编码+页组基=E×dcp),**证明从没用过"E=b"**——块表是 [5,7]、[0,1] 还是随机页号都成立。写侧落点 `_update_mla_index_cache`(npu_paged_attention.py:1391-1395):`cache_view.index_copy_(0, row_indices, values)` 槽当行号直写零换算(注释:Custom scatter 经 FP32 排序对奇数行宽如 257 丢整数精度,故用 index_copy_+int64)。

**localize 一函数两调用点两输入基**(kv_shard_layout.py:79-100):写侧喂槽(E 基)输入 61→输出 29(物理页空间);读侧喂位置(b 基)输入 13→输出 5(块表下标空间)。函数只做除法取余不查表;输入高位是什么基,输出就是什么基。

**materialized 表——第三套编号基**(sharded 读侧,npu_paged_attention.py:730-739):`(b×ncols+c)×dcp+s`(表内位置基)vs expanded 的 E 值基;恒等表下数值重合是一切"感觉是同一个 id"的根源。混用失败画面:表 [5,7] 里 E=7 那列——位置基算页 [2,3]、E 基算 [14,15],数据躺在页 14 而读核去页 2——**读到别的块的陈旧数据,无报错,最阴的一类 bug**。

**换错表的失败画面 ×3**:①latent 写拿全局槽 61 当本地槽→写进页 15 行 1(别家逻辑块)→污染+静默错算,页 15 未分配则越界写;②INDEX 写用 local_slots→rank0 跳过、rank1 写第 29 行,读侧永远读第 61 行→旧值/垃圾→topk 选错;③SFA 读拿 expanded 查→block_table[1]=11→读 INDEX 页号空间的页 11→张冠李戴。

### remap 链(位置→L→落格)

**公式卡片**(最该记住的五行):

```
p(位置)
 ├─ b     = p // 8        逻辑块号
 ├─ off   = p % 8         块内偏移
 ├─ owner = off // 4      条带号(0→rank0,1→rank1)
 ├─ f     = off % 4       条带内行
 └─ L     = b × 4 + f     (owner 是本 rank 时;peer 得 -1)
```

消费者:`_remap_sparse_indices`(sfa_dcp.py:329-383,decode 分支 :559;prefill 不 remap)。**不换的失败画面**(rank1、位置 5 与 13 直读):位置 5→页列 1 行 1→槽 29=**位置 13 的格子**(张冠李戴);位置 13→页列 3→块表只有 2 有效列→越界读陈旧垃圾页。**两处都不崩溃**——SFA 拿到看起来合法的数字就去读,静默错算,比崩溃更危险。

**手算三连**(统一数字):
- toy 例 {0,2,3}:全落条带 0 → rank0 行 [0,2,3](L 恰等于位置,第 0 块条带 0 的巧合)、rank1 行 [-1,-1,-1](退化互补,合并后输出全来自 rank0);扩展变体 {0,2,3,5}:位置 5→owner=1、f=1→rank1 得 L=1——rank0 [0,2,3] / rank1 [1],**并集=原选中集,这才是 dcp=2 的常态(真拆分)**;
- 位置 13 七步:b=1→off=5→s=1(rank1)→f=1→L=5→SFA 页列 5//4=1 查 block_table[1]=7→物理格 7×4+1=29——与写侧 localize_slots(61)=29 **写读落点闭环,同一格**(但 remap 路径不经过 61);
- 位置 21(b=2):L=2×4+1=9→页列 2→block_table[2]=9→格 37;o=21%8=5 不变而 L 从 5 变 9——L 用 b 编码,不是偏移。

**打包 `pack_owned_slots`**(kv_shard_layout.py:102-119):`pack_keys = 列号 + (localized<0)×width`——有效列键∈[0,width)、无效列键∈[width,2×width),**所有键两两不同→排序结果唯一(不依赖 stable)、有效列相对序=原列序**。手算(rank1、width=14):localize [0..13]→[-1,-1,-1,-1,0,1,2,3,-1,-1,-1,-1,4,5];键 [14,15,16,17,4,5,6,7,22,23,24,25,12,13];排序+gather→**[0..5,-1×8]**(6 个有效);rank0 对称得 **[0..7,-1×6]**(8 个有效)。两份合起来恰好是同一选中集合的两个互补视图,**全程零跨 rank 顺序协调**——顺序语义上不重要(注意力对集合求加权求和),打包才是硬需求(SFA 首停契约),"保持原序"是最省事且确定的选择。快路 TileLang 融合核(sfa_dcp_remap.py:167-179)同键分层思想方向相反(降序+负无穷),语义等价单测钉死。

**宽行 `_fill_wide_remap`**(sfa_dcp.py:88-110,必选尾):`owned_prefix=(packed_prefix>=0).sum(-1)`(数列=scatter 起点,与池的"一对多"数行内成员是两个维度不矛盾)→`tail_offsets=owned_prefix+arange(tail)`→fill_(-1) 铺底→前缀拷入→尾列 localize+打包后 scatter 到前缀正后方。微例(rate=4, topk=8, 尾宽 3):前缀 [0,5,-1,...] + 尾 [12,13,-1]→pack 尾得 [4,5,-1]→scatter→**out=[0,5,4,5,-1,...]**。**反例**:不做 fill,尾列留在第 8-10 列而前缀第 2 列已有 -1→SFA 第 2 列停→必选的最新 token 每步都丢(docstring:"drops the newest force-selected tokens from every decode step")。必选尾=三配置合取(index_kpool>1 ∧ compress ∧ always_select_tail,glm5_next.py:895-899),行宽从 topk 变 topk+rate-1。

**融合核约束**(sfa_dcp.py:66-85):topk==2048 且行宽==2048 且 0<tokens≤256 且 page/dcp 均 2 的幂且 int32 连续 NPU;不满足走 torch 参考(sfa_dcp_ref.py)——spec-verify 宽批(N×(spec+1)>256)会静默回退,正确但慢。

### 交换与融合(载荷与拓扑)

**路由与四维**:dim0=dcp(all_to_all_single 沿 dim0 等分,每目的 rank 一片)、dim1=H_local(头份额)、dim2=T(token 轴复制)、dim3=D+4。`rank_idx = head_idx // H_local`(头 h 归第 h//H_local 个 rank 的片,dcp_packed_a2a.py:62-69);H_full=4、H_local=2 时 h0,h1→片 0、h2,h3→片 1;第 r 片发 rank r。**AllToAll 后每 rank 收齐"自己那 H_local 个头 × 所有条带"的部分结果。借头还头**:gather 把头借出去(全头 query 打自己条带),AllToAll 把头还回来(别人头在我条带的份发还头主人);token 轴全程不动。

**4 格手工浮点再编码**(dcp_packed_a2a.py:82-100,不是裸位拆):lse_exponent=floor(log2|lse|) clamp [−126,127];significand=|lse|×2^(23−exp) 拆 hi=floor(sig/65536)、mid=floor(rem/256)、lo=rem−mid×256;格0=sign×(exp+128)(**非有限恒 0,兼作失效旗**);每格 ≤255 小整数——bf16 的 8 位有效尾数精确可存,往返零误差。手算 lse=12.0:exp=3、sig=12,582,912 → **[131,192,0,0]**;解码 192×65536×2^(3−23)=12.0 ✓。为什么 4 格不是 2 格:bf16 一格只有 ~8bit 精确载荷,24bit 尾数要 3 格+指数 1 格;裸拆出的 16bit 位段超 bf16 尾数就失真(LSE 指数敏感)。复盘金句:"约束不是 bf16 装不下 fp32,而是 bf16 装不下 fp32 的任意位段——改成装 bf16 装得下的等价信息,问题就解了。"

**valid_lse 三道防御**(:196-214):`packed_valid=exponent_code≠0` & 非 NaN & 非 ±inf → weight=0、partial=0、分母保底 1——全 -1 行(本 rank 无 owned)不污染融合;所有份都空转→输出全 0 不产 NaN。手算(份 A lse=12 valid、份 B invalid):weight_A=1、weight_B=0→份 B 蒸发。

**LSE 合并数学**(每 (token,头)):max=max(lse_r);w_r=exp(lse_r−max);输出=Σw_r·O_r/Σw_r。统一数字手算:lse_0=12、lse_1=13→max=13→w_0=e⁻¹≈0.368、w_1=1→rank1 占约 **73%**。三行代数精确性(实数域):w_r=exp(m_r−M)·S_r;w_r·O_r=Σ_{i∈r} exp(x_i−M)·v_i;Σ_r w_r·O_r=Σ_{全部选中 i} exp(x_i−M)·v_i——分子分母同缩一个基准,merged=全局 softmax ∎(M 取哪个基准都行)。**w_r 不是"rank 的重要性",是"这个 rank 拥有的那些位置在全局 softmax 质量里的份额"**。三种错误融合:直接平均(偏差随 lse 差指数放大)/只融 max+sum(手里没有任何一份全局正确的向量)/按位置对齐拼接(位置列表从未交换,没有可对齐的轴)。

**拓扑走查**((2,2,2,6) 载荷的一生,rank0 视角):SFA 输出 [2,4,2]+lse [2,4]→pack 按 rank_idx=h//2 路由→send=[2片,2头,2token,6]→all_to_all_single 片 0 留自己、片 1 发 rank1,同时收 rank1 的片 0→recv=头 h0,h1 的 {条带 0 份,条带 1 份}→fused combine→merged=[2,2,2]——只有 rank0 自己的头。dcp 组构成(py_causal_lm.cpp:307-328):`cp_size_==1 && kv_split>1 → CHECK_EQ(tp_size_%kv_split,0); init_process_group("dcp",...)`——头轴与条带轴是同一组 rank 上的两条独立坐标。

### 图模式 padding(三件套)

eager 与 graph 跑**同一批 kernel**(lightning_indexer.cpp:93-108 纯转发 CANN,无模式切换);唯一区别=图模式 batch 被补齐、存在 padding 行。**-1/0 分界在张量种类×消费者契约**:

| 张量种类 | 例子 | 消费者 | 对 -1 的行为 | 结论 |
|---|---|---|---|---|
| slot 列表 | slot_mapping、local_slots、remap 后稀疏索引 | 写 KV 的核 / SFA | 跳过 / 首停 | -1 安全 |
| 页表 | expanded、INDEX 读表 | CANN indexer 核 | 当无符号页号寻址 | -1 致命,必须全合法 |

**三件套**=块表 0 / 槽 -1 / 长 1(decode_acl_graph.py:1409-1480 捕获、1727-1741 回填;注释 "Padded lanes must remain valid inputs for sparse MLA tiling...one KV token is safe")——块 0 喂 indexer 核(合法页号)、槽 -1 喂写核(跳过)、长 1 喂 SFA(tiling 有效)和 indexer(kv_len≥1 不退化),一石三鸟。

**手算四算**(dcp=2、图捕获 batch=4、真实 2、page=4):①expanded:padding 行块表 [0,0]→0×2+{0,1}=[0,1] 全合法;②写路径:槽 -1→localize 判无效原样 -1→写核跳过→块 0 不被虚构数据污染;③local_seq_lens(kv_len=1):rank0=1、rank1=0(rank0 的 SFA 看到 1 行 tiling 有效);④padding 行 topk:被清成全 -1(见勘误②的三机制,非 attention_mask)→remap 保持全 -1→SFA 首停第 0 列空转→输出丢弃,陪跑结束无副作用。**反事实**(expanded 的 0 改 -1):indexer 核拿 -1 当无符号页号=int32 语义 0xFFFFFFFF→寻址偏移≈4294967295×页大小→**MTE 越界(崩溃),或静默错 top-k(更糟)**(glm5_next.py:1059-1063 注释 "silently returns wrong top-k sets";真实路径配套 clamp(min=0) 两处)。

**深水区**:块 0 是永久保留块,真实序列也会用;padding 行去**读**它无害(结果被丢)、**写**被槽 -1 挡住——一读一写两头堵死;若块 0 被分配器挪作他用,padding 行会混入真实 batch 的结果(错算而非崩溃,更难查)。浪费量级:padded=8 真实=5→37.5%(结构推导,无 benchmark);kv_len=1 是"合法"(长度 0 使稀疏 MLA tiling 非法)与"最省"的唯一交点。

## 工程决策与验证

- **为什么 Q-gather 不 KV-gather**:每步通信 ∝ token×头(几十 KB)而非上下文(GB 级);方案对比:A all-gather KV(带宽打爆,稀疏压缩红利被复制放大 dcp 倍)/B 分数 all_reduce 归约再 topk(每层每步对整张分数图同步)/C 复制各自重算(显存换通信,INDEX 每 token 仅 128 维 vs latent 512+ 维)。选 C,同选 AllToAll 头轴散射+单载荷(一次通信搬完、零冗余;复盘:decode 小包多次场景,通信次数是延迟主导项)。
- **为什么 INDEX 复制**:indexer 打分要全量历史;复制换读零通信,且与 prefill PCP 复制写同构,PD 传输两边通用同一资源描述(cache_layout_builder.cpp:118-149 describe_replicated_index_pages:一段连续 dcp 页无空洞无跳排)。sharded 写模式的坑:每 rank 只写自家条带进页组第 0 页,组内其余页陈旧,RANK_LOCAL 1:1 搬运会把陈旧页当有效页搬过去——replicated 模式任何单个 writer 都能供出全部有效页。
- **M9 开关**(讨论催生):`XLLM_CP_INDEX_WRITE_MODE`(replicated 默认/sharded 支线)——用户判"无显存收益+引入 allgather 是纯纯负向优化"方向纠偏:latent 才是省显存的地方,INDEX 本来就每 rank 全量复制;sharded 只对 owner-sharded 读路径正确,是半成品默认关。
- **拓扑锚定 cp=1**:master 门控对 cp>1 要求"kv_split>1 必须 PD+PREFILL 角色",cp=1 提前返回不做此检查(设计而非漏洞)——正确性由后端选择保证(executor.py:126-176:NPU+cp1+dcp 组>1+MLA → SfaDcp),准入配对测试钉死准入-选择配对。
- **PD 宽度协商**(kv_cache_transfer.cpp:97-145):等宽→RANK_LOCAL(1:1 块映射;src kv_split>1 且 sharded 写模式→reject,stale INDEX 页风险);D 侧 kv1→STRIDED;其余 UNRECONCILABLE。一致性实验 链接日志 `Successfully linked instance, prefill_kv_split_size: 2`。
- **验证神谕**:等宽 D(kv2 SfaDcp) vs 交错 D(kv1) 对照——greedy 逐位一致、48/48 logprobs、max|dlogprob|=0.0。
- 遗留:overlap×SfaDcp 组合当时未合测;spec-verify 宽批掉速回退。

## 讨论独有沉淀(书未覆盖,来自会话考古)

- **pr370 同构 vs CP owner-shard 的本质冲突**:等宽 D(kv2) 走 SfaDcp 需要 INDEX 全副本由双 owner 各供一页,而 reshard planner 的 per-resource overlap 每资源只能路由一页——"数学上表达不了";候选修法"+rank"写行是 vacuous fix(owner≥1 会推自己从未写过的行,测试全绿但症状依旧)。
- **M7 传输损坏根因链**(故障分析范例):manifest 与字节级 plan 均正确,损坏在 block-id 映射层——width-blind 的 rank_local_mapping 不看 D 声明的 kv_split_size,异构 2→1 对错走 1:1 映射,stripe-1 owner 用陈旧内存覆盖有效 prompt KV(逐层 last-writer-wins),token 0 精确(token 1+ 全错)。
- **prefix admission 的生产意义**(三轮施压后的认知修正):prefix 门是 **host offload 的前置条件**——validate_host_cache_options 要求 prefix 开启→不放行 prefix 则 `--host_blocks_factor>1` 设不了→host offload(Mooncake store 分层)对 glm5_next CP 永远关死;"不是省 prefill 算力那点收益,是打开 host 分层"。dengyingxu1 主线(#462)=prefix 放行→host offload→线性检查点进 Mooncake 分层;本会话 device 层 PD 复用是支线。glm5.3-flash 需重做 prefix 的根因:混合线性架构的前缀复用=KV 块+LINEAR 检查点双轨,而 LINEAR 检查点有发布时机断点(发布循环跳过 LINEAR、cache_read_source 依赖"下一次 allocate_prefill"、PD 交接后不存在"下一次"——同时解释 M4 零复用)。
- **PD 三路线裁决**:A(D 侧 bookkeeping)/B(P 检查点存活)/C(trim 放宽),用户拍板 B;四路上游横比(vLLM delay_free→evictable 池/SGLang release(checkpoint=True) 入树)证实 P 侧 cached-but-evictable 是生态共识。
- **D 侧传输跳过"半个机制"**:remote_shared_num→advance_transfer_block_idx 链路存在(D try_allocate→probe 前缀→命中变 shared_blocks_num→P 从 cursor 起推)但端到端未测;P 自身命中只进记账不缩传输;正确第一步是先测量不是写代码。
- **scheduler overlap 真相**:调度器是 C++ 不是 Python;greedy argmax over CP-replicated logits 跨 rank 字节一致;non-greedy 每 rank 本地采样、RNG lockstep 只在 init 种子,分歧真实存在。

## 关键源码锚点

| 机制 | 位置 |
|---|---|
| 后端类/prepare/execute_mla | sfa_dcp_backend.py:15/:134-202/:223-276 |
| 执行者(remap/merge/gather) | sfa_dcp.py:29/:66-110/:315-383/:385-471/:526-573 |
| 几何算术(localize/pack/expand) | kv_shard_layout.py:69-138 |
| select_topk 坐标/行宽/F.pad | glm5_next.py:1090-1105/:1261/:1293-1296/:1348-1357 |
| indexer 复制注释/打分循环 | glm5_next.py:3058-3065/:1221-1236;glm5_2.py:44/:896-902/:680-692/:557-561 |
| 4 格编码/valid_lse/shape/路由 | dcp_packed_a2a.py:62-69/:82-116/:142-214/:354-364 |
| dcp 组创建 | py_causal_lm.cpp:305-330 |
| 页表不变量(本地页=块号)/C++ 孪生 | npu_cp_plan.cpp:914-953 |
| INDEX 写直通/materialized 表 | npu_paged_attention.py:1230-1237/:1391-1395/:730-751 |
| 图 padding 三件套/真执行 | decode_acl_graph.py:1406-1480/:1727-1741/:2065-2066 |
| 后端选择 | model_executor/executor.py:126-176 |
| 宽度协商 | framework/kv_cache_transfer/kv_cache_transfer.cpp:97-145 |
| INDEX 页数/资源描述 | cache_layout_builder.h:35-49;cache_layout_builder.cpp:118-149;kv_cache_shape.cpp:455-481 |

⚠️ **证据边界**(四条,全部来自书内显式标注):①SFA"首个 -1 停"与写核"-1 跳过"在 CANN 侧闭源(证据=四处 docstring+_fill_wide_remap 防御代码的存在动机);②LightningIndexer/pool_key_indexer 把 -1 当巨大页号→MTE 越界闭源(证据=两注释+两 clamp+测试断言);③Q1"逐位一致"由相同输入+复制权重+确定性 kernel 推出,无显式跨 rank 比对测试(端到端佐证 一致性实验 max|dlogprob|=0.0);④SFA 对 attention_out 的归一约定无法直接读(由融合代数+端到端 0.0 偏差反推)。另有:融合核 padding 行核内处理[待验证]、padding 浪费无 benchmark、个别行号 ±1 漂移(以 src/PROVENANCE.md hash 锚定为准)。

## 来源与处置(重结算版,如实记录)

**输入与通读方式**:
- 一轮书 5q(2707 行)/二轮书 9q(2746 行)/三轮书 round3(878 行):**经子代理逐本全文通读**(5q 分 8 段 1-2707、9q 分 7 段 1-2746、round3 分 3 段 1-878,无抽样无跳读),各产出高保真摘要(312/276/281 行),本页由摘要+现有骨架重写;
- 会话讨论:两个会话(pipeline+结算会话 50 条/首轮开发会话 416 条)用户消息全集 3341 行经子代理全文考古,纠错回合 30 条、讨论独有点 12 项;
- 输入总计:书 6331 行+讨论 3341 行 → 本页(约 240 行)。

**章节覆盖率表**(硬门:每章映射到本页小节,无遗漏):

| 书 | 章节 | 承接小节 | 裁决 |
|---|---|---|---|
| 5q | 卷首导航/概念地图 | §概念地图/§机制主线 | 进正文 |
| 5q | 第 1 章 indexer 复制地图 | §indexer 层细节 | 进正文 |
| 5q | 第 2 章 INDEX 几何与三张表 | §INDEX 几何与三张表 | 进正文 |
| 5q | 第 3 章 打包交换融合 | §remap 链(打包)/§交换与融合 | 进正文 |
| 5q | 第 4 章 图模式 padding | §图模式 padding | 进正文 |
| 9q | 第 1 章 indexer 头与掩码 | §indexer 层细节(glm5_2 对照表) | 进正文 |
| 9q | 第 2 章 两套编号空间 | §概念地图(四空间)/§INDEX 几何(两个"5"/materialized) | 进正文 |
| 9q | 第 3 章 尾列/LSE/AllToAll | §remap 链(宽行)/§交换与融合 | 进正文 |
| 9q | 第 4 章 图 padding 真执行 | §图模式 padding | 进正文 |
| round3 | 第 1 章 (a)(b)(c)(d) | §结论先行(3 行)/§remap 链(公式卡片/手算三连) | 进正文 |
| 三书卷末 | 全局自检/勘误裁决 | §勘误链/§证据边界 | 进正文 |
| 讨论 | 纠错回合/独有点 | §工程决策与验证/§讨论独有沉淀 | 进正文(精选) |

略过项:三书的 Step-by-step 教学叙事(铺垫/类比/自检清单)未逐条搬入——机制事实、手算数字、失败画面、源码锚点、勘误裁决全部保留;完整版在原书。

**处置**:三轮书工作区与历史遗留旧版书已于 2026-10-13 全部清理。
- 关联主题:prefill 侧 PCP 复制写、KV 分条 rank 布局、PD 传输、宽度准入(同系列深潜笔记)。
