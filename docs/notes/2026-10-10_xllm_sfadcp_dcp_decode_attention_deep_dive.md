---
title: SfaDcp:GLM-5.3-Flash decode KV 分条注意力三轮教学结算
date: 2026-10-10
updated: 2026-10-10
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

# SfaDcp:GLM-5.3-Flash decode KV 分条注意力三轮教学结算

> 一句话:**SfaDcpAttentionBackend 让 dcp 组内每个 rank 只存 KV latent 的 1/kv_split 条带,靠"广播小小的 query、寄回部分输出、LSE 精确合并"买回全量注意力——KV 从不跨 rank 移动,INDEX 全量复制让"去哪找"零通信**;本页吸收三轮教学书(共 6331 行)与会话讨论,独立可读。

## 场景:三轮教学与勘误链

主题源自一次实验:给 glm5_next 启用 D(kv_split) decode,真机多卡验证 `max|dlogprob|=0.0`。围绕它做了三轮教学:

1. **一轮(5 问,2707 行)**:query 全量/INDEX 自然行/打包与 LSE 融合/三张表/图 padding;
2. **二轮(9 问,2746 行)**:glm5_2 indexer 对照、kPool mask、E/b 兑换、两个"5"、必选尾、载荷 D+4、AllToAll 头拓扑、padding 真执行;
3. **三轮(1 问 4 子,878 行)**:mask 的坐标空间与 remap 链。

**勘误链(读旧材料以本页为准)**:一轮书 3.4 把位置 13 当槽 13(三空间混淆,二轮勘误);二轮书"attention_mask 对 padding 行为 False"(实际恒全 1,三轮勘误);二轮书手算例 C 的 `[0,2,3,-1]`(实际 `[0,2,3]` 宽 3 无 -1,三轮勘误);二轮 pack"位置→全局槽→owner"表述(读侧 remap 从不经过全局槽,三轮澄清)。

## 结论先行(十五问判决索引)

| 问 | 判决 |
|---|---|
| indexer query 切 TP 吗 | **不切**。glm5_2/glm5_next 的 indexer 三投影(wq_b/wk/weights_proj)TP 全复制,打分+topk 单核零归约;"glm5.2 64 头+allreduce sum"是**主注意力**(64 头、TP 切、o_proj all_reduce)与 kernel `GSIZE=64` 垫零头的**叠影**;"切头→topk 不同→拼错"的失败画面只对"切而不归约"的假想配置成立 |
| kPool mask | = indexer 每步算出的 **topk 稀疏选择掩码**(select_topk 构造,SFA 消费),非随机、非新 cache;"共享"靠复制计算零通信;glm5_2 无 kPool |
| mask 存什么坐标 | **0 基 kv 位置**(docstring "absolute kv-position",glm5_next.py:1102-1104);行宽稳态 = index_topk = 2048,开必选尾再加 kpool−1 |
| 兑换公式 | `INDEX_page = E×dcp + s`;开 dcp 后 latent 页号与 INDEX 页号是两套空间,同 id 不同义 |
| 两套 INDEX 输入 | 写侧全局槽直通(恒等式 槽≡自然行),读侧 expanded 表查页 |
| 两个"5" | 槽 61 的块内偏移 `o=61%8=5`(E 基)与读侧索引 `L=b×4+f=5`(b 基)的 **b=s 数值巧合**;位置 21(b=2)分道:L=9≠o=5 |
| b 与 E | b=序数(位置除出,读侧查表列号),E=门牌(块表值,写侧槽编码基),桥=block_table[b]=E;从槽只能除出 E 除不出 b |
| 有效前置 | 分**三阶段**:构造期(选中天然在前,按分数降序——torch.topk 默认 sorted=True)/remap 期(owned 前置、peer 沉底)/宽行期(尾列并入游程);契约根源 = SFA 碰到行内第一个 -1 就停 |
| remap 怎么做 | 纯位置算术 `p→b=p//8→owner=(p%8)//4→本rank L=b×4+(p%4)/peer −1` + 打包前置 + SFA 查表落格;读侧不经过全局槽;两 rank 得**互补的不同**本地行 |
| 载荷 (dcp,H_local,T,D+4) | 第一维=每目的 rank 一片;4 格不是裸位拆,是**手工浮点再编码**:格0=符号指数码(exp+128,兼失效旗)、格1-3=尾数拆段,每格 ≤255 的 bf16 精确小整数(lse=12→[131,192,0/0]);LSE 是 per-(token,head) 标量 |
| AllToAll 为何分头 | gather 轴=**头维**(token 不动);AllToAll 按头份额"借头还头"——移动小账本(query/部分输出),不动大仓库(KV) |
| 图 padding | 真执行,浪费∝padding 行占比、行代价被 kv_len=1 压最小;页 0..dcp-1=保留块 0 经 E×dcp+s 的合法展开(-1 会被 CANN indexer 核当页号→MTE 越界);attention_mask 恒全 1,清 -1 靠 span 剥离+F.pad+validity 过滤 |

## 概念地图:三层分离与坐标空间

三层不许混叫一个"cache":**模型语义**(MLA latent/DSA indexer K/kPool 尾)、**框架对象**(block table、slot_mapping、dcp 组、Attention backend)、**物理布局**(页/行/条带)。

### 四层坐标空间(全书手算统一参数:page=4, dcp=2, 逻辑块表 [5,7], 13 token)

| 空间 | 例子(第 14 个 token) | 编码公式 | 谁在用 |
|---|---|---|---|
| 位置 position | 13 | 序列内 0 基序号 | indexer 打分、**mask/topk 存的就是这个** |
| 全局槽 global slot | 61 | E×8+5 | 调度器 slot_mapping、INDEX 写直通(写侧量) |
| 本地写槽 | 29(rank1) | E×4+行 | latent 写入 localize_slots(61)(写侧量) |
| 读侧本地索引 L | 5(rank1;rank0 得 −1) | b×4+f | **remap 的产物**、SFA 稀疏索引列坐标 |

一行记住:**L 与本地写槽差一次块表查询**——L=5 → 页列 5//4=1 → block_table[1]=7 → 才到槽 29。位置 21 对照(表扩为 [5,7,9]):b=2、E=9、槽 77、L=9≠o=5,两个"5"正式分家。

### 谁持有什么(TP=2, kv_split=2 的 decode 实例)

```
rank 0:query 头 [0..H/2) │ KV latent 条带0(每逻辑块第0页) │ INDEX 全量副本
rank 1:query 头 [H/2..H) │ KV latent 条带1(每逻辑块第1页) │ INDEX 全量副本
        └────── dcp 组 = TP 域内连续 kv_split 个 rank(py_causal_lm.cpp:305-330)──────┘
```

- **latent**:每逻辑块每 rank 只有一页,本地页号=块表值 E(npu_cp_plan.cpp:914-953);
- **INDEX**:每逻辑块 dcp 页 `[E×dcp, (E+1)×dcp)`(cache_layout_builder.h:44-49),自然行恒等式 `row(s)=s`(INDEX 总行数恒=token 数——多配页是"门牌换算"非扩容);
- INDEX 复制强制:NPU 平台 `requires_dsa_indexer_cache_replication()`(platform.h);与 prefill PCP 复制写(replicate_pool_write_block_table)同一套 E×dcp+stripe 算术,PD 传输同一份资源描述两边通用。

## 机制主线:decode 一步(每个 DSA 层)

```
① mla_index_context:block_table 换成 expanded 表([5,7]→[10,11,14,15])
② indexer(select_qli):读 INDEX 副本打分 → topk(0 基位置,宽度 2048)
③ execute_mla:write_mla_paged_cache(local_slots)只写本 rank 条带
④ query all-gather(异步,头维):[T,H_local,D]→[T,H,D] 双方齐
⑤ remap:topk 位置 → L(owned 前置/peer −1)
⑥ 等 query 到齐 → ⑦ SFA:本 rank 条带 + L 查表,sparse_mode=0,return_lse
⑧ pack [dcp,H_local,T,D+4] → AllToAll → fused LSE 合并 → [T,H_local,D]
```

**KV 从头到尾没跨 rank**;跨边界的只有 query(④)和部分输出+LSE(⑧),大小 ∝ token×头,与上下文长度无关。同实例跑 prefill 时走反向经济学:num_prefills>0 → KV 块 index_select 后沿块轴 all-gather、sparse_mode=3 全注意力、零输出交换(prefill 一次 T 个 token 摊薄 gather;decode 一次 1 个 token 摊不动,才广播 query)。

### indexer 层细节

- glm5_2:indexer `index_n_heads=32`(glm5_2.py:44/115),三投影 TP 全复制(:896-902),aclnnQuantLightningIndexer 单核打分+topk 零归约(:680-692);kernel 要求 `num_heads_q/num_heads_k==64`,垫零头补到 64、零权重不改分数(:43-47/85)。**64 头+all_reduce 是主注意力**(:557-561)的事。
- glm5_next:indexer 投影同样复制(glm5_next.py:3058-3065 注释 "each rank attends its head-subset to the SAME mask");topk 一致性=复制计算(相同输入×复制权重),零通信。
- kPool mask:select_topk 产出(:1245-1297),构造期三重与+两道 masked_fill+行级兜底;-1 五个来源:不足宽 F.pad(:1294-1295)/无效候选 masked_fill/padding 行整行/图 token 维 padding/remap 后 peer 沉底。

### remap 链(位置→L→落格)

- 消费者:`_remap_sparse_indices`(sfa_dcp.py:329-383,decode 分支 :559;prefill 不 remap);不换的失败画面:rank1 上位置 5 直读本地槽 5 → 页列 1 行 1 = 槽 29 = **位置 13 的格子**(张冠李戴),位置 13 越界读陈旧垃圾——静默错算。
- 打包:`pack_owned_slots`(kv_shard_layout.py:102-119),排序键=列号+(无效?width),键全唯一→恒保原序、零跨 rank 协调;宽行(kPool 尾)走 `_fill_wide_remap`(sfa_dcp.py:88-110):owned_prefix=前缀有效列数(scatter 起点),尾列 scatter 到 `owned_prefix+arange`,续进有效游程——排在 -1 后会被 SFA 首停吞掉。
- 融合核约束(sfa_dcp.py:66-85):topk==2048 且行宽==2048 且 0<tokens≤256 且 page/dcp 均 2 的幂且 int32 连续 NPU;不满足走 torch 参考(sfa_dcp_ref.py)。spec-verify 宽批(N×(spec+1)>256)会静默回退,正确但慢。
- 两 rank remap 同一 mask → 互补的不同本地行(toy {0,2,3}:rank0 得 [0,2,3]、rank1 全 -1;扩展 {0,2,3,5}:rank0 [0,2,3] vs rank1 [1])。全 -1 行由 valid_lse 防御(见下)。

### 交换与融合(载荷与拓扑)

- `packed_send_shape=(dcp, H_local, T, D+4)`(dcp_packed_a2a.py:354-364):dim0=每目的 rank 一片(all_to_all_single 沿 dim0 等分)、dim1=H_local(头份额)、dim2=T(token 轴复制)、dim3=D+4。
- 4 格编码(:82-116):格0=sign×(exp+128)——**exp+128==0 兼作失效旗**;格1-3=尾数拆段;每格 ≤255 小整数,bf16 精确可存;lse=12→[131,192,0,0] 编解码双向可手算。
- **valid_lse 防御**(:196-214):`valid_lse = 旗立 & 非 NaN & 非 ±inf` → weight=0、partial=0、分母保底 1——全 -1 行(本 rank 无 owned)不污染融合。
- 路由:`rank_idx = head_idx // H_local`(头 h 归第 h//H_local 个 rank 的片);AllToAll 后每 rank 收齐"自己那 H_local 个头 × 所有条带"的部分结果。**借头还头**:gather 把头借出去,AllToAll 把头还回来;token 轴全程不动。
- LSE 合并数学:两半部分 softmax 各带 (O_r, lse_r),全局 m=max、权重 exp(lse_r−m)、加权和÷权重和——实数域精确,故 一致性实验 能测出精确零。

### 图模式 padding(三件套)

- 构造(decode_acl_graph.py:1406-1480 捕获/1727-1741 回填):**块 0 / 槽 -1 / 长 1**;DSA 管理表同款(npu_paged_attention.py:1857-1860,保留块 0 zero_),host 侧 KV 长度补 1(:1940-1947)。
- padding 行**真执行**(图形状固定),浪费∝行占比、行代价被 kv_len=1 压最小;输出被丢弃。
- 页 0..dcp-1 = 保留块 0 经 E×dcp+s 的合法展开(sfa_dcp_backend.py:157-161 注释):**不能写 -1**——三类消费者契约不一致:写核见 -1 跳过(安全)、SFA 首个 -1 停(安全)、**CANN indexer 核把 -1 当页号 → MTE 越界(致命)**。
- ⚠️ 勘误:decode(图/eager 同)attention_mask **恒全 1**(glm5_next.py:2774-2775 缺省 ones;图模式只传两参 decode_acl_graph.py:2065-2066);padding 行 topk 清 -1 实际靠压缩尾路径 span 剥离+F.pad(-1)(glm5_next.py:1348-1357/1293-1296)与 causal/validity 过滤(:1246-1252/1276-1277)。

## 工程决策与验证

- **为什么 Q-gather 不 KV-gather**:每步通信 ∝ token×头(几十 KB)而非上下文(GB 级);方案对比:A all-gather KV(带宽打爆)/C 选中()/D ring 式(复杂度数量级)。
- **为什么 INDEX 复制**:indexer 打分要全量历史;复制换读零通信,且与 prefill PCP 复制写同构,PD 传输两边通用同一资源描述。
- **拓扑锚定 cp=1**:master 门控对 cp>1 要求"kv_split>1 必须 PD+PREFILL 角色",cp=1 提前返回不做此检查(设计而非漏洞)——正确性由后端选择保证(executor.py:126-176:NPU+cp1+dcp 组>1+MLA → SfaDcp),准入配对测试钉死准入-选择配对。
- **PD 宽度协商**(kv_cache_transfer.cpp:97-145):等宽→RANK_LOCAL(1:1 块映射;src kv_split>1 且 sharded 写模式→reject,stale INDEX 页风险);D 侧 kv1→STRIDED;其余 UNRECONCILABLE。一致性实验 链接日志 `Successfully linked instance, prefill_kv_split_size: 2`。
- **验证神谕**:等宽 D(kv2 SfaDcp) vs 交错 D(kv1) 对照——greedy 逐位一致、48/48 logprobs、max|dlogprob|=0.0。
- 遗留:overlap×SfaDcp 组合当时未合测;spec-verify 宽批掉速回退。

## 关键源码锚点

| 机制 | 位置 |
|---|---|
| 后端类/prepare/execute_mla | sfa_dcp_backend.py:15/:134-202/:223-276 |
| 执行者(remap/merge/gather) | sfa_dcp.py:29/:66-110/:315-383/:385-471/:526-573 |
| 几何算术(localize/pack/expand) | kv_shard_layout.py:69-138 |
| select_topk 坐标/行宽/F.pad | glm5_next.py:1102-1104/:1261/:1293-1296/:1348-1357 |
| indexer 复制注释 | glm5_next.py:3058-3065;glm5_2.py:44/:896-902 |
| 4 格编码/valid_lse/shape | dcp_packed_a2a.py:82-116/:196-214/:354-364 |
| dcp 组创建 | py_causal_lm.cpp:305-330 |
| 页表不变量(本地页=块号) | npu_cp_plan.cpp:914-953 |
| 图 padding 三件套 | decode_acl_graph.py:1406-1480/:1727-1741 |
| 后端选择 | model_executor/executor.py:126-176 |
| 宽度协商 | framework/kv_cache_transfer/kv_cache_transfer.cpp:97-145 |

⚠️ 证据边界:CANN 闭源核内部行为(SFA 首停、LightningIndexer -1 页号、融合核顺序)由 xllm 侧 docstring/调用参数交叉印证;逐位一致性前提无跨 rank 比对测试;padding 浪费量级无 benchmark。

## 来源与处置

- 三轮教学书(教学流水线产出):一轮 5 问 2707 行 / 二轮 9 问 2746 行 / 三轮 1 问 878 行,内容已全吸收进本页;
- 会话讨论:单会话首轮教学 + 三轮追问的纠错回合(glm5.2 allreduce 记忆裁决、两个"5"、"allgather 了所有 token"方向纠偏);
- 关联主题:prefill 侧 PCP 复制写、KV 分条 rank 布局、PD 传输、宽度准入(同系列深潜笔记)。
