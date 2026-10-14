---
title: 深入解析 linear tail prefix cache（四代教学书结算）
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

# 深入解析 linear tail prefix cache（四代教学书结算）

> **结算来源**：四代教学书（round1 2487 行 / round2 3968 行 / round3 2902 行 / round4 1262 行，共 10,619 行，经子代理逐本全文通读）+ findings.md 审查轨（254 行，本人通读）+ 本会话 4 轮追问讨论。工作区已清理，**本笔记自含全部关键内容**。
> **核验基准**：4 commits，34 文件 +1436/−277。数值约定：KV 逻辑块 = hash 粒度 = **128**，LINEAR 检查点步距 chunk_stride = **8192**；hash 下标 0 基（8064/128−1=**62**，共 63 块——0 基/1 基双写纪律贯穿四代书）。

## §一 MR 总览与作者意图

**一句话总纲**：给「只能整段恢复」的线性状态（GLM-5.3 系 GDN linear attention），在每一个物理层级（HBM→host→Mooncake Store）钉死同一套「可恢复边界 + 边界键」，让 8191 这类不对齐 prompt 的前缀复用从「全部作废」变成「只重算最后一段」（8191 prompt 复用深度 0→8064，98.5%）。

**4-commit 演化**：
- **commit-1** feat 全量引入（25 文件 +864/−222，含 llm/vlm_engine 启动期校验）
- **commit-2** revert 撤 engine 校验（净 diff 于 distributed_runtime 为空；校验落 LinearStateBlockManager 构造期 `CHECK_EQ(chunk_stride % token_block_size, 0)` linear_state_block_manager.cpp:114-117——host LINEAR 叶在层级池直接构造 hierarchy:448，engine 层校验挡不住；此解释为教学推断非作者原话）
- **commit-3** bugfix 边界保持（21 文件 +563/−127：tail 贯穿 host admission/prefetch、hash 按 (hasher_type,stride) 分域、调度读构造期 stride；commit 自述在其开发环境通过 mtp_prefix_cache_test 173 / sequence_mtp_bootstrap_test 9 / scheduler_test 122）
- **commit-4** refactor 集中化 hash 映射（9 文件：`block_utils.h:29-36` 成为三消费方共用的单一事实源，删 `linear_state_hashes_` 独立链全套接口）

勘探意图验证 15/15 一致，无实质意图-代码偏差；2 条 [待验证]（stride 桥固定读 dp_rank 0 的隐式同构前提、测试通过数无法只读复跑）。

## §二 tail 检查点数学（一个纯函数）

### 2.1 四条路径逐行

**`linear_state_tail_checkpoint_tokens(prompt, kv_bs, stride)`**（scheduler_config.h:102-118，入口包装 linear:343-350）：

```
路径1 守门：stride==0 || kv_bs==0 || prompt<=kv_bs → 0
路径2 算边界：tail_tokens = (prompt / kv_bs) * kv_bs   // 最后一个完整 KV 块边界
路径3 双排除：tail_tokens == prompt → 0     // 规则一
             tail_tokens % stride == 0 → 0   // 规则二
路径4 唯一有效出口：return tail_tokens
```

**手算四例**（kv_bs=128, stride=8192）：
- prompt=8191：floor(8191/128)=63 → tail=**8064**；8064≠8191 且 8064%8192=3584≠0 → **有效**
- prompt=16383（逻辑 256）：63×256=**16128**；16128%8192=7936≠0 → 有效
- prompt=20000：156×128=**19968**；%8192=3584≠0 → 有效（forward 余量 32）
- prompt=8192：tail=8192==prompt 且 8192%8192=0 → **双条件同时命中 → 0**
- prompt=8193：tail=8192，8192%8192=0 → 规则二 → 0

### 2.2 两条排除规则的物理意义

**规则一（tail==prompt→0）**：命中后 forward 至少要算 1 个 token 才能产出输出。且 KV 是 128 粒度、LINEAR 是整段折叠——就算让 KV 退到 8192−128=8064（64 块给 63 块），LINEAR 也无"读到 8064 时刻"的中间态，两条恢复边界差 128 永不收敛：重算夹缝→LINEAR 二次折叠污染；跳过→注意力读不到这 128 token 的 KV，第一个输出 token 就错。（⚠round1 勘误：「退 1 个 token」缺 128 粒度前提，物理上不存在退 1 token。）
**规则二（tail%stride==0→0）**：该边界已是常规检查点，存两遍只会双份拷贝、新旧说不清。

### 2.3 "8064 恰命中 tail 没 token forward"构造不出来（三重兜底）

用户质疑：查常规链时切了 1 个 token（linear:580 `token_ids.slice(0, size-1)` 给 forward 留 token），查 tail 点查（linear:367-368）不切——prompt 恰好 8064 命中 tail 岂不是没 token forward？裁决：**构造不出来**：
1. 规则源头：prompt=8064 → tail==prompt → 0，点查不发起；有效 tail ≤ prompt−1（prompt∈[8065,8191] 时 forward 余量 ≥1）
2. 挂载断言：`CHECK_LT(num_shared_tokens, current_total_num_tokens)` sequence_kv_state.cpp:255-256
3. 消费侧：`num_need_compute_tokens() = num_tokens_ − kv_cache_tokens_num()`（sequence.h:210-211），深度==prompt 则待算 0，无法组 batch

截尾不对称=同一不变量（恢复深度<prompt）的两种写法；手算证明 tail 若也切 1：prompt=8191 时 (8190/128)×128=8064 同值——重复设防无收益。

### 2.4 prompt=8192 全链路（五步）

①调度整片 0→8192（scheduler_policy.cpp:505-517，tail=0 不触发 min；对照 8191：min(8192,8064)=8064）→②检查点永不发布（cache_completed_state 闸门 linear:385-395；常规分支只在下一分片准入触发，8192 无下一分片）→③重复请求 LINEAR 零命中（截尾后 8191/8192=0 个可探测检查点；tail 因规则返回 0）→④KV 64 块全退（trim 中 regular_tokens=0→safe_count=0→全进 to_drop）→⑤**正确的全量重算**——"不管有没有 tail 都有问题"直觉对，规则把"都有问题"提前变成返回 0（宁可不错用）。

### 2.5 调度封顶

`cap_prefill_chunk_at_linear_checkpoint`（scheduler_policy.cpp:485-521，三调用路径 unified_policy.cpp:176/:222/:245）：四提前返回（candidate=0/未开 chunked prefill/未开 prefix cache/无 linear 层）→stride 优先读 manager 桥（`BlockManagerPool::linear_state_checkpoint_stride()` block_manager_pool.cpp:147-156）→核心三步：到常规检查点距离（offset==0?chunk_size:chunk_size−offset）、tail 距离、取近再与预算取小。

**手算（逻辑 kv_bs=256，物理 128×kv_split 2，测试 scheduler_policy_test.cpp:747 双策略）**：
- 8191：tail=31×256=7936 → chunk1=min(8192,7936)=**7936**，chunk2=min(255,256)=**255** → 分块 **{7936,255}**；对齐校验 7936%256=0
- 16383：tail=16128 → **{8192, 7936, 255}**
- tail 时刻断言：tail 标记=7936、LINEAR 2 块、设备前块 hash==block_hashes()[7936/256−1]=**block_hashes[30]**（tail 键 30 vs 常规 k=0 键 31 相邻不冲突）

固化 stride 防单例漂移（测试 :714-740：manager stride=512，SchedulerConfig 单例改 8192 后 cap(384,192) 仍 128u——若读单例结果会跨过边界 512）；decode 实例 stride=0 不封顶（continuous_scheduler_test.cpp:745-747）。

## §三 键空间：两把尺子、hash 域与 MTP lookahead 的物理本质

### 3.1 两把尺子与统一映射

- **尺子一 checkpoint stride=8192**：构造期从 SchedulerConfig 单例读走后固化（composite:165-173）；调度读桥函数而非单例
- **尺子二 hash stride=128**（=KV 逻辑块）：键粒度（make_linear_state_options linear:33-52，block_size 与 hash_block_size 两个正交刻度同框）
- **键=KV 共享哈希链稀疏采样**：`linear_state_checkpoint_hash_index = checkpoint_tokens/hash_block_size − 1`（block_utils.h:29-36，含对齐 CHECK_EQ——防边界落块中间：如 8192%300=92≠0 时公式给 26 但第 26 号只覆盖前 8100 token，可能撞别的前缀错算）。减 1 因节点 i 覆盖 [i·bs,(i+1)·bs)，边界 N 由最后覆盖它的节点 N/bs−1 见证。三消费方：linear_state_prefix_cache.cpp:27-32 / linear_state_block_manager.cpp:120-125 / hierarchy_block_manager_pool.cpp:548-556——漂移=静默 cache miss（commit-4 收口动机）
- **KV-split**：物理 128×kv_split=逻辑 256；tail 边界与键必须用逻辑块（scheduler_policy.cpp:512-513 注释"logical KV block size already includes KV split"）。用物理 128 算键：16128/128−1=125 vs 正确 62 → 存 62 取 125 永远 miss

**手算四把钥匙**（hash_bs=256, stride=8192）：常规 8192→32−1=**31**、16384→**63**；tail 7936→**30**、16128→**62**。tail 键与常规键同链相邻不冲突；相邻检查点隔 8192/256=32 个节点。测试锚点 block_manager_test.cpp:788（hash_bs=128）：tail=8064 键下标=8064/128−1=**62**（共 63 块；断言 shared_blocks_num(KV)==63u 是块数非下标——round1 五处"63→62"勘误的根因）。

### 3.2 hash 域双层桶

`block_hashes_by_domain_[hasher_type][block_size]`（sequence.h:648-658）；`hash_type_`/`hash_block_size_` 是"最近一次 update 选中的台牌"（成员变量=**合理妥协**：12 处读点免传参 vs 隐藏可变状态无编译期防护；忘 update 四步失败画面：LINEAR 发布静默拿到 MTP 域桶→块盖错键→probe 永远 miss 无日志）。**无 MTP 1 桶**（KV+LINEAR 同 TEXT@128）；**MTP 2 桶**（MTP_TEXT@128 KV 用 + TEXT@128 LINEAR 用——**LINEAR 构造期剥 MTP 变体** composite:182-186 三元式）。切台牌不删桶（三证据：update 只延长自己桶 / block_hashes() 两级 find / invalidate 遍历所有桶）。链式摘要：第 i 格=hash(本格 128 token + 上格摘要)，首格空指针标记（block_hasher.cpp:160-165）；改写任一早期 token，所在格起全部摘要变脏。

**三连调用**：update_block_hashes（选桶切台牌）→ hash_tokens(hasher_type)（备料：MTP+overlap 时裁到已确认游标 slice(0, cur_generated_token_idx_)，草稿不进 MTP 料）→ extend_prefix_hashes（增量接龙到 num_hash_blocks=(tokens−lookahead)/bs 封顶）。

### 3.3 MTP lookahead 的物理本质（round4 终裁——四代书最深的一层）

**用户直觉裁决**：「前缀相同就该共享 block」在 TEXT 域**完全正确**（四件代码证据：block_hasher.cpp:72-76 lookahead=0 / :78-86 N/bs / composite:41-48 min(kv,N) / prefix_cache.cpp:52-61 逐块走链；测试 CopyRetainsMtpInvalidationPolicy TEXT 对照组：改 token[16] 后块 0 键不变 EXPECT_EQ）。

**MTP 域块是复合材料**（五步证据链）：
1. 一个 Block 字节 = target 层 + draft 层之和（speculative_engine.cpp:310-313；draft KV 独立显存池但同块号两池各一份，mtp_worker_impl.cpp:918-931 镜像同型）
2. **draft 输入右移一位**：prepare_prefill_inputs（mtp_worker_impl.cpp:1505-1517）slice(start_idx+1) 丢第 0 个 + 尾补 extra_token_ids ⇒ **draft 层槽位 j 的 KV 依赖 t_{j+1}**（DeepSeek 式 MTP 头输入 = h_j 拼 Embed(t_{j+1})，"看着当前猜下一个"）
3. 越界一格的 extra token 由 target 侧打包（batch_input_builder.cpp:868-899）
4. draft 层数=num_nextn_predict_layers 独立（worker_impl.cpp:2622-2639）

⇒ MTP 块 i 依赖到 t_{(i+1)·bs}，比 TEXT 块恰多一个 token；MTP hash 的 end+1（block_hasher.cpp:42-46）与此**精确匹配**——键覆盖 ⊇ 内容依赖是 key→value map 语义的基本要求（同键必同内容），不是性能优化。**"前缀相同、下一 token 不同"的两个请求，MTP 块是 draft 半边内容不同的两个块——键区分下一 token 是如实区分两个内容不同的东西**（测试 HostMatchRejectsDifferentNextToken：33 全 7 token，fork[16]=8→**0 命中**（块 0 draft 槽位 15 吃 t_16）、fork[32]=8→**16 token 恰一块**（块 0 完全相同）；每格拒绝共享对应真实内容差异，无过保守）。

**失败画面六环链**（若无 lookahead 撞键）：①B 键与 A 相同→命中挂载〔键侧〕②draft 槽位按 A 的 t 算，对 B 是错误内容〔内容侧〕③MTP 头拿坏 draft KV→提案质量降④**verify 阶段 target 输出仍正确（target 半边干净）——不是输出错误**〔结构性质〕⑤提案差→拒绝率升→**静默持续掉速**（污染槽位在游标前永不覆盖）〔幅度无基准 [待验证]〕⑥同键不同内容——身份契约破坏〔定义层〕。（⚠round4 勘误 round3："正确性事故"降格为"静默吞吐事故"；"多算 bonus token"讲法错误，真机制=复合材料+右移。）

**LINEAR 剥 MTP=同一公理反方向**：LINEAR 检查点是边界确定函数 S_n=f(t_0..t_{n−1})（递归状态**不可能**依赖未来——对比 KV 草稿半边可以）。三件证据：hybrid spec verify 下 LINEAR 物理推进投机窗口（conv 状态长 kernel−1+k、SSM 槽数 k+1，kv_cache_estimation.cpp:278-286/:828）但**检查点只落确认边界**（linear:69-89 只用 last_confirmed+accepted_length；注释"bonus token also advances the recurrent cache"——推进≠发布）；剥域代码 composite:182-186。定量四行表（bs=16，33 确认 token，改 t_16 触发 invalidate(16)）：

| 策略 | lookahead | 首脏块 | 作废 | 道理 |
|---|---|---|---|---|
| ① MTP 现状 | 1 | 0 | 2 格全废 | 有：draft 半边真吃 t_16 |
| ② TEXT 对照 | 0 | 1 | 1 格 | 有 |
| ③ LINEAR 现状 | 0 | 1 | 检查点 0 保住 | 有：S_16=f(t_0..t_15) 不含 t_16 |
| ④ LINEAR 若用 MTP 键 | 1 | 0 | 2 格全废 | **无：纯自残** |

### 3.4 invalidate 传播与盖键

`invalidate_block_hashes_from`（sequence.cpp:1013-1027，仅两调用点：commit_token:567（MTP/overlap decode 位覆写）/ update_token:596（PD 回填））：遍历所有域各用自己的 lookahead；`stale_token = token_index − lookahead`，首脏块 `⌊stale/bs⌋`——**回退量=覆盖量的逆运算**（MTP 块 1 前瞻吃过 token 8 所以块 1 也脏）。手算（bs=4 改 token 8）：MTP stale=7→首脏块 **1**；TEXT stale=8→首脏块 2。错误版不回退（8/4=2）留脏键→五环失败链：脏键发布→登记→未来请求算出相同摘要→命中→恢复的状态 token 8 贡献是旧值→每步 decode 复利叠加。

**盖键**（`Block::set_hash_value` block.h:169-171，memcpy 16 字节；行李箱挂名牌——键跟着数据走）：三个时机——发布（linear:423/:440）、预取分配（linear:230）、D2H 配对（hierarchy:565 `pair->dst.set_hash_value(pair->src.get_immutable_hash_value())`）。hash 链节节都有（每 128 token），**键只盖完成可复用边界的块**是设计非失误；不做→"有数据没钥匙"→host 已盖键版 insert 拿垃圾键登记→预取永远 miss（白拷一份）。

## §四 账本与生命周期

### 4.1 数据结构与 erase 全删

**KVCacheState 四 map 按 BlockType 一条记录**（sequence_kv_state.h:191-227：composite_blocks_/block_sizes_/num_owned_shared_blocks_/num_cached_blocks_ + linear_state_tail_tokens_:227），**无单块元数据**（单块 id_/size_/ref_count_/manager_/hash_value_ 在 Block 句柄 block.h:184-199）——"删一个块"需要的按块信息恰是账本没存的。因此 `erase_blocks`（:156-164）整 type 全删（"要么不删要么删完"=数据结构直接后果，**好设计**；欠债=描述-向量同步无断言）。**部分删除三步流水线**：take（:166-183 整串取出抹元数据）→vector 修剪（resize+deallocate）→replace（放回+重申报 (num_shared, cursor)）——四实例：KV 预取收尾 / trim_blocks_from_back / LINEAR 预取收尾 / retain_read_source 原地修剪。erase 全库仅四调用方：fork/beam（sequence.cpp:374-378，私有槽不继承）、KV trim 后空向量、LINEAR trim 无源、host 裁光——共同点=整页失去存在意义。"删完"删的是账，物理块死活由引用计数决定。

### 4.2 linear_state_tail_tokens_（tail 便签）的一生

语义"本 tier 保留/发布过 tail，边界在此数字"；设备/host 各一份。写入 3 点（发布 linear:430-443、准入挂载调用方补设 composite:926-930、预取放回 linear:264-268）；清零 5 入口（erase:158 / take:168 / mount:271 / replace:294 / reset:431）——**向量动过则便签作废，调用方在语义明确后补设**（假便签失败画面：mount 不清零→新序列以为有 12345 深度的 tail，向量里没有源块，恢复从第一步错位）。

**皮之不存毛将焉附的答案**：便签序列私有随 finish 归零（Sequence::reset sequence.cpp:962-969→两份 KVCacheState::reset）；**发布后真正存活的是缓存条目**（key→Block，LRU 节点，ref≥1）——未来序列 B 自己重算 tail 键（链式 hash 确定性）→find 命中 ref+1。**B-Lifecycle 手算**（prompt=8191，tail 块 id=77，chunk=8064+127）：T1 A 发布块 77 **ref=2**（序列+缓存）→T3 A finish 五步收尾 **ref 2→1**（只剩缓存）、便签归零→T4 B probe find 命中 **ref 1→2**（并行 C 也命中则 3）→T5 B mount+便签 8064→T6 B 只 forward **127** token。发布块不可变只读所以共享安全；LRU 驱逐只收 ref==1（prefix_cache.cpp:248-253）；覆盖只发生在归零回池→再 allocate 写新数据。

### 4.3 finish 五步与块三结局

**finish≠erase_blocks**（round2 裁决 round1）。五步（hierarchy:491-510）：①最后一搏发布（cache_full_blocks_for_sequence）②collect_offload_pairs 配对下沉（:512-685，源块盖键抄给目的块）③host 账本掉引用 ④设备 deallocate_for_sequence（"ref 2→1"发生地 composite:538-549）⑤reset 只抹账面。三结局：**留缓存**（块 77：发布过 ref≥1，活到 LRU 驱逐）/ **回池待覆盖**（块 78：decode 活写槽从未发布——只有完成到检查点/tail 边界的槽有发布资格，ref 1→0→free()）/ **D2H 下沉 host 第二次生命**（块 79：跨检查点的源块，finalize_host_blocks hierarchy:44-63 后 host 目的块发布、设备源块释放）。

### 4.4 双账本

常规账 `num_cached_blocks`（量纲=检查点个数，1 基）vs tail 账 `linear_state_tail_tokens`（量纲=token 深度）——**量纲不同注定分账（合理妥协）**；无任何一行同时写两账。**四个汇合点**：①准入比大小（trim 用 probe 结果）②挂载此消彼长（composite:928 清零+:929 设 tail——mount 把游标设成向量长度，tail 挂载向量 1 块→游标 1 是假账；**游标虚高 k 的账**：发布守卫 `checkpoint_index<=num_cached_blocks` linear:414 跳 k 次真发布+补齐起点把 tail 状态错标第 k−1 检查点——k=3 时账面 24576 vs 实际 8064，**错位 16448 token** 复利放大）③发布各自守卫（常规 :414 用常规账；tail :430-431 专用）④预取 tail 优先短路否则常规定起点。成长期两账可并存（(常规 3, tail 19968)——三个不同物理源各记各的；固定 prompt 下"tail 挂载后常规账再动"构造不出来：剩余 prefill≤127 < 到下一检查点距离 128，三步数学证明）。

### 4.5 probe 哨兵与 LRU

**ProbeResult.logical_token_count 哨兵**（默认 0，全库唯一非零赋值点 composite:751 trim 的 tail 胜出分支写、挂载读 composite:927）——"是 tail"的判定靠数据携带（组合表达：非零 logical_token_count+游标 0）非显式 flag。**probe 与 LRU**：命中即 `lru_lst_.remove_node`+push_back 刷新到最新端（头部=最旧先死 evict 从 get_first() 开始）；**浅命中块从没离开过 LRU**，release_probes（composite:621-640）只掉引用不动 LRU 且在 trim 之后——非假阳性（键真内容真，被丢是跨叶对齐"队友拖后腿"），"探测性保鲜"=**合理妥协（行业常规）**（vLLM get_cached_block 同刷新 [待验证]、SGLang 命中向 root promote）。LRU 4 节点手算：probe 前 [N1,N2,N3,N4]（左=最旧）→请求 probe N1/N2 命中摘出→收尾 push_back→[N3,N4,N1,N2]→trim 砍检查点 2→LRU 纹丝不动——实际只挂载 N1 但 N1/N2 都跳到最新端。

## §五 发布与准入流水线

### 5.1 发布

- **cache_completed_state**（linear:385-395）守门：cached_tokens==tail 才发（只发布恰好停在 tail 边界的完整状态；常规检查点由 cache_read_source 经 chunked prefill 每步发布）。cache_read_source 三道门（:397-412）：host_kv_state 直接 return（**host 槽是 D2H 目的地，只有传输完成回调可发布**——预定座位≠数据已到）/ graph warmup / 无 prefix_cache
- **tail 分支**（:430-443）：幂等条件（`tail_checkpoint_tokens(seq)!=cached_tokens || kv_state.linear_state_tail_tokens()==cached_tokens` 则 return）→ tail_checkpoint_hash 取键（hash_index=cached/128−1）→盖键（注释"Stamp the source itself so D2H carries the same tail key"）→ insert → set 便签
- **常规分支**（:412-427）：checkpoint_index=cached/8192 →游标守卫→取键→盖键→insert 最后一块切片→游标推进
- **insert 发布了什么**：Block 五字段（id/size/ref_count/manager/hash_value_[16B]），**不带 BlockType**（每叶独立 PrefixCache 隐含）、**不带逻辑下标**（深度编在链式键里——链式第 N 节代表前 (N+1)×128 token）；发布=登记 map+ref+1 **无数据搬运**。三种 insert 重载：带 token_ids 版（KV 常规）/ 已盖键版（LINEAR 常规+tail+D2H 后 host）/ LINEAR gap-tolerant 版
- **失败画面**：去幂等→LRU 同键两节点引用计数虚高占槽；去盖键→host 有数据没钥匙；去守门（算到 8000 发 8064 检查点）→64 token 贡献凭空消失，长上下文质量劣化不报错

### 5.2 probe=并列两问（round2 推翻 round1 叙述）

**不是二选一回退链**：问一常规链 **gap-tolerant 全遍历**（linear_state_prefix_cache.cpp:39-125：miss 放无效占位继续走、hit 拿共享块+LRU 刷新，记录最深命中；头注释"deeper hit subsumes earlier ones"）；问二 **tail 单键点查**（linear:365-383：新 prompt 自己算 tail→算键→find 这一个键，零探索；tail 不比常规深当场 deallocate composite:602-614）。**汇合 trim_flat_kv_linear**（composite:674-712）：tail 优先分支（tail_safe_count×128==tail_tokens 整块盖满且块有效→用 tail、常规命中全退）；常规回退分支 **每次减 1 个 KV 块（128 token）**直到踩在有效 LINEAR 检查点（linear_index=safe_hit/8192−1 且 is_valid()）——**"按 block 回退"在 trim 的 KV 对齐，不在 probe**（vLLM mamba 的右→左 Early-stop 才是逐块回退查找）。

**F-probe 手算（prompt=20000 重复请求）**：A 留下 tail=19968（chunk 划分 8192→16384→停 19968→32）、发布过检查点@8192（键 hashes[63]）/@16384（键 hashes[127]）/tail@19968（键 hashes[155]，共 156 块）。B 问一：可探测检查点 19999/8192=2 个→hashes[63]✓、hashes[127]✓ 最深 16384；问二：tail 19968→hashes[155]✓。汇合：tail_safe_count=min(156,156)=156，156×128==19968 ✓→**用 tail**、常规 2 块退还、KV 保留 156 块。挂载：游标清 0+tail 标记 19968→B forward 剩 32 token。

### 5.3 trim 的本质

KV 与 LINEAR 裁到同一 token 边界，**宁可少复用不可错复用**。失败画面（无 trim）：KV 命中 8064、LINEAR 常规只到 4096 直接挂→4097..8064 共 3968 token 贡献重复计入→语义漂移不崩溃。手算 E 例（bs=4/stride=8/prompt=13，TailMismatch 测试 block_manager_test.cpp:827）：常规命中 8；tail 键 12/4−1=2 查不到→回退 safe_count=min(2,2)=2，8%8==0 有效→KV 2 块全留、mount 游标不清零（输出 KV=2/LINEAR=1/num_cached_blocks=1）。纯 tail 对照（:763，prompt=7/kv_bs=4）：tail=4 命中→mount 清零+set(4)。

## §六 三层存储与 prefetch 事务

### 6.1 三层全景（round3 图先行）

**HBM 宿舍 / host DRAM 仓库 / Mooncake Store 云盘**；D2H 下沉、H2D 恢复、G2H 远端拉取。命中顺序：**第 0 层**（入队前，ContinuousScheduler::add_request:254→prefetch_from_storage hierarchy:985-1096）：host 准入 probe→base→target→各叶分配→取货单→G2H 异步→finalize 回调→才入队；**第 1-2 层**（正式调度，allocate_shared hierarchy:854-892）：先 device probe 后 host probe（:867-877 代码顺序），host 更深则 H2D 补差额（collect_load_block_transfer_infos :783-852，begin=hbm_cached/128：设备 8192/host 16384→只拷第 64..127 号块）。三条主干：**三层叠加补差额非三选一；Store 永不被 probe 只被取货单向它要东西；两段时机被异步回调切断**。优先级 device（免搬运）>host（省 Store 往返）>Store（兜底最深）。

### 6.2 D2H 引用计数与 EXPLICIT

**配对=move 所有权搬家零减引用**（hierarchy:557-575，注释"pair owns both blocks"；源块盖检查点键:557、键抄目的块:565、空壳 erase）→发起（:1129-1161 folly::collectAll，worker 数=host_transfer_worker_count）也不减→**减引用只发生在完成回调**（:1163-1209：deallocate 设备源+finalize_host_blocks insert host 目的）——"每份别名恰好减一次，异步只是推迟；同步/异步最终都减"。六行手算（KV 块 #201 配 host 空块 #501）：t0 设备发布 #201 ref=2；t1 host 分配 2/1；t2 配对 move 2/1；t3 发起 2/1；t4a 回调成功 #201=1（缓存持）/#501=1（host 缓存持）；t4b 失败 #501=0 回池。失败画面（早减）：空闲池重分配→D2H 还在写→两份数据同时报废。

**EXPLICIT 发布模式**（枚举 composite_block_manager.h:77-80；host 构造显式传 hierarchy:469-472，设备默认 AUTOMATIC 三钩子）：设备块随时成品（forward 写完即终态）；host 必须显式因**已分配≠数据到位**（linear:399-403 注释）——防半成品发布（AUTOMATIC 会把空柜子 insert 进缓存→别的请求命中半截数据）。三个显式发布入口：D2H 完成回调 / 预取收尾 / LINEAR trim 常规发布。用户两个猜测（"host 需显式发布"与"D2H 已发布后不再发"）=同一枚硬币两面（机制禁用+效果已发）。

**D2H 门禁**：全 worker 完成才可见（两层：results.size()==expected_workers :1171-1176 + 逐 worker 拷贝数对 :1188）；失败只释放不发布。手算 G 例（2 worker，测试 hierarchy:2658）：finish_worker(0) 后仍 miss ✅→finish_worker(1) 后可见（tokens=12、LINEAR 1 块=destination_id）✅。失败画面：2 worker 回 1 个就发布→整段折叠张量半新半旧=整块作废且无一致性标记。

### 6.3 prefetch 关键变量与取货单

**八变量速查（prompt=20000 tail 命中）**：unit_size=8192（hierarchy:1013）；max_prefix=19999（留 1 token）；cacheable=19968（各叶 min 木桶短板）；regular_target=16384（整单元向下取整）；**base_tokens=19968**（:1039 host 命中深度，"已经不需要再取的部分"）；**target=max(regular,base)**（:1043——零常规目标不能丢已命中 tail；例 D：prompt=5000→regular=**0** 但 target=4992）；final=19968。`set_prefix_cache_matched(false)`（:1038）=诚实标记"Store 回调验证前 host 不算正式匹配"。

**取货单=缺什么取什么**（最小重复三处证据）：①从 base_units（:212-214）起开到可用单元数——本地已有不重复取；②只列有效槽且分门禁（KV=gated 一个单元 64 块任一没分到整个单元砍单；LINEAR=non-gated 可选）；③收尾防半吊子（transfer_ok=units.empty()||gated 全中，缺一整单作废——链式 hash 半截键链断下次也命中不了；final=base+连续命中门控单元数×8192）。**粒度澄清（round3 裁决直觉）**：G2H 只取 8192 整单元；**tail 深度从不经取货单**（host 的 tail 条目来自 D2H 下沉）；"host 命中 16384、Store 有 19968→取差额"**不成立**（host=16384 时 target 也 16384、取货单为空；3648 非整单元）——正确非空单例：host 命中 8192→只开单元 1。

**tail 短路与放回**：LINEAR 叶 allocate_for_prefetch 进门查 tail（linear:150-159 两 CHECK：目标≥tail 且同检查点区间——例 E：tail=12/stride=8，CHECK2 12/8==12/8 同区间 1，目标跑 16 则区间 2 宁崩不错）→return true（物理源一份、不占常规向量槽位，"including when the tail crosses one or more regular checkpoints"）。trim_prefetch_blocks（linear:254-269）：take 清过标记→`replace_composite_blocks(...,1,0)`+`set_linear_state_tail_tokens`——**放回≠发布**（该分支无 insert，块本就是 host 缓存共享块；抽屉类比：取出核对→原样放回→补标签；两个申报数字 num_shared=1 如实/cursor=0 因 tail 不占常规名次——写错会被盖更早检查点的键）。"max_hit<tail→deallocate"分支**主路径不可达**（tail 命中⇒取货单恒空 hierarchy:192-194⇒final=base=tail 三步不变量链）=防御护栏（round2 裁决旧书过度演绎）。

**冷缓存全程（prompt=20000，Store 存到 19968）**：base=0→target=16384→host 叶 KV 建 128 块+LINEAR 2 检查点位→取货单 2 单元（base_units=0）→G2H 门禁全中 final=0+2×8192=16384→入队→正式调度 H2D 拷 128 块→forward 3616（chunk：16384→停 19968→32）→期间 D2H+发布检查点 1/2/tail。**下一个同 prompt 请求：host 准入命中 tail 19968→取货单空→只 forward 32 token**（tail 的两副面孔：本请求算出来，下一请求命中到）。

## §七 设计评价汇总（四代书终裁）

| 对象 | 结论 | 一句话理由 |
|---|---|---|
| MTP lookahead 进 hash | **好设计**（hash 链路线内） | 键覆盖=内容依赖的必然（复合材料+右移）；复用条件编码进键零运行时校验；是 hash 链追平树表达力的税 |
| invalidate 域差异 | **好设计** | 参数化而非特判（一条循环各域按各自值回退）；回退量=覆盖量逆运算；欠债=新增域忘登记静默 0 |
| 双账本 | **合理妥协** | 量纲不同注定分账；挂载互斥有 composite:928 钉住；欠债=成长期并存无断言、漏一处 set 静默错位 |
| 三层+tail 短路"不敞亮" | **合理妥协，三七开** | 七分讲法（round2 无全景图已修复）三分实现欠债（短路散两处/回调切断控制流/标志先立后破） |
| erase 全删 | **好设计** | 组级账本+单块元数据下沉 Block 的分工；欠债=描述-向量同步缺断言 |
| hash_type_ 成员变量 | **合理妥协** | 12 读点免传参 vs 隐藏可变状态 |
| probe 即 LRU 刷新 | **合理妥协（行业常规）** | 浅命中浪费根因在跨叶对齐不在 LRU |

**元问题裁决**（"是巧妙还是傻逼/炫技还是不灵光/存在的意义"）：复杂度 9 项三分账——**物理必然 5**（end+1 / 发布晚一格 / fake 不进链 / 按域回退 / 剥 MTP）、**路线税 2**（域分桶选择器 / 全链 13 处 update-then-read 对齐）、**欠债 2**（无"块是复合材料所以键要 lookahead"的串联注释——**用户三轮追问才得到答案的根因**；键覆盖=内容依赖无结构化断言，新增域忘登记静默 0）。指控裁决：**深思熟虑的工程决策**（四条可证伪证据：2f0affd37 定向 bugfix 列四条对齐项、450 行专项测试五维度、参数化无特判、vLLM 独立遇到同一物理写同款 docstring）**+ 文档欠债**——"决策好"与"文档烂"分开算账；若那段 10 行机制注释早存在，三轮追问至少两轮不会发生。**可带走的五步审计框架**：找内容物理事实（翻到写入处）→写约束公理→反事实手算（代价算成格数）→找参照系（别人怎么解付了什么）→查工作痕迹（设计/文档两结论分开）。

## §八 三方对照（xLLM / vLLM / SGLang）

同一物理问题（MTP 块含依赖下一 token 的 draft 层 KV——**vLLM find_longest_cache_hit docstring 亲口承认**"the dropped block's draft-layer KVs... depend on up to num_speculative_tokens−1 tokens past the matched prefix, which the block hash does not cover"）三种解法：

| 维度 | xLLM | vLLM | SGLang |
|---|---|---|---|
| 键原料 | 链式摘要，MTP 域多吃 1 token | 父 hash+本块 token+extra_keys，无 lookahead | token 序列本身（radix 树，page_aligned） |
| 防撞键在哪层 | **键层**（烤进键） | **消费层**（丢边界块重算 hit_length−=min(alignment,bs)，重算重写 draft KV 冲掉污染） | **结构层**（不同子节点无法构造同键） |
| draft KV 在共享块 | 在（复合材料） | 在（docstring 承认） | 不在（draft 侧独立管理，树只插确认 token） |
| 代价 | 三处对齐+读者不直观 | 每次命中少复用一块（**无条件税**） | 放弃 O(1) 增量链；树内存/锁；Store 键格式随路线 |
| 复用的东西 | target+draft 半边整体（**三家唯一**） | 到丢块为止 | 仅 target 侧，draft 每请求自算 |

复用深度手算（A 33 token 已发布；B_match 全同 / B2 改 t_32=8）：xLLM **2 块 / 1 块**（实测 mtp_prefix_cache_test.cpp:594-606）；vLLM **1 块 / 1 块**（两列相同=丢块无条件税，B2 这次丢"对了"）；SGLang **全程 / 32 token**（B2 更深因共享块不含 draft 半边，代价 draft 永不共享）[vLLM/SGLang 数字为推演待验证]。SGLang 线性检查点网格=lcm(mamba_chunk_size,tree_page)、KV 命中向下对齐 `(hit//grid)*grid`；xLLM 有 tail、SGLang 无（SuffixReplay 论文视其为现有系统做不到的事）、vLLM 有近亲 partial_tail [待验证]。

## §九 Review findings（已发布 MR + 工程记录）

**已发布 6 条**（读回验证 6/6 逐字一致，与 MR 已有讨论零重叠）：
- **P2×1**：构造期 `chunk_stride%token_block_size==0` 对齐 CHECK 使先前可运行配置启动即崩（KV split 场景风险最高：kv_split=3 时逻辑块 384，chunk 8192%384=128≠0），报错不含实际值、docs 未载；建议 CHECK 补两值或 build_composite_leaves 向上取整（有 chunk_stride≤0→1 归一化先例）
- **P3×3**：tail 快路径 CHECK 恒真自证断言缺 guarded-scenario rationale；`hash_block_size==0` 兜底三元三处复制（漂移风险）；match() 降级路径从可续链退化为全链重算（低频性能）
- **Q×2**：mm 感知 hasher 从 per-sequence 改引擎级映射（"TEXT 引擎+mm_data 序列"可达性需作者确认）；性能契约三项可测成本无数字（封顶步数增量/双域哈希开销/冷热复用收益）
- 总结论：**未发现 P0/P1**（tail 边界数学/probe-trim-mount 状态机/prefetch 收敛/D2H 门禁/hash 域隔离均经手算+双向调用链核验，推导记录 N1-N6）；建议 P2 修复后合入

**unverified 5 条（未发布）**：commit 声称测试数无法复跑；端到端性能无 benchmark；兄弟分支（同主题早期迭代，与本 MR 无祖先关系，日后合入将大范围冲突）未确认废弃；TEXT+mm 错配可达性；Mooncake 冷缓存未实测。

**发布工程**：每发一条评论 MR version 即刷新，多 op 必须逐条「快照→改 expectedVersion→发布」循环（批量单会话连锁 REVIEW_INPUT_CHANGED）；发布前重新快照防合入竞态（此前一次 MR 的教训）；去重对账以 body 为准。

## §十 学术脉络与会话讨论沉淀

**dengyingxu1 的 linear-state prefix cache 血脉**（GitHub PR 时代，jd-opensource/xllm→xLLM-AI/xllm）：六部曲 **#1799**（hash 原语 1/n）/**#1802**（选项 2/n）/**#1846**（block manager+prefix cache 3/n——`linear_state_hashes_` 独立链在此出生，本 MR 将其杀死）/**#1940**（恢复+容量 4/n）/**#1947**（batch+scheduler 5/n）/**#1956**（引擎接线 6/n）→**#1979** VLM 启用+chunk-stride clamp→**#2161**（prefix-cache prefill attention 的 **Python graph 路径**修复——正是 glm5_next 唯一走的 TORCH 组图路径）→**#2163**；内部平台另有三步：VLM overlap 恢复 slots、去 MTP 冗余扩张省显存、本 MR 键域重做 v2。设计笔记站 yingxudeng.github.io/xllm-notes：qwen35-linear-prefix-cache-pr1839 设计报告（PR #1839 未合入的原型）、尾块缓存笔记（commit 29b510a8e，CANN9 端到端证据）、ReplaySSM ring 设计审查。

**论文参照**（作者公开材料未引论文，两候选）：**SuffixReplay**（arXiv:2609.33477，2026-09-27，TeleAI——"Just Let Linear States Forget the Distant Past"：重放派，不存检查点、留 anchor 隐状态、命中时重放近期后缀做**近似**重建；与本 MR 检查点派（精确快照）相反；时间（发表 5 天后即作者尾块缓存笔记）与问题意识（页边界复用、检查点网格外分支、SGLang 8192 网格）高度吻合）；**Marconi**（arXiv:2411.19379，混合 LLM 前缀缓存开山，Bloom 准入）。注意 SuffixReplay 论文称 SGLang 默认 8192 checkpoint cache 与当前 main 代码（lcm(64,page)）不一致 [待验证]。

## §十一 教学演化与勘误链（四代书，元知识）

| 代 | 行数 | 病 | 代表性裁决 |
|---|---|---|---|
| round1 | 2487 | **讲错了**（事实错误） | 5 处 hash 下标 63→**62**（0 基 vs 块数混淆）；probe 混入"按 block 回退"；finish≠erase_blocks；"max_hit<tail deallocate"写成常规场景（实为不可达护栏）；"退 1 token"缺 128 粒度前提 |
| round2 | 3968 | **没讲细**（跳步黑话） | 13 条裁决（推翻 1/更正 4/重讲 2/维持 4）；黑话裸用→先定义后使用；每变量单独成段 |
| round3 | 2902 | **没讲透**（结论无实现） | 14 条裁决；引用结论必贴实现；先机制后类比（"便签"类比弃用换"装箱单"）；控制流先全景后展开；4 个设计评价 |
| round4 | 1262 | 挑战轮（物理层） | 3 处修正："多算 bonus token"→**复合材料+右移输入**；"正确性事故"→**静默吞吐事故**（target 半边干净 verify 兜底）；"LINEAR 无草稿写入"→**物理推进窗口但检查点只落确认边界**（结论反而加强）——三错同根因：**解释停在键层、物理层是空的** |

一以贯之的教训：**教学验收标准必须是"读者能沿证据复算"，而非"技术上没说错"；事实正确救不了教学失败**。四代书收口：**没有乱打补丁，只有量纲与异步带来的、可指认的复杂度**。

## §十二 追问精华：四轮问题的原文与终答（检索索引）

> 用户原话困惑作小标题——半年后回看，"我当时为什么困惑"是最快的检索键。答案只给终态（含历代书勘误后的最终版本），细节见对应节。

### Round 2（7 问，书 3968 行）

- **"prompt 只有 8192 个，恢复了怎么就错了？那该怎么办？不管有没有 tail 都会有问题吧"** → 直觉对：8192 本来就无解（整除边界天然不可复用），规则把它提前变成返回 0，走**正确的全量重算**（§2.4 五步链）；KV 是 128 粒度不存在"退 1 token"，LINEAR 整段折叠无中间态
- **"linear_state_tail_tokens_ 皮之不存毛将焉附？sequence finish 之后还重要吗"** → 便签序列私有随 finish 归零；**发布后活的是缓存条目**（key→Block），未来序列自己重算 tail 键 find 命中（§4.2）
- **"hash_type_ 都是固定的，双层桶难道只有一个桶吗"** → 是"最近选择器"非配置；无 MTP 1 桶、MTP 2 桶（LINEAR 特意剥 MTP 变体）；切台牌不删桶（§3.2）
- **"invalidate 的 token_index−lookahead 太细节看不懂"** → MTP 块 i 的哈希多吃下一格首 token（块 1 前瞻吃过 token 8 所以块 1 也脏）；回退量=覆盖量逆运算（§3.4）
- **"盖键是什么？为什么块上可能没有 hash"** → set_hash_value 抄 16 字节进 Block；键只盖完成可复用边界的块是设计非失误；不盖→D2H 后"有数据没钥匙"（§3.4）
- **"游标不清零就算错了呢？不懂"** → mount 把游标设成向量长度是假账；虚高 k=3 时账面 24576 vs 实际 8064，错位 16448 token 复利（§4.4）
- **"prefetch 说的什么玩意乱七八糟"** → 教学失败非读者问题（round1 Ch4 密度过高+黑话未定义+无全景图）；零基础重讲后核心：三层叠加补差额、target=max 保 tail、放回≠发布（§6）

### Round 3（11 问，书 2902 行）

- **"为什么 prompt=8192 时 checkpoint_tokens==0？函数实现从没说过"** → 四路径贴码：双条件同时命中走路径 3（§2.1）
- **"常规链退了 token，tail 点查没退，8064 命中没 token forward 不还是有问题吗"** → 构造不出来，三重兜底（§2.3）
- **"哪里讨论了这个分支是不是 tail"** → ProbeResult.logical_token_count 哨兵（全库唯一赋值 composite:751），数据携带分支结果（§4.5）
- **"erase 是全删？要么不删要么删完，这么极端对吗"** → 数据结构决定（四 map 按 type 一条记录无单块元数据）；部分删除走 take→修剪→replace；"trim=全清"是教学误会（§4.1）
- **"EXPLICIT 发布模式是什么？device 自动 host 显式？"** → host 块内容来自异步拷贝，"已分配≠数据到位"，防半成品发布（§6.2）
- **"hash_type_ 当临时变量不行吗，成员变量看着别扭"** → 合理妥协：12 读点免传参 vs 隐藏可变状态（§3.2）
- **"MTP 为什么这么特殊？hash 多算 token 块数却减，什么逻辑"** → 见 round4 终答
- **"区别对待是巧妙设计还是傻逼设计？炫技还是不灵光"** → 好设计：参数化非特判；"弯弯绕"感来自四环链条环环相扣+无串联注释（§7）
- **"两个独立的账怎么统一用？checkpoint_index<=num_cached_blocks 用哪本"** → 常规账（量纲=检查点个数）；四汇合点地图（§4.4）
- **"假阳性白赚 LRU 头部不公平吧"** → 方向纠正：尾部=最新端被保护；浅命中块从没离开过 LRU，非假阳性（§4.5）
- **"一会儿 host 一会儿 Store 一会儿短路，不敞亮，是你没讲明白还是实现的人太狗了"** → 三七开：七分讲法（无全景图已修复）三分实现欠债（短路散两处/回调切断控制流）（§6/§7）

### Round 4（3+1 问，书 1262 行，挑战轮）

- **"前缀相同就该共享 block，凭什么管他妈下一个 token？我不明白你在为他们掩饰什么"** → **直觉在 TEXT 域完全正确**；MTP 块是 target+draft 复合材料、draft 输入右移一位⇒槽位 j 依赖 t_{j+1}——"前缀相同下一 token 不同"的两个请求的 MTP 块是 draft 半边内容不同的两个块，**键区分下一 token 是如实区分，不是立场**（§3.3）
- **"难道 vLLM 和 SGLang 也会 lookahead 吗"** → 都不做；vLLM docstring 亲口承认同一物理、解法=丢边界块重算；SGLang token-id 树结构免费（§8）
- **"LINEAR 剥 MTP 既不是公理也不显然，像随便决策的人在掩饰"** → 递归状态 S_n=f(t_0..t_{n−1}) 不可能依赖未来（数学性质非实现选择）；LINEAR 物理推进投机窗口但检查点只落确认边界；用 MTP 键=多废一格买到零的纯自残（§3.3 四行表）
- **"不优雅、逻辑不硬、还不讲清楚，存在的意义是什么"** → 复杂度三分账（物理必然 5/路线税 2/欠债 2）；裁决=深思熟虑工程决策+文档欠债；无串联注释正是三轮追问的根因（§7）

## §十三 手算示例总表（四代书全部，参数→结果）

| 例 | 场景 | 参数 | 结果 |
|---|---|---|---|
| A | tail 边界判定 | 8191/128/8192 | tail=**8064** 有效 |
| A | 同上 | 8192/128/8192 | 双条件→**0** |
| A | 同上 | 20000/128/8192 | **19968**（forward 余 32） |
| A | KV-split | 16383/逻辑256/8192 | **16128** |
| B | 四把钥匙 | hash_bs=256 | 常规 8192→**31**、16384→**63**；tail 7936→**30**、16128→**62** |
| B | 测试锚点 | block_manager_test:788，hash_bs=128 | 8064/128−1=**62**（共 63 块） |
| B-Lifecycle | 块 77 引用轨迹 | prompt 8191 | 发布 ref=2→finish→1→B 命中→2→只 forward **127** |
| C | invalidate 传播 | 改 token 8，bs=4 | MTP stale=7→首脏块 **1**；TEXT stale=8→首脏块 2 |
| C | 桶快照 | bs=4，[t0..t5,d6,d7] | MTP 1 格（裁草稿）/TEXT 2 格（T1 掺草稿）；覆写位置 6 后 MTP 不变、TEXT 裁 T1 |
| D | KV-split 调度分块 | 8191/逻辑256 | **{7936,255}**；16383→**{8192,7936,255}**；tail 键=block_hashes[30] |
| D′ | 固化 stride 竞态 | manager stride=512，单例改 8192 | cap(384,192) 仍 128u（读单例会跨边界） |
| E | probe 命中竞争 | bs=4/stride=8/prompt=13，tail 不匹配 | 常规命中 8、回退 safe_count=2 有效、游标不清零；纯 tail 对照（:763）清零+set(4) |
| E′ | MTP 拒绝共享 | 33 全 7，fork[16]=8 / fork[32]=8 | **0 命中 / 16 token 恰一块** |
| F | 多 chunk tail 预取 | stride=8/kv_bs=4/prompt=13 | tail=12 跨检查点 1；CHECK 12/8==12/8 同区间；短 prompt 例 D：5000→regular=0 但 target=4992 |
| F-probe | probe 全程 | prompt=20000 | 常规命中 16384（hashes[63]/[127]）+tail 19968（hashes[155]）→trim 选 tail→forward 32 |
| G | D2H 门禁 | 2 worker | finish_worker(0) 后仍 miss→(1) 后可见（tokens=12/块=destination_id） |
| 引用六行表 | D2H 计数 | #201 配 #501 | t0 ref 2/1→t4a 成功 1/1（各缓存持）→t4b 失败 1/0 |
| 游标虚高 | k=3 假账 | tail 8064 挂载游标 3 | 账面 24576 vs 实际 8064，**错位 16448** |
| 双账时间线 | 20000 成长期 | 检查点 1、2 在缓存 | T2 发布 tail 后**（常规 3，tail 19968）并存**（三个物理源各记各的） |
| LRU 链 | 4 节点 | [N1,N2,N3,N4] | probe N1/N2→[N3,N4,N1,N2]，trim 后纹丝不动 |
| 三层全流程 | 20000 冷缓存 | Store 存到 19968 | base=0→target=16384→2 单元→final 16384→H2D 128 块→forward 3616；**下一请求 tail 命中取货单空只 forward 32** |
| 三方复用深度 | B_match/B2 | A 33 token | xLLM 2 块/1 块；vLLM 1 块/1 块（无条件税）；SGLang 全程/32 token [推演] |
| 四行 A 表 | 键策略作废 | bs=16 改 t_16 | MTP 2 格全废/TEXT 1 格/LINEAR 检查点 0 保住/LINEAR 若用 MTP 键 2 格纯自残 |



## §十四 来源与处置（覆盖率表 + 通读声明）

| 书 | 章节 | 笔记承接 | 裁决 |
|---|---|---|---|
| r1 卷首/Ch1 意图与两把尺子 | §一/§二/§三 | 进正文 | |
| r1 Ch2 数据结构与键空间 | §三/§四 | 进正文（旧独立链史→§十） | |
| r1 Ch3 设备侧发布挂载闭环 | §五 | 进正文 | |
| r1 Ch4 跨层调度验证 | §六 | 进正文（prefetch 细节由 r2 Ch5/r3 Ch5 增强） | |
| r1 卷末意图覆盖/勘误注 | §一/§十一 | 合并 | |
| r2 Ch1 边界数学与 hash 统一 | §二/§三 | 合并进 r1 对应节（失败传播链+粒度前提） | |
| r2 Ch2 标记与块生老病死 | §四 | 进正文（B-Lifecycle+块 77/78/79） | |
| r2 Ch3 键的制造维护 | §三 | 合并（三连调用/lookahead 数学/盖键） | |
| r2 Ch4 发布准入账本 | §五 | 进正文（游标 0/1/3 推演+probe 两问+F-probe） | |
| r2 Ch5 prefetch 零基础重讲 | §六 | 进正文（八变量+例 D/E/F+边界场景） | |
| r2 附录 D/E | §十一 | 合并 | |
| r3 Ch1 函数实现/截尾/哨兵 | §二/§五 | 进正文（四路径+三重兜底+哨兵链） | |
| r3 Ch2 账本与 erase 全删 | §四 | 进正文（三步流水线四实例） | |
| r3 Ch3 hash 域与 lookahead | §三 | 合并（成员变量评价/桶快照手算） | |
| r3 Ch4 双账本与 LRU | §四 | 进正文（读写点矩阵+固定 prompt 证明+4 节点链） | |
| r3 Ch5 三层全景 | §六 | 进正文（全景+六行引用表+取货单粒度澄清） | |
| r3 附录 D/E/F | §七/§十一 | 合并 | |
| r4 Ch1 MTP 块物理内容 | §三 | 进正文（复合材料五步证据链+两域匹配表+六环失败链） | |
| r4 Ch2 vLLM/SGLang | §八 | 进正文 | |
| r4 Ch3 LINEAR 剥域+元问题 | §三/§七 | 进正文（四行 A 表+三分账+五步审计框架） | |
| r4 附录 C/D/E | §七/§十一 | 合并（6 项 [待验证] 收进对应节） | |
| findings.md / unverified.md | §九 | 进正文 | |
| 会话讨论（论文考证/PR 考古/发布过程） | §九/§十 | 进正文 | |

**通读声明**：四本书经子代理逐本全文通读（1-2487 / 1-3968 / 1-2902 / 1-1262，分段 read 累计=wc -l，无抽样），摘要保留全部手算数字/file:line/裁决；findings.md 与 unverified.md 由本人通读（发布阶段）。初稿 156 行触发 <5% 自查，回炉扩充手算全程与失败画面至本版。**遗留 [待验证]**（均不承重，正文就地标注）：dp_rank0 stride 同构前提；commit 声称测试数；SGLang 8192 网格与树插入时机/page 粒度；vLLM partial tail 与 alignment_tokens 取值；撞键掉速幅度；vLLM/SGLang 复用深度推演数字；兄弟分支废弃状态。**工作区四目录已删**（含四本 book.md 与全部中间产物——内容已被本笔记吸收，书是脚手架不归档）。
