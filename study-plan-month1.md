# AI 基础月度计划 — Month 1:从梯度到 kernel(4 周)

> 节奏:工作日 = 模块 A(30~40min 概念/数学,通勤午休可做)+ 模块 B(60~75min 动手代码,晚上);
> 休息日 = ≤3h 连续项目(把本周碎片拼成完整作品)。
> 弹性总则:**宁可砍 A 模块,不砍 B 动手;周末作品是硬产出,其余可顺延。**

## 每周产出物(完整跑一遍的作品)

- W1:mini-torch(张量级 autograd)+ service-bed-push 全链路架构图
- W2:mini-GPT(能生成)+ KV cache 推理(带加速比实测)+ DIN toy
- W3:mini 推理引擎(并发 batch)与 vLLM 对拍 + top-10 kernel 报告 + 双塔 ONNX 化
- W4:LoRA SFT 对话模型 + 第一个自制 Triton kernel + 30 题面试卡

---

## Week 1:梯度从零(autograd)+ 业务全图

- [ ] **D1-A** 反向传播 = VJP;手推 sigmoid、softmax+CE 梯度。AI 出 5 题验证。
- [ ] **D1-B** micrograd:`Value` 类(__add__/__mul__/__neg__/tanh),手工 `.backward()`,数值梯度验证。
- [ ] **D1-业务(30min)** service-bed-push 目录走读:asb/ams(特征+TF)、asb/rta(召回排序线程)、asb/are(服务引擎)、asb/common(特征公共件),画骨架图 v0。
- [ ] **D2-A** 计算图与拓扑排序 backward;梯度为什么是累加不是覆盖(共享参数)。
- [ ] **D2-B** micrograd 补全 MLP+SGD;make_moons 训练;LR=0.01/0.1/1.0 三条曲线。
- [ ] **D2-业务** 特征入口:`common/module/feature_generator_module.h`、`ams/tensorflow_feature_generator_module.h`;特征从哪来(请求/画像 dict/track)。
- [ ] **D3-A** 数值梯度与 gradcheck;初始化(Xavier/Kaiming);sigmoid 深层为什么死。
- [ ] **D3-B** 加 gradcheck;激活换 ReLU(分段函数 backward)再训。
- [ ] **D3-业务** 特征盘点:`ams/functor/feature_column.h` 的 Sparse/Dense;特征字典(`common/dict/`);**FeatureGraph + pruning 是什么、为什么裁剪**。
- [ ] **D4-A** LR→FM:二阶交叉参数爆炸;FM=embedding+内积。
- [ ] **D4-B** toy CTR 数据(自造 1000 条):LR vs FM,对比 AUC。
- [ ] **D4-业务** track 回流:`track_log.proto`/`ams_track_log.proto`;曝光点击怎么 join 成样本;写 3 个穿越风险点。
- [ ] **D5-A** 召回/粗排/精排/重排分工(候选量/模型/时延三栏表);双塔 DSSM 一页纸。
- [ ] **D5-B** 双塔 toy(mini-torch):in-batch negative + temperature;穷举 ANN top-k。
- [ ] **D5-业务** 多路召回分类:`are/ards/recall/` 的 Recaller 家族(i2i/x2i/d2i/aps/sem_id),每路一句话。
- [ ] **D6-A** Wide&Deep/DeepFM(记忆 vs 泛化);MMoE 动机(shared-bottom 负迁移)。
- [ ] **D6-B** 双塔改精排式小模型(特征+交叉→MLP→pCTR),对比 AUC。
- [ ] **D6-业务** 排序侧:`are/server/rank/`(Ranker/SortRanker/ExpRanker)+ `rta/module/rank_module.h`;粗精排边界与特征来源。
- [ ] **D7(休息日 ≤3h)** mini-torch:Tensor 类支持 +,-,*,matmul,broadcast 前向反向(matmul 三个 backward 公式 + unbroadcast)→ 重训 D6 精排 toy + 全绿 gradcheck → 架构图 v2(每环节标真实类名)→ 笔记《从 VJP 到 mini-torch》。

**W1 验收**:①不查资料手推 softmax+CE;②mini-torch 训练收敛;③架构图能讲 10 分钟;④说出 3 个特征穿越风险点及代码位置。

---

## Week 2:结构从零(transformer)+ 搜推模型族谱

- [ ] **D8-A** attention 手推:QKᵀ/√d;为什么除 √d(E[q·k] 方差推一遍);causal mask 的 -inf。
- [ ] **D8-B** mini-torch 单头 attention(无 mask),与 torch SDPA 对拍 allclose。
- [ ] **D8-业务(20min)** 把消息推荐抽象成"用户表示×物品表示":x2i/i2i 本质是查表双塔。
- [ ] **D9-A** 多头:维度账与参数量;多头=子空间。
- [ ] **D9-B** multi-head + causal mask;故意去 mask 观察 loss 反常(理解机制)。
- [ ] **D10-A** 位置编码三代:正弦/可学习/RoPE(复数旋转,相对位置=角差)。
- [ ] **D10-B** 组装 Block(pre-LN/GELU/dropout),定义 ~1M 参数模型。
- [ ] **D11-A** tokenizer:BPE 原理;中文切分。
- [ ] **D11-B** tiktoken/minbpe + 10MB 语料 + 随机窗口 Dataset。
- [ ] **D12-A** GPT 全景:初始化(残差投影缩放)、weight decay 只加 2D 参数。
- [ ] **D12-B** nanoGPT-lite 训练循环:先过拟合 1 个 batch(loss→0),再全量小训。
- [ ] **D13-A** 采样:temperature/top-k/top-p;KV cache 需求预演。
- [ ] **D13-B** 训到能生成;存 checkpoint + 生成样例。
- [ ] **D14(休息日 ≤3h)** KV cache 推理(prefill+decode 分离)→ 无 cache vs 有 cache 吞吐实测与解释 → 搜推 toy:DIN target-attention(复用 W2 attention 模块)→ 笔记《从 attention 到我的 GPT》。

**W2 验收**:①白板默写 attention;②mini-GPT 生成通顺样本;③KV cache 加速比有数字有解释;④讲清 DIN 与双塔异同。

---

## Week 3:推理从零(mini engine → vLLM)+ 静态图 + brpc 服务

- [ ] **D15-A** KV 显存账:代入 W2 的 GPT 算 KB/token;反推可容纳并发;decode 为什么 memory-bound。
- [ ] **D15-B** decode batch 1/4/16 吞吐曲线(用 W2 的 GPT),解释拐点。
- [ ] **D16-A** PagedAttention 精读:外碎片/内部碎片/写时复制/block table。
- [ ] **D16-B** mini-engine v0:单请求 prefill+decode + 整块预分配 KV。
- [ ] **D17-A** Orca continuous batching:iteration-level scheduling。
- [ ] **D17-B** mini-engine v1:3 并发请求、完成即退、新请求即进(最简调度器)。
- [ ] **D18-A** vLLM V1 源码地图:EngineCore→Scheduler→KVCacheManager→GPUModelRunner;一个请求的生命周期。
- [ ] **D18-B** vLLM 跑同规模模型,抓 TTFT/TPOT/gpu_cache_usage/preemption,与 mini-engine 对照。
- [ ] **D19-A** profiler:torch.profiler + chrome trace;Nsight Compute 三指标(sm/dram throughput)。
- [ ] **D19-B** H800 上 profile vLLM prefill/decode 各一次 → **top-10 kernel 表**(名称/占比/shape/bound 判断)= 毕业作。
- [ ] **D20-A** 静态图优化:常量折叠/融合/layout/量化在图层的表达;ONNX Runtime 的图优化 pass。
- [ ] **D20-B** W1 双塔/精排 toy 导 ONNX → onnxruntime vs torch 延迟对比 → 尝试 fp16。
- [ ] **D20/21-业务** brpc 服务精读:`BaseSyncModule`/`BaseAsyncRpcModule` 线程模型;列出 service-bed-push 为高 QPS 做的 5 件事并找代码证据(无锁结构/对象池/批处理/超时降级/feature_graph_pruning)。
- [ ] **D21(休息日 ≤3h)** mini-engine 与 vLLM 输出对拍(温度 0)→ 一次 push 请求端到端时序图 + 各步时延量级 → 笔记《mini engine 对拍 vLLM》+ brpc/静态图面试卡 10 题。

**W3 验收**:①KV 显存账会算;②mini-engine 并发跑通且对拍一致;③top-10 kernel 报告能判 bound;④画出 push 请求完整时序 + 说出 5 个高 QPS 工程点。

---

## Week 4:训练与算子从零(SFT/DDP/Triton)+ 面试收口

- [ ] **D22-A** fp32/fp16/bf16 位分配;loss scaling 为什么 bf16 不需要。
- [ ] **D22-B** mini-GPT 加 bf16 AMP:step 时间与 loss 对比。
- [ ] **D23-A** DDP=allreduce 梯度(与 PS 对照,呼应搜推);ZeRO 1/2/3 切分什么。
- [ ] **D23-B** 双卡 DDP 训 mini-GPT(H 卡):1 卡 vs 2 卡扩展效率。
- [ ] **D24-A** SFT 拆解:chat template/loss mask/packing。
- [ ] **D24-B** TRL LoRA SFT 0.5~1.5B 模型(200 条自造指令,目标是流程完整):训练→合并→推理→对话。
- [ ] **D25-A** 量化:W4A16/分组反量化/AWQ 思想;KV int8。
- [ ] **D25-B** AWQ 量化 D24 模型 → vLLM 加载 → 延迟/显存/质量对比。
- [ ] **D26-A** 算子优化七步闭环 SOP;Nsight Compute 读一个 memory-bound kernel 报告。
- [ ] **D26-B** 从 W3 top-10 表挑一个 memory-bound kernel,写优化假设(如 LayerNorm+残差三读一写可融合)。
- [ ] **D27-A** Triton 语法 40min(program_id/mask/block,与 CUDA SIMT 对照)。
- [ ] **D27-B** Triton 写 fused RMSNorm 或 fused softmax:allclose 对拍 + 加速比;进自己的 kernel 仓库。
- [ ] **D28(休息日 ≤3h)** 面试卡 30 题定稿(AI 扮演面试官过一遍,记录答不好的)→ 月度复盘博客《四周:从 autograd 到 Triton kernel》→ 写 Month 2 计划(vllm-omni 深入,目标一个可 PR 的优化)。

**W4 验收**:①LoRA 模型能对话;②DDP 扩展效率有数字;③自制 kernel 加速比>1 且 allclose;④30 题卡答对 80%。

---

## 面试卡 30 题(月末验收,附 service-bed-push 实证答案)

**搜推/业务(15)**
1. 一次 push 请求经过哪些模块?各步时延量级?
2. 召回为什么多路?i2i/x2i/d2i 适用场景与冷启动表现?
3. 双塔与 i2i 查表的关系?向量召回 ANN 索引怎么选?
4. 粗排/精排的模型、特征、时延预算差异?为何不直接全量精排?
5. 稀疏/稠密/序列/统计特征在代码里怎么组织?Sparse/Dense FeatureColumn 区别?
6. 特征穿越是什么?track→join 链路哪几步可能发生?怎么防?
7. FeatureGraph 与 pruning 解决什么(线上只算模型用到的特征子图)?
8. embedding 表线上加载与热更新?dict/seekable_table 的作用?
9. TF 静态图线上化流程?freeze 与图优化做了什么?
10. 模型更新时效性?特征与模型版本一致性怎么保证?
11. 高 QPS 工程:线程模型/对象池/批处理/超时降级/监控,各举代码证据?
12. A/B 分流怎么做?experiment.proto 的角色?
13. pCTR 为什么排序好也要校准?怎么校准?
14. 多目标(点击/打开/负反馈)怎么融合?MMoE 与加权的取舍?
15. 若把召回升级为向量双塔,要动哪些模块?

**深度学习基础(8)**
16. 手推 softmax+CE 梯度。 17. attention 为什么除 √d。 18. LayerNorm vs BatchNorm。
19. 残差为什么有效。 20. KV cache 显存公式并算例。 21. PagedAttention 解决什么。
22. LoRA 原理与秩的取舍。 23. W4A16 反量化过程。

**推理系统(7)**
24. prefill 与 decode 区别。 25. continuous batching 是什么。 26. decode 为什么 memory-bound。
27. prefix cache 命中条件。 28. 量化提升吞吐的路径。 29. 怎么判断一个 kernel 是 compute/memory bound。
30. 讲一遍 vLLM 中一个请求的生命周期。

---

## AI 使用军规(贯穿全月)

1. 每个 A 模块后让 AI 出 3~5 道题(费曼检验),答错回到材料。
2. **对拍纪律**:自己先写,再让 AI 出参考实现 diff;顺序不能反。
3. 卡住 >20min 才问,且要求"只给提示不给答案"。
4. 休息日:AI 当面试官/出题人。
5. 笔记的机制描述必须自己写,AI 只润色。

## 资源清单(够用就好,不贪多)

- W1:Karpathy micrograd(代码+视频);王喆《深度学习推荐系统》第 1~4 章
- W2:nanoGPT;minbpe;The Illustrated Transformer;DIN 论文(只抓动机)
- W3:PagedAttention(SOSP'23);Orca(OSDI'22);vLLM 文档+源码;onnxruntime 文档;brpc 官方文档
- W4:TRL 文档;Triton 官方 tutorial;Karpathy《Let's reproduce GPT-2》选看

## Month 2 / Month 3 展望(月末细化)

- **M2(推理深入)**:vllm-omni 多模态输入路径、mm encoder cache、chunked mm prefill;diffusion worker 的 attention 后端族与缓存族 → 产出:一个可 PR 的优化 + 一次 50% 归因报告。
- **M3(训练与算子深入)**:DPO/GRPO 数学与 verl 结构;FlashAttention v2 读码;Triton 进阶(flash attention 简化版);搜推侧:生成式推荐(HSTU/OneRec)与 LLM 的汇合点。
