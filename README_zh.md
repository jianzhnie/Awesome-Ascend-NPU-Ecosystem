# Awesome Ascend NPU Ecosystem — 昇腾 NPU 生态全景

昇腾（Ascend）NPU 生态在 2024-2026 年经历了爆发式增长。本文从硬件、官方仓库、社区仓库、训练/推理/应用框架、开发者工具六个维度，系统梳理 Ascend 生态全貌。

---

## 目录

- [硬件路线图](#硬件路线图)
- [官方仓库](#官方仓库)
  - [核心计算平台](#核心计算平台)
  - [训练套件](#训练套件)
  - [推理引擎](#推理引擎)
  - [算子与编译器](#算子与编译器)
  - [训练/推理配方](#训练推理配方)
  - [领域库与工具](#领域库与工具)
- [社区仓库](#社区仓库)
  - [推理引擎](#社区推理引擎)
  - [训练与强化学习](#训练与强化学习)
  - [内核与编译器](#内核与编译器)
  - [Agent 与应用框架](#agent-与应用框架)
  - [推理服务部署](#推理服务部署)
  - [量化压缩](#量化压缩)
  - [多模态](#多模态)
  - [llama.cpp / GGUF](#llamacpp--gguf)
  - [HuggingFace 生态](#huggingface-生态)
- [超大规模模型训练实践](#超大规模模型训练实践)
- [开发者工具链](#开发者工具链)
- [社区与学习资源](#社区与学习资源)
- [训练框架对比](#训练框架对比)
- [快速入门](#快速入门)

---

## 硬件路线图

| 芯片 | 状态 | 发布时间 | FP16 | FP8 | 显存 | 内存带宽 | 定位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ascend 910B** | 量产 | 2023 Q1 | ~320 TFLOPS | — | 64 GB | ~1.2 TB/s | 训练+推理 |
| **Ascend 910C** | 量产（当前主力） | 2025 Q1 | ~800 TFLOPS | — | 128 GB | ~3.2 TB/s | 训练+推理 |
| **Ascend 950PR** | 规划中 | 2026 Q1 | — | ~1 PFLOPS | 128 GB (HiBL 1.0) | ~1.6 TB/s | 推理（Prefill） |
| **Ascend 950DT** | 规划中 | 2026 Q4 | — | ~1 PFLOPS (FP4: ~2P) | 144 GB (HiZQ 2.0) | ~4.0 TB/s | 训练+推理（Decode） |
| **Ascend 960** | 规划中 | 2027 Q4 | — | — | — | — | 下一代 |
| **Ascend 970** | 规划中 | 2028 Q4 | — | — | — | — | 下一代 |

**关键架构变化**：950 系列引入 SIMD/SIMT 双模架构，离散内存访问效率提升 4×，向量计算比提升 30%，任务调度延迟降低 50%。与 NVIDIA 的竞争策略是：单卡性能落后，但通过超大集群互联（CloudMatrix 384 超节点、Atlas 950 SuperPoD 8192 卡）形成系统性优势。

---

## 官方仓库

华为昇腾官方代码仓库主要托管在 **[GitCode（AtomGit）](https://gitcode.com/Ascend)**。2025 年 9 月华为宣布 CANN 全面开源，开放 50+ 代码仓、800+ 算子。

### 核心计算平台

| 仓库 | 定位 |
| :--- | :--- |
| [CANN](https://gitcode.com/cann) | Ascend 计算架构，对标 NVIDIA CUDA |
| [torch_npu](https://gitcode.com/Ascend/pytorch) | PyTorch Ascend 后端插件（Ascend Extension for PyTorch） |
| [torchair](https://gitcode.com/Ascend/torchair) | PyTorch 图模式（`torch.compile`）编译后端 |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | 基于 torchtitan 的昇腾大模型全流程训练插件，即插即用 |
| [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | NVIDIA TransformerEngine 昇腾移植版，配合 Megatron-LM 使用 |

### 训练套件

| 仓库 | 定位 |
| :--- | :--- |
| [MindSpeed](https://gitcode.com/Ascend/MindSpeed) | 大模型加速核心库 |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | 大模型分布式训练套件（基于 Megatron-LM，支持 100+ 模型） |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | 多模态大模型训练加速（Wan2.1 / Qwen2.5-VL / InternVL3 / HunyuanVideo / GLM4.1V 等） |
| [MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL) | 强化学习训练框架（GRPO / DAPO / PPO / DPO） |
| [ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) | 昇腾官方模型仓库，提供预优化模型脚本（Flash Attention / 算子融合等） |

### 推理引擎

| 仓库 | 定位 |
| :--- | :--- |
| [MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM) | 华为自研 LLM 推理引擎（Continuous Batching / PageAttention / FlashDecoding） |
| MindIE-SD | 扩散模型推理引擎（MindIE 套件的一部分） |
| MindIE Turbo | 已并入 vLLM-Ascend，无需单独安装 |

### 算子与编译器

| 仓库 | 定位 |
| :--- | :--- |
| [ops-nn](https://gitcode.com/cann/ops-nn) | 神经网络高阶算子库（矩阵乘 / 激活 / 归一化 / 池化等） |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | Transformer 专用算子（FlashAttention / FusedAttention / GroupedMatmul） |
| [ascend-transformer-boost](https://gitcode.com/cann/ascend-transformer-boost) | ATB（Ascend Transformer Boost）— 融合算子 + 图算子 + 插件机制 |
| [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) | 基于 MLIR 的昇腾亲和算子编译中间表示 |
| ops-adv | 高级融合算子库，随 CANN 工具包分发（FlashAttention / GroupedMatmul 等） |

### 训练/推理配方

| 仓库 | 定位 |
| :--- | :--- |
| [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train) | 训练配方：大模型预训练 / SFT / 微调示例 |
| [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) | 推理配方：主流模型推理部署示例 |

### 领域库与工具

| 仓库 | 定位 |
| :--- | :--- |
| [fbgemm-ascend](https://gitcode.com/Ascend/fbgemm-ascend) | 推荐系统高性能 PyTorch NPU 算子库 |
| [vision](https://gitcode.com/Ascend/vision) | TorchVision 昇腾 NPU 适配 |
| [apex](https://gitcode.com/Ascend/apex) | NVIDIA Apex 混合精度训练的昇腾移植版 |
| [RecSDK](https://gitcode.com/Ascend/RecSDK) | 推荐系统 SDK，含 TorchRec NPU 适配 |
| [AMCT](https://gitcode.com/cann/amct) | 昇腾原生模型压缩工具包（AWQ / GPTQ / SmoothQuant / FlatQuant；INT8 / INT4 / MXFP8 / MXFP4 / HiF8） |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | 昇腾官方企业级 Agent 训推框架（v26.0），集成 MindSpeed-RL GRPO 训练 |

### 推荐软件栈版本

| 组件 | 推荐版本 |
| :--- | :--- |
| CANN | 8.5.0 / 9.0.0 / 9.1.0 |
| PyTorch | 2.7.1 / 2.8.0 / 2.9.0 |
| torch_npu | 2.8.0 / 2.9.0 |
| Python | 3.10 / 3.11 |
| Megatron-LM (Mcore) | core_r0.12.1 |

---

## 社区仓库

### 社区推理引擎

| 仓库 | 定位 |
| :--- | :--- |
| [vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 昇腾推理的事实标准入口 — vLLM 官方社区维护插件，支持 DeepSeek-V3/V4 / Qwen3 / GLM5 / Kimi-K2 等 |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能 LLM Serving 引擎，2025 年 7 月起原生支持 Ascend；擅长结构化生成和复杂前端 |
| [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | SGLang Ascend 内核库：DeepEP-Ascend / MLA / GQA / FlashAttention / SwiGLU / Mamba |
| [openfuyao-vllm-ascend](https://github.com/qsunnyy/openfuyao-vllm-ascend) | vLLM-Ascend 社区维护分支 |

### 训练与强化学习

| 仓库 | 定位 |
| :--- | :--- |
| [verl](https://github.com/volcengine/verl) | 火山引擎开源 RL 训练框架，2026 Q1-Q2 将 Ascend 作为一等公民支持 |
| [OpenRLHF](https://github.com/jianzhnie/OpenRLHF) | 开源 RLHF 框架 Ascend 社区适配版（PPO / GRPO / DPO / KTO / REINFORCE++） |
| [slime-ascend](https://gitcode.com/Ascend/slime-ascend) | 智谱 AI Post-Training / RL 框架昇腾版（Megatron + SGLang），支持 GLM-5 / Qwen3 / DeepSeek V3/R1 / Llama 3 |

### 内核与编译器

| 仓库 | 定位 |
| :--- | :--- |
| [triton-ascend](https://github.com/triton-lang/triton-ascend) | Triton 语言昇腾后端 — 支持 600+ Triton 算子，100% 接口兼容 |
| [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | Tile 级编程框架昇腾适配 — 300+ TileLang 算子，Beginner / Developer / Expert 三种模式 |

### Agent 与应用框架

| 框架 | 定位 | Ascend 状态 |
| :--- | :--- | :--- |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | 昇腾官方企业级 Agent 训推框架 | ✅ 原生（v26.0） |
| [LangChain](https://python.langchain.ac.cn/docs/integrations/providers/ascend/) | LLM 应用开发框架 | ✅ 官方 `AscendEmbeddings` 集成 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent 图编排框架 | ✅ DeepSeek + 昇腾验证通过 |
| [Dify](https://github.com/langgenius/dify) | LLM 应用开发平台 | ✅ 原生 `langgenius/dify:latest-gpu-cn` 镜像 |
| [FastChat](https://github.com/lm-sys/FastChat) | 对话服务框架 | ⚠️ 需修改 `model_adapter.py` 适配多卡 NPU |

### 推理服务部署

| 框架 | 定位 | Ascend 状态 |
| :--- | :--- | :--- |
| [Xinference](https://github.com/xorbitsai/inference) | 异构算力推理框架 | ✅ Transformers 后端；企业版支持 MindIE |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA 推理服务器 | ⚠️ 昇腾社区有部署参考 |
| [KServe](https://github.com/kserve/kserve) | K8s 原生模型服务 | ✅ 通过 vLLM / SGLang 运行时支持 |

### 量化压缩

| 框架 | 定位 | 支持格式 |
| :--- | :--- | :--- |
| [AMCT](https://gitcode.com/cann/amct) | 昇腾原生模型压缩工具包 | INT8 / INT4 / MXFP8 / MXFP4 / HiF8（AWQ / GPTQ / SmoothQuant / FlatQuant） |
| msModelSlim | MindStudio 量化压缩组件 | W8A8 / W4A8 / W8A16 / W4A4（LLM + 多模态 + 扩散模型） |
| SGLang Quant | 内置于 [SGLang](https://github.com/sgl-project/sglang) | AWQ / GPTQ / AutoRound / W4A16 / W8A16 |

> **推荐**：AWQ（W4A16）是 Ascend ≥7B 模型的默认量化方案——几乎无损。GPTQ 次之。W4A4 在长上下文推理中存在逻辑退化风险。

### 多模态

| 框架 | 支持模型 | 关键优化 |
| :--- | :--- | :--- |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | Qwen2-VL / InternVL2/2.5 / GLM-4.1V / LLaVA / CogVideoX / HunyuanVideo / Wan2.1 / SD3.5/SDXL / Flux | DiTCache 1.6× 视频推理加速；Ulysses 并行 3.75×；融合 RoPE 算子 +15% |
| vLLM-Ascend（多模态） | InternVL2 / Qwen-VL / Qwen2-VL / LLaVA / MiniCPM | OpenAI 兼容 Vision API |
| [mllm](https://github.com/UbiquitousLearning/mllm) | Qwen3-VL / Qwen2-VL（W8A8） | 端侧 Ascend NPU 推理（2026.05 新增 Ascend 后端） |

### llama.cpp / GGUF

| 项目 | 定位 |
| :--- | :--- |
| [BitCPM-CANN](https://huggingface.co/openbmb/BitCPM-CANN-0.5B-gguf) | 首个昇腾原生 1.58-bit 三元量化模型（OpenBMB，2026）— 推理内存节省 6×，性能保留 ≥95.7% |
| llama.cpp CANN | 社区适配进行中（[ggml-org/llama.cpp#11698](https://github.com/ggml-org/llama.cpp/discussions/11698)）— GGUF + CANN 后端探索 |

### HuggingFace 生态

| 组件 | Ascend 支持 | 最低版本 |
| :--- | :--- | :--- |
| [transformers](https://github.com/huggingface/transformers) | ✅ 通过 `torch_npu` 自动适配 | ≥ 4.32.0 |
| [accelerate](https://github.com/huggingface/accelerate) | ✅ 通过 `torch_npu` 适配 | ≥ 0.22.0 |
| [peft](https://github.com/huggingface/peft) | ✅ LoRA / Prefix Tuning 可用 | ≥ 0.5.0 |
| [trl](https://github.com/huggingface/trl) | ✅ 可用 | ≥ 0.5.0 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ✅ 通过 `torch_npu` 适配 | 最新版 |

---

## 超大规模模型训练实践

### 华为盘古系列（昇腾原生）

| 模型 | 参数量 | 训练集群 | MFU / 关键指标 |
| :--- | :--- | :--- | :--- |
| **盘古 Ultra（Pangu Ultra）** | 135B Dense | 8192 张昇腾 NPU | MFU **52%**；13.2T tokens |
| **盘古 Ultra MoE** | 718B（激活 39B） | 万卡级 CloudMatrix 384 | MFU 初始 18.9% → 6K 卡 **30%** → 万卡 **41%**；256 路由专家，动态激活 8 专家 |
| **盘古 Pro MoE** | 72B（激活 16B） | 4000 颗昇腾 NPU | 13T tokens（2025 年 6 月开源） |
| **盘古 Embedded** | 7B | 昇腾 NPU | 端侧小模型，已开源 |

训练技术创新点：DSSN（Depth-Scaled Sandwich-Norm）稳定架构、TinyInit 小初始化、EP-Group 负载均衡、MC² 通算融合、NFA 亲和融合算子。

### DeepSeek 系列

| 模型 | Ascend 训练情况 |
| :--- | :--- |
| **DeepSeek-V3** | 官方使用 2048 H800 训练；后续在昇腾 910B 上完成适配与训练验证 |
| **DeepSeek-V4** | **首个公开宣称与昇腾完成全栈适配的万亿参数 frontier 模型**（V4-Pro 1.6T，V4-Flash 284B）；基于 910B/910C 集群完成推理适配与全参数后训练（SFT+RLHF），MFU 达 34.9% |
| **DeepSeek-R1** | 基于 H800 集群训练；昇腾侧主要完成推理适配 |

### 其他厂商

| 厂商 / 模型 | Ascend 训练情况 | 关键信息 |
| :--- | :--- | :--- |
| **美团 LongCat-2.0** | 5-6 万张国产加速卡完成预训练 | 万亿参数级（1T+），1M 上下文 |
| **讯飞星火 V4.0 / X2** | "飞星一号/二号"（昇腾万卡集群） | 训练效率从 A100 30-50% → 85-95% |
| **百度 ERNIE 4.5** | 300B MoE（A47B）；PaddlePaddle 支持昇腾 | H800 MFU 47%（昇腾适配版推进中） |
| **智谱 GLM-4.5** | 355B MoE；slime 框架支持昇腾 | 23T tokens |
| **智谱 GLM-Image** | **首个基于昇腾+MindSpore 全程训练的 SOTA 多模态模型** | 全流程国产 |
| **阿里 Qwen 系列** | ModelArts 官方支持昇腾训练 | SFT / LoRA / GRPO 全流程 |

### 训练实践总结

| 维度 | 现状 |
| :--- | :--- |
| **可用规模** | 千卡→万卡级，可训百亿→准万亿参数模型 |
| **稠密 MFU** | 盘古 Ultra 达 52%（对标 H100 45-55%） |
| **MoE MFU** | 盘古 Ultra MoE 达 41%（万卡），All-to-All 通信是主要瓶颈 |
| **与 NVIDIA 差距** | 单卡 BF16 算力：910B 约 H100 的 32%，910C 约 80%；集群训练效率约为 H100/H800 的 35%-70% |
| **关键挑战** | 超长稳训练稳定性、MoE All-to-All 通信、FP8 极致优化、大规模 RL 训练 |

---

## 开发者工具链

### MindStudio（2025 年底全量开源）

华为昇腾全栈 IDE，围绕三大工具链 + 通用层构建：

| 工具链 | 组件 | 用途 |
| :--- | :--- | :--- |
| **msTT（训练）** | msProf / msprof-analyze / msMemScope / msProbe | 性能采集、自动分析（advisor / compare / cluster）、内存诊断、精度调试 |
| **msIT（推理）** | msServiceProfiler / msModelSlim / msPrechecker | 推理服务 profiling（支持 vLLM / SGLang / MindIE）、量化压缩、环境预检 |
| **msOT（算子）** | msKPP / msOpGen / msProf / msDebug / msSanitizer | 性能预测建模、算子生成、算子调优、调试、异常检测 |
| **通用** | msInsight / msMonitor / msPTI | 可视化分析（时间线/算子/内存/通信）、在线集群监控、Profiling API |

### MindStudio Insight（原名 Ascend Insight）

可视化性能调优工具，支持百卡/千卡集群分析，可处理 20 GB 集群性能文件。

核心视图：**Timeline**（流水线 / Overlap / 通信计算覆盖）、**Operator**（算子耗时统计）、**Memory**（内存使用折线图）、**Communication**（全网链路 / 慢节点分析）、**Jupyter**（在线编辑）

### CANNBot Agent

AI 辅助编程工具，支持 Claude Code / Cursor / Trae 等，集成算子开发、精度调试、性能分析等能力。单个算子开发效率提升 5×+。

---

## 社区与学习资源

| 资源 | 说明 | 链接 |
| :--- | :--- | :--- |
| **Ascend（GitCode）** | 官方仓库合集 | [gitcode.com/Ascend](https://gitcode.com/Ascend) |
| **CANN（GitCode）** | CANN 核心开源社区主阵地，含 Issue / PR / 贡献 | [gitcode.com/cann](https://gitcode.com/cann) |
| **Ascend（Gitee）** | 部分官方仓库镜像 | [gitee.com/ascend](https://gitee.com/ascend) |
| **昇腾社区** | 官方文档、SDK 下载、技术文章、活动 | [hiascend.com](https://www.hiascend.com) |
| **cann-learning-hub** | 四级学习路径（入门→进阶→高级→专家） | [gitcode.com/cann/cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) |
| **cann-community** | CANN 社区治理与 SIG 协作 | [gitcode.com/cann/community](https://gitcode.com/cann/community) |
| **HiDevLab** | 免费申请昇腾 NPU 算力（初始 100 卡时） | [hiascend.com](https://www.hiascend.com) |
| **Docker 基础镜像** | CANN 预装镜像 | `quay.io/ascend/cann:8.5.0-a3-openeuler24.03-py3.11` |

---

## 训练框架对比

| 框架 | 并行策略 | 适用场景 | 昇腾适配入口 |
| :--- | :--- | :--- | :--- |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | FSDP2 / TP / CP / PP / EP | PyTorch 原生训练，DeepSeek-V4 / Qwen3 续训练 | `pip install -e .`，即插即用 |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | TP / PP / DP / EP / CP + ZeRO | 100+ 模型端到端训练，Megatron 生态 | `import mindspeed.megatron_adaptor` 一行适配 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO-1/2/3/Infinity + Ulysses SP | ZeRO-Infinity CPU/NVMe 卸载、超长序列（1M+） | 通过 `torch_npu` 原生适配 |
| PyTorch FSDP2 | FSDP2 + Hybrid Shard | 中小模型（<70B），社区标准路径 | `torch_npu` 已原生支持 |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | PP / TP / VPP / CP / Encoder-DP | 多模态大模型训练（视频/图像/VL） | 训练性能比通用方案快 **19-24%** |

### torchtitan-npu 性能基准（Atlas 800T A3，64 卡）

| 模型 | 精度 | 并行策略 | 吞吐 (tokens/p/s) | MFU |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V4-Flash | BF16 | FSDP128+EP128 | 1056 | 27.67% |
| DeepSeek-V3-671B | BF16 | FSDP32+TP4+EP128 | 546 | — |
| DeepSeek-V3-671B + compile(AutoFuse) | BF16 | FSDP32+TP4+EP128 | 576 | — |

---

## 快速入门

```bash
# 1. 安装 PyTorch Ascend 后端
pip install torch_npu

# 2. 拉取 MindSpeed 加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && pip install -e .

# 3. 拉取 MindSpeed-LLM 训练套件
git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM && pip install -e .

# 4. 安装 vLLM-Ascend 推理引擎
pip install vllm-ascend

# 5. 参考官方配方
git clone https://gitcode.com/cann/cann-recipes-infer.git   # 推理示例
git clone https://gitcode.com/cann/cann-recipes-train.git   # 训练示例
```

---

## 量化选型建议

| 场景 | 推荐方案 | 说明 |
| :--- | :--- | :--- |
| 推理（≥7B 模型） | AWQ（W4A16） | 几乎无损，Ascend 默认推荐 |
| 推理（<7B 模型） | GPTQ（W4A16）或 W8A8 | 小模型对量化更敏感 |
| 训练 | BF16 + AMCT 感知量化 | QAT 优于 PTQ |
| 多模态 | msModelSlim（W8A8） | 原生支持 Qwen-VL / InternVL / Flux |
| 扩散模型 | msModelSlim（W8A8）+ MindIE-SD | DiTCache 进一步加速 |

---

## Contributing

欢迎提交 PR，补充昇腾 NPU 相关的开源项目、框架和学习资源。
