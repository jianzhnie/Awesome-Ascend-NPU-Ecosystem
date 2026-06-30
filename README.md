# Awesome Ascend NPU Ecosystem — 昇腾 NPU 生态全景

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

> 一份精心整理的昇腾（Ascend）NPU 生态全景图：从芯片、算子、训练/推理框架到 Agent、量化、多模态应用，覆盖开发者从入门到生产所需的核心资源。
>
> 🌐 [English Version](./README_en.md)

华为昇腾 NPU 生态在 2024–2026 年间经历了爆发式增长。本文从**硬件路线图、官方仓库、社区仓库、大规模训练实践、开发者工具、学习资源**六个维度，系统梳理 Ascend 开源生态全貌，并给出不同场景下的选型建议。

**适用读者**：大模型训练/推理工程师、国产化算力架构师、AI Infra 从业者、希望将业务从 CUDA 迁移到 Ascend 的开发者。

---

## 目录

- [昇腾生态分层速览](#昇腾生态分层速览)
- [硬件路线图](#硬件路线图)
- [官方仓库](#官方仓库)
  - [核心计算平台（必读）](#核心计算平台必读)
  - [训练与微调套件](#训练与微调套件)
  - [推理引擎](#推理引擎)
  - [算子与编译器](#算子与编译器)
  - [训练/推理配方（Recipes）](#训练推理配方recipes)
  - [领域库与工具](#领域库与工具)
- [社区仓库](#社区仓库)
  - [推理引擎](#社区推理引擎)
  - [训练与强化学习](#社区训练与强化学习)
  - [算子与编译器](#社区算子与编译器)
  - [Agent 与应用框架](#agent-与应用框架)
  - [推理服务与部署](#推理服务与部署)
  - [量化压缩](#量化压缩)
  - [多模态](#多模态)
  - [llama.cpp / GGUF](#llamacpp--gguf)
  - [HuggingFace 生态](#huggingface-生态)
- [主流模型 Ascend 适配矩阵](#主流模型-ascend-适配矩阵)
- [超大规模模型训练实践](#超大规模模型训练实践)
- [开发者工具链](#开发者工具链)
- [社区与学习资源](#社区与学习资源)
- [场景化选型指南](#场景化选型指南)
- [训练框架深度对比](#训练框架深度对比)
- [快速入门](#快速入门)
- [常见问题 FAQ](#常见问题-faq)
- [Contributing](#contributing)

---

## 昇腾生态分层速览

```text
┌─────────────────────────────────────────────────────────────┐
│  应用层：Dify / LangChain / AgentSDK / FastChat / RAG         │
├─────────────────────────────────────────────────────────────┤
│  服务层：vLLM-Ascend / SGLang / MindIE / Xinference / KServe  │
├─────────────────────────────────────────────────────────────┤
│  框架层：MindSpeed-LLM / MindSpeed-MM / torchtitan-npu / verl │
├─────────────────────────────────────────────────────────────┤
│  加速层：torch_npu / torchair / ops-transformer / Triton      │
├─────────────────────────────────────────────────────────────┤
│  底座层：CANN / AscendNPU-IR / Ascend C / 驱动 / 固件          │
├─────────────────────────────────────────────────────────────┤
│  硬件层：910B / 910C / 950PR / 950DT / 超节点 / SuperPoD      │
└─────────────────────────────────────────────────────────────┘
```

**核心结论**：Ascend 生态已经覆盖从底层芯片到上层应用的全链路，但**软件优化空间大于硬件单卡差距**——集群级调优和原生 Ascend 优化栈（MindSpeed、vLLM-Ascend、SGLang）是释放性能的关键。

---

## 硬件路线图

| 芯片 | 状态 | 发布时间 | FP16 | FP8 | 显存 | 内存带宽 | 定位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ascend 910B** | 量产 | 2023 Q1 | ~320 TFLOPS | — | 64 GB | ~1.2 TB/s | 训练 + 推理 |
| **Ascend 910C** | 量产（当前主力） | 2025 Q1 | ~800 TFLOPS | — | 128 GB | ~3.2 TB/s | 训练 + 推理 |
| **Ascend 950PR** | 规划中 | 2026 Q1 | — | ~1 PFLOPS | 128 GB (HiBL 1.0) | ~1.6 TB/s | 推理（Prefill） |
| **Ascend 950DT** | 规划中 | 2026 Q4 | — | ~1 PFLOPS (FP4: ~2P) | 144 GB (HiZQ 2.0) | ~4.0 TB/s | 训练 + 推理（Decode） |
| **Ascend 960** | 规划中 | 2027 Q4 | — | — | — | — | 下一代 |
| **Ascend 970** | 规划中 | 2028 Q4 | — | — | — | — | 下一代 |

### 与 NVIDIA 的对比

| 对比维度 | Ascend 910B | Ascend 910C | NVIDIA H100 |
| :--- | :--- | :--- | :--- |
| FP16 算力 | ~320 TFLOPS | ~800 TFLOPS | ~989 TFLOPS |
| 显存 | 64 GB | 128 GB | 80 GB |
| 内存带宽 | ~1.2 TB/s | ~3.2 TB/s | ~3.35 TB/s |
| 单卡 BF16 算力比 H100 | ~32% | ~80% | 100% |

**关键趋势**：
- **950 系列**引入 SIMD/SIMT 双模架构，离散内存访问效率提升 **4×**，向量计算比提升 **30%**，任务调度延迟降低 **50%**。
- 华为的竞争策略不是单卡追赶，而是**集群级超越**：CloudMatrix 384 超节点、Atlas 950 SuperPoD（8192 卡）通过超高带宽互联形成系统性优势。

> **注意**：910B 有 B1/B2/B3 等多个版本，带宽和算力因 HBM 代际和制裁限制存在差异，表中为 B3 常见部署规格。

---

## 官方仓库

华为昇腾官方代码仓库主要托管在 **[GitCode（AtomGit）](https://gitcode.com/Ascend)**。2025 年 9 月，华为宣布 **CANN 全面开源**：开放 50+ 代码仓、800+ 算子。

### 核心计算平台（必读）

| 仓库 | 定位 | 关键作用 |
| :--- | :--- | :--- |
| [CANN](https://gitcode.com/cann) | Ascend 计算架构 | 对标 NVIDIA CUDA，是整个生态的底座 |
| [torch_npu](https://gitcode.com/Ascend/pytorch) | PyTorch Ascend 后端插件 | 让 PyTorch 模型无感迁移到 NPU |
| [torchair](https://gitcode.com/Ascend/torchair) | PyTorch 图模式编译后端 | `torch.compile` / GE 图模式，提升推理性能 |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | PyTorch 原生分布式训练插件 | 支持 FSDP2 / TP / CP / PP / EP，即插即用 |
| [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | TransformerEngine 昇腾版 | 配合 Megatron-LM 使用，FP8 训练加速 |

### 训练与微调套件

| 仓库 | 定位 | 适用场景 |
| :--- | :--- | :--- |
| [MindSpeed](https://gitcode.com/Ascend/MindSpeed) | 大模型加速核心库 | 一行 `import mindspeed.megatron_adaptor` 让 Megatron 跑在 NPU |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | 端到端 LLM 训练套件 | 预训练 / SFT / RLHF，支持 100+ 模型 |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | 多模态训练套件 | 视频生成（Wan2.1 / HunyuanVideo）、VLM（Qwen2.5-VL / InternVL3 / GLM4.1V） |
| [MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL) | RL 训练框架 | GRPO / DAPO / PPO / DPO，训推分离/共卡 |
| [ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) | 官方模型仓库 | 预优化脚本：Flash Attention、算子融合、并行策略 |

### 推理引擎

| 仓库 | 定位 | 特点 |
| :--- | :--- | :--- |
| [MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM) | 华为自研 LLM 推理引擎 | Continuous Batching / PageAttention / FlashDecoding，政企高合规场景首选 |
| MindIE-SD | 扩散模型推理引擎 | 文生图 / 视频推理，DiTCache 加速 |
| MindIE Turbo | 已并入 vLLM-Ascend | 无需单独安装 |

### 算子与编译器

| 仓库 | 定位 | 说明 |
| :--- | :--- | :--- |
| [ops-nn](https://gitcode.com/cann/ops-nn) | 神经网络高阶算子库 | MatMul / Activation / Norm / Pooling 等基础算子 |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | Transformer 专用算子 | FlashAttention / FusedAttention / GroupedMatmul |
| [ascend-transformer-boost](https://gitcode.com/cann/ascend-transformer-boost) | ATB 加速库 | 融合算子 + 图算子 + 插件机制，被 MindIE / MindSpeed 广泛调用 |
| [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) | MLIR 中间表示 | 昇腾亲和算子编译底座，已开源 |
| ops-adv | 高级融合算子库 | FlashAttention / GroupedMatmul 等，随 CANN 工具包分发 |

### 训练/推理配方（Recipes）

| 仓库 | 定位 |
| :--- | :--- |
| [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train) | 大模型预训练 / SFT / 微调示例 |
| [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) | 主流模型推理部署示例 |

### 领域库与工具

| 仓库 | 定位 | 适用场景 |
| :--- | :--- | :--- |
| [fbgemm-ascend](https://gitcode.com/Ascend/fbgemm-ascend) | 推荐系统 NPU 算子库 | 搜索/推荐/广告大规模 Embedding |
| [RecSDK](https://gitcode.com/Ascend/RecSDK) | 推荐系统 SDK | 含 TorchRec NPU 适配 |
| [vision](https://gitcode.com/Ascend/vision) | TorchVision 昇腾适配 | CV 模型迁移 |
| [apex](https://gitcode.com/Ascend/apex) | Apex 混合精度昇腾版 | AMP / 融合优化器 / 梯度融合 |
| [AMCT](https://gitcode.com/cann/amct) | 模型压缩工具包 | AWQ / GPTQ / SmoothQuant / FlatQuant；INT8 / INT4 / MXFP8 / MXFP4 |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | 企业级 Agent 训推框架 | v26.0，集成 MindSpeed-RL GRPO |

### 推荐软件栈

| 组件 | 推荐版本 | 说明 |
| :--- | :--- | :--- |
| CANN | 8.5.0 / 9.0.0 / 9.1.0 | 新硬件选 9.x，910B 可选 8.5 |
| PyTorch | 2.7.1 / 2.8.0 / 2.9.0 | 与 torch_npu 版本配套 |
| torch_npu | 2.8.0 / 2.9.0 | PyTorch Ascend 后端 |
| Python | 3.10 / 3.11 | 推荐 3.10 |
| Megatron-LM (Mcore) | core_r0.12.1 | MindSpeed 推荐版本 |

---

## 社区仓库

### 状态说明

| 符号 | 含义 |
| :--- | :--- |
| ✅ | 官方支持或原生适配，可直接使用 |
| ⚠️ | 社区维护或需手动适配，存在一定限制 |
| 🚧 | 正在积极开发中，功能不完整 |

### 社区推理引擎

| 仓库 | 定位 | 状态 |
| :--- | :--- | :--- |
| [vllm-ascend](https://github.com/vllm-project/vllm-ascend) | 昇腾推理的事实标准入口，官方 vLLM 插件 | ✅ |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能 LLM Serving，结构化生成 / PD 分离 / 投机解码 | ✅ |
| [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | SGLang Ascend 内核库：DeepEP-Ascend / MLA / GQA / FlashAttention | ✅ |

### 社区训练与强化学习

| 仓库 | 定位 | 状态 |
| :--- | :--- | :--- |
| [verl](https://github.com/volcengine/verl) | 字节跳动开源 RL 训练框架，2026 Q1-Q2 一等公民支持 Ascend | ✅ |
| [OpenRLHF](https://github.com/jianzhnie/OpenRLHF) | OpenRLHF 社区 Ascend 适配版（PPO / GRPO / DPO / KTO / REINFORCE++） | ⚠️ |
| [slime-ascend](https://gitcode.com/Ascend/slime-ascend) | 智谱 AI Post-Training / RL 框架（Megatron + SGLang） | ✅ |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | 异构推理/微调框架，2025.10 起支持 Ascend NPU，可与 LLaMA-Factory 联调超大 MoE 模型 | ✅ |
| [self-llm](https://github.com/datawhalechina/self-llm) | Datawhale《开源大模型食用指南》Ascend 版，覆盖 Qwen3 / DeepSeek / Llama 等 50+ 模型部署与微调 | ✅ |

### 社区算子与编译器

| 仓库 | 定位 | 状态 |
| :--- | :--- | :--- |
| [triton-ascend](https://github.com/triton-lang/triton-ascend) | Triton 语言昇腾后端，600+ 算子，100% 接口兼容 | ✅ |
| [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | Tile 级编程框架，300+ 算子，DeepSeek-V4 算子实践验证 | ✅ |

### Agent 与应用框架

| 框架 | 定位 | 状态 |
| :--- | :--- | :--- |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | 昇腾官方企业级 Agent 训推框架 | ✅ |
| [LangChain](https://python.langchain.ac.cn/docs/integrations/providers/ascend/) | LLM 应用框架，提供 `AscendEmbeddings` | ✅ |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent 图编排框架 | ✅ |
| [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | 基于 LangChain 的本地 RAG + Agent 应用框架，支持 Qwen / ChatGLM / Llama 等 | ✅ |
| [Dify](https://github.com/langgenius/dify) | LLM 应用开发平台，原生 `latest-gpu-cn` 镜像 | ✅ |
| [FastChat](https://github.com/lm-sys/FastChat) | 对话服务框架，需修改 `model_adapter.py` | ⚠️ |

### 推理服务与部署

| 框架 | 定位 | 状态 |
| :--- | :--- | :--- |
| [Xinference](https://github.com/xorbitsai/inference) | 异构算力推理框架，支持 Transformers / GGUF / MindIE | ✅ |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA 推理服务器，社区有 Ascend 适配参考 | ⚠️ |
| [KServe](https://github.com/kserve/kserve) | K8s 原生模型服务，通过 vLLM/SGLang 运行时支持 | ✅ |

### 量化压缩

| 框架 | 定位 | 推荐格式 |
| :--- | :--- | :--- |
| [AMCT](https://gitcode.com/cann/amct) | 昇腾原生压缩工具包 | INT8 / INT4 / MXFP8 / MXFP4 |
| msModelSlim | MindStudio 量化组件 | W8A8 / W4A8 / W4A4 |
| SGLang Quant | 内置于 [SGLang](https://github.com/sgl-project/sglang) | AWQ / GPTQ / AutoRound |

> **量化选型建议**：
> - **推理 ≥7B 模型**：首选 AWQ（W4A16），几乎无损。
> - **推理 <7B 模型**：GPTQ（W4A16）或 W8A8，小模型对量化更敏感。
> - **长上下文推理**：避免 W4A4，可能出现逻辑退化。
> - **训练场景**：BF16 + AMCT 感知量化（QAT）优于 PTQ。

### 多模态

| 框架 | 支持模型 | 关键优化 |
| :--- | :--- | :--- |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | Qwen2-VL / InternVL2.5 / GLM-4.1V / Wan2.1 / HunyuanVideo / SD3.5 / Flux | DiTCache 1.6× 视频加速；Ulysses 并行 3.75×；融合 RoPE +15% |
| vLLM-Ascend（多模态） | InternVL2 / Qwen-VL / LLaVA / MiniCPM | OpenAI 兼容 Vision API |
| [mllm](https://github.com/UbiquitousLearning/mllm) | Qwen3-VL / Qwen2-VL（W8A8） | 端侧/边缘 Ascend NPU 推理 |

### llama.cpp / GGUF

| 项目 | 定位 | 状态 |
| :--- | :--- | :--- |
| [BitCPM-CANN](https://huggingface.co/openbmb/BitCPM-CANN-0.5B-gguf) | 首个昇腾原生 1.58-bit 三元量化模型，内存节省 6× | ✅ |
| llama.cpp CANN | GGUF + CANN 后端社区探索 | 🚧 |

### HuggingFace 生态

| 组件 | Ascend 支持 | 最低版本 |
| :--- | :--- | :--- |
| [transformers](https://github.com/huggingface/transformers) | ✅ 通过 `torch_npu` 自动适配 | ≥ 4.32.0 |
| [accelerate](https://github.com/huggingface/accelerate) | ✅ 通过 `torch_npu` 适配 | ≥ 0.22.0 |
| [peft](https://github.com/huggingface/peft) | ✅ LoRA / Prefix Tuning | ≥ 0.5.0 |
| [trl](https://github.com/huggingface/trl) | ✅ RL 训练 | ≥ 0.5.0 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ✅ 通过 `torch_npu` 适配 | 最新版 |

---

## 主流模型 Ascend 适配矩阵

以下汇总主流开源大模型在 Ascend NPU 上的推理/训练适配情况，基于 [vllm-ascend 官方支持列表](https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_models.html) 和 [Datawhale self-llm Ascend 教程](https://github.com/datawhalechina/self-llm/blob/master/support_model_Ascend.md)。

### 文本生成模型

| 模型系列 | vLLM-Ascend | SGLang | MindIE | 训练框架 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek V3 / V3.1 / V3.2 / R1 / Distill** | ✅ | ✅ | ✅ | MindSpeed-LLM / torchtitan-npu / verl | Day-0 支持，MoE 优化成熟 |
| **DeepSeek-V4-Pro / Flash** | ✅ | ✅ | ✅ | torchtitan-npu / verl | 首个万亿参数 Ascend 全栈适配 |
| **Qwen3 / Qwen3-MoE / Qwen3-Next / Qwen3-Coder** | ✅ | ✅ | ✅ | MindSpeed-LLM / ms-swift / LLaMA-Factory | 阿里官方重点支持 |
| **Qwen2 / Qwen2.5 / QwQ-32B** | ✅ | ✅ | ✅ | MindSpeed-LLM / ms-swift / LLaMA-Factory | 社区最常用系列之一 |
| **Llama 2 / 3 / 3.1 / 3.2 / 4** | ✅ | ✅ | ⚠️ | MindSpeed-LLM / LLaMA-Factory | Llama-4 实验性支持 |
| **GLM-4 / GLM-4.1V / GLM-5** | ✅ | ✅ | ⚠️ | MindSpeed-LLM / slime-ascend | 智谱原生支持 |
| **Kimi-K2 / MiniMax-M2 / MiniMax-M3** | ✅ | ✅ | ⚠️ | slime-ascend / verl | Day-0 支持逐步完善 |
| **Gemma-2 / Gemma-3 / Phi-3 / Phi-4 / Mistral** | ✅ | 🟡 | ⚠️ | — | 部分需测试 |
| **Baichuan / ChatGLM / InternLM** | ✅ | 🟡 | ⚠️ | MindSpeed-LLM | 国产模型适配较好 |

### 多模态模型

| 模型 | vLLM-Ascend | SGLang | MindIE-SD | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL / Qwen2.5-VL / Qwen3-VL** | ✅ | ✅ | — | Vision-Language 理解 |
| **Qwen2.5-Omni / QVQ / Qwen2-Audio** | ✅ | 🟡 | — | 音频/多模态 |
| **LLaVA 1.5/1.6 / InternVL2 / InternVL2.5** | ✅ | ✅ | — | 视觉对话 |
| **MiniCPM-V / Gemma-3 (MM)** | ✅ | 🟡 | — | 端侧多模态 |
| **Wan2.1 / HunyuanVideo / CogVideoX** | — | — | ✅ | 视频生成（MindSpeed-MM / MindIE-SD） |
| **SD3.5 / SDXL / Flux** | — | — | ✅ | 图像生成 |

> **图例**：✅ 官方/原生支持；🟡 社区验证/部分支持；⚠️ 需适配或有限支持；— 不适用或待验证。

---

## 超大规模模型训练实践

### 华为盘古系列（昇腾原生）

| 模型 | 参数量 | 训练集群 | MFU / 关键指标 | 参考链接 |
| :--- | :--- | :--- | :--- | :--- |
| **盘古 Ultra** | 135B Dense | 8192 张昇腾 NPU | MFU **52%**；13.2T tokens | [arXiv:2504.07866](https://arxiv.org/abs/2504.07866) |
| **盘古 Ultra MoE** | 718B（激活 39B） | 万卡级 CloudMatrix 384 | 18.9% → 30% → **41%**（万卡）；256 路由专家 | [arXiv:2505.04519](https://arxiv.org/abs/2505.04519) |
| **盘古 Pro MoE** | 72B（激活 16B） | 4000 颗昇腾 NPU | 13T tokens（2025.06 开源） | [arXiv:2505.21411](https://arxiv.org/abs/2505.21411) |
| **盘古 Embedded** | 7B | 昇腾 NPU | 端侧小模型，已开源 | — |

**关键技术创新**：
- **DSSN（Depth-Scaled Sandwich-Norm）**：稳定超深网络训练
- **TinyInit**：小初始化抑制 loss spike
- **EP-Group 负载均衡**：提升 MoE 专家特化能力
- **MC² 通算融合**：将 AllReduce 与矩阵乘计算重叠
- **NFA（NPU Fusion Attention）**：昇腾亲和注意力融合算子

### DeepSeek 系列

| 模型 | Ascend 训练情况 | 参考链接 |
| :--- | :--- | :--- |
| **DeepSeek-V3** | 官方 2048 H800 训练；后续在昇腾 910B 完成适配验证 | [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) |
| **DeepSeek-V4** | **首个公开宣称与昇腾完成全栈适配的万亿参数 frontier 模型**（V4-Pro 1.6T / V4-Flash 284B）；基于 910B/910C 集群完成 SFT+RLHF 全参数后训练，MFU **34.9%** | [Model Card (PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) / [虎嗅-910C训练V4-Pro](https://www.huxiu.com/ainews/12966.html) / [搜狐-910C万亿参数训练](https://www.sohu.com/a/1032474134_114760) |
| **DeepSeek-R1** | 基于 H800 训练；昇腾侧以推理适配为主 | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |

### 其他厂商

| 厂商 / 模型 | Ascend 训练情况 | 关键信息 | 参考链接 |
| :--- | :--- | :--- | :--- |
| **美团 LongCat-2.0** | 5–6 万张国产加速卡预训练 | 万亿参数级，1M 上下文 | [LongCat-2.0 Preview](https://www.longcat.ai/) |
| **讯飞星火 V4.0 / X2** | "飞星一号/二号" 昇腾万卡集群 | 训练效率从 A100 的 30-50% → 85-95% | [讯飞星火官网](https://xinghuo.xfyun.cn/) |
| **百度 ERNIE 4.5** | 300B MoE（A47B）；PaddlePaddle 支持昇腾 | H800 MFU 47%；昇腾适配推进中 | [ERNIE 4.5 GitHub](https://github.com/PaddlePaddle/ERNIE) |
| **智谱 GLM-4.5** | 355B MoE；slime 框架支持昇腾 | 23T tokens | [智谱 AI](https://www.zhipuai.cn/) |
| **智谱 GLM-Image** | **首个基于昇腾 + MindSpore 全程训练的 SOTA 多模态模型** | 全流程国产 | [智谱 GLM-Image](https://www.zhipuai.cn/) |
| **阿里 Qwen 系列** | ModelArts 官方支持昇腾训练 | SFT / LoRA / GRPO 全流程 | [Qwen GitHub](https://github.com/QwenLM/Qwen) / [ModelArts](https://www.huaweicloud.com/product/modelarts.html) |

### 训练实践总结

| 维度 | 现状 |
| :--- | :--- |
| **可用规模** | 千卡→万卡级；百亿→准万亿参数模型可训 |
| **稠密 MFU** | 盘古 Ultra 达 **52%**（对标 H100 的 45-55%） |
| **MoE MFU** | 盘古 Ultra MoE 达 **41%**（万卡），All-to-All 通信是主要瓶颈 |
| **与 NVIDIA 差距** | 单卡 BF16：910B 约 H100 的 32%，910C 约 80%；集群效率约为 H100/H800 的 35%-70% |
| **关键挑战** | 超长稳训练稳定性、MoE All-to-All 通信、FP8 极致优化、大规模 RL 训练 |

### 相关论文与 HuggingFace 模型仓库

| 模型 | HuggingFace 仓库 | 技术报告 / 论文 |
| :--- | :--- | :--- |
| **盘古 Ultra** | 尚未在 HuggingFace 公开发布 | [arXiv:2504.07866](https://arxiv.org/abs/2504.07866) |
| **盘古 Ultra MoE** | 尚未在 HuggingFace 公开发布 | [arXiv:2505.04519](https://arxiv.org/abs/2505.04519) |
| **盘古 Pro MoE** | 尚未在 HuggingFace 公开发布 | [arXiv:2505.21411](https://arxiv.org/abs/2505.21411) |
| **DeepSeek-V3** | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) |
| **DeepSeek-V4** | [deepseek-ai/DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4) | [Model Card (PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) |
| **DeepSeek-R1** | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| **Qwen3** | [Qwen/Qwen3](https://huggingface.co/Qwen/Qwen3) | [Qwen GitHub](https://github.com/QwenLM/Qwen3) |
| **GLM-4 / GLM-4.1V / GLM-5** | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) / [THUDM/GLM-4.1V](https://huggingface.co/THUDM/GLM-4.1V) | [ChatGLM GitHub](https://github.com/THUDM/GLM-4) |
| **ERNIE 4.5** | 百度文心系列主要在 ModelScope / 百度平台 | [ERNIE GitHub](https://github.com/PaddlePaddle/ERNIE) |

### 相关新闻报道

- [深圳国产芯片成功训练万亿级 AI 大模型，依托华为昇腾 910C](https://www.sohu.com/a/1032474134_114760) — 搜狐，2026.06
- [昇腾 910C 千卡集群完成 1.6 万亿参数大模型全参数训练](https://post.smzdm.com/p/ae6pl3g4) — 什么值得买，2026.06
- [国产 AI 算力突破：华为昇腾 910C 完成 1.6 万亿参数大模型训练](https://k.sina.com.cn/article_7096020433_1a6f4add106801ifzw.html) — 新浪，2026.06
- [Huawei Ascend 950 PR Is Powering China’s Push to Break CUDA Dependence](https://www.trendforce.com/news/2026/04/07/news-decoding-deepseek-v4-how-huaweis-ascend-950-pr-is-powering-chinas-push-to-break-cuda-dependence/) — TrendForce，2026.04
- [Huawei Cloud Model-as-a-Service on the CloudMatrix384 SuperPod](https://arxiv.org/abs/2508.02520) — arXiv，2025.08

---

## 开发者工具链

### MindStudio（2025 年底全量开源）

围绕三大工具链 + 通用层构建：

| 工具链 | 组件 | 用途 |
| :--- | :--- | :--- |
| **msTT（训练）** | msProf / msprof-analyze / msMemScope / msProbe | 性能采集、自动分析、内存诊断、精度调试 |
| **msIT（推理）** | msServiceProfiler / msModelSlim / msPrechecker | 推理服务 profiling、量化压缩、环境预检 |
| **msOT（算子）** | msKPP / msOpGen / msProf / msDebug / msSanitizer | 性能建模、算子生成、算子调优、调试、异常检测 |
| **通用层** | msInsight / msMonitor / msPTI | 可视化分析、集群监控、Profiling API |

### MindStudio Insight（原名 Ascend Insight）

可视化性能调优工具，支持百卡/千卡集群分析，可处理 20 GB 性能文件。

核心视图：Timeline / Operator / Memory / Communication / Jupyter。

### CANNBot Agent

AI 辅助编程工具，支持 Claude Code / Cursor / Trae，集成算子开发、精度调试、性能分析。单个算子开发效率提升 **5×+**。

---

## 社区与学习资源

| 资源 | 说明 |
| :--- | :--- |
| [Ascend（GitCode）](https://gitcode.com/Ascend) | 官方仓库合集 |
| [CANN（GitCode）](https://gitcode.com/cann) | CANN 核心开源社区主阵地 |
| [昇腾社区](https://www.hiascend.com) | 官方文档、SDK、技术文章、活动 |
| [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) | 四级学习路径：入门 → 进阶 → 高级 → 专家 |
| [cann-community](https://gitcode.com/cann/community) | CANN 社区治理与 SIG 协作 |
| [HiDevLab](https://www.hiascend.com) | 免费申请昇腾 NPU 算力（初始 100 卡时） |
| Docker 基础镜像 | `quay.io/ascend/cann:8.5.0-a3-openeuler24.03-py3.11` |

---

## 场景化选型指南

| 你的需求 | 推荐方案 | 理由 |
| :--- | :--- | :--- |
| 已有 Megatron 脚本，想迁移到 NPU | `MindSpeed` | 一行 import 即可适配 |
| 7B–70B 模型预训练/微调 | `MindSpeed-LLM` 或 `torchtitan-npu` | 开箱即用，配套完整 |
| PyTorch 原生训练（FSDP2） | `torchtitan-npu` | 对齐社区标准，AutoFuse 自动优化 |
| RLHF / GRPO 后训练 | `verl` + `vLLM-Ascend` 或 `MindSpeed-RL` | 训推一体化，吞吐高 |
| 生产级 LLM 推理服务 | `vLLM-Ascend` 或 `SGLang` | 社区原生支持，生态活跃 |
| 政企高合规推理 | `MindIE-LLM` | 华为自研，私有化部署成熟 |
| 多模态训练/推理 | `MindSpeed-MM` + `MindIE-SD` | 视频/图像/VLM 全链路支持 |
| 快速搭建 RAG / Agent 应用 | `Dify` + `LangChain` | 原生镜像，接入成本低 |
| 超长序列（32k–1M tokens） | `torchtitan-npu` 或 `DeepSpeed-Ulysses` | 长序列优化更成熟 |
| 推荐/广告大模型 | `RecSDK` / `fbgemm-ascend` | 搜广推专用算子 |
| 端侧/边缘部署 | `mllm` / `MindIE` 轻量化方案 | 低功耗、W8A8 量化 |

---

## 训练框架深度对比

| 框架 | 并行策略 | 最佳场景 | 昇腾适配入口 | 性能特点 |
| :--- | :--- | :--- | :--- | :--- |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | FSDP2 / TP / CP / PP / EP | PyTorch 原生训练、DeepSeek-V4 / Qwen3 续训练 | `pip install -e .` | AutoFuse 自动融合，MFU 27.67%（V4-Flash） |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | TP / PP / DP / EP / CP + ZeRO | 100+ 模型端到端训练 | `import mindspeed.megatron_adaptor` | 大模型训练首选，优化层级 L0/L1/L2 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO-1/2/3/Infinity + Ulysses SP | CPU/NVMe 卸载、超长序列 | `torch_npu` 原生支持 | 适合超大规模或超长上下文 |
| PyTorch FSDP2 | FSDP2 + Hybrid Shard | 中小模型（<70B） | `torch_npu` 原生支持 | 社区标准，学习成本低 |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | PP / TP / VPP / CP / Encoder-DP | 多模态训练 | 完整套件 | 比通用方案快 **19-24%** |

### torchtitan-npu 性能基准（Atlas 800T A3，64 卡）

| 模型 | 精度 | 并行策略 | 吞吐 (tokens/p/s) | MFU |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V4-Flash | BF16 | FSDP128 + EP128 | 1056 | 27.67% |
| DeepSeek-V3-671B | BF16 | FSDP32 + TP4 + EP128 | 546 | — |
| DeepSeek-V3-671B + compile(AutoFuse) | BF16 | FSDP32 + TP4 + EP128 | 576 | — |

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

## 常见问题 FAQ

**Q1：我已经有 CUDA 上的 PyTorch 模型，迁移到 Ascend 需要改多少代码？**

通常只需：
1. 安装 `torch_npu`；
2. 将 `.to("cuda")` 改为 `.to("npu")`；
3. 替换 CUDA-specific 的算子（如 `torch.cuda` → `torch.npu`）。

复杂模型建议使用 `MindSpeed` 或 `torchtitan-npu` 获得更好性能。

**Q2：为什么 HuggingFace 模型在 Ascend 上能跑但性能不好？**

HF + `torch_npu` 提供的是"GPU 等价能力"，没有 Ascend 专属 kernel 优化。追求极限性能需使用 `MindSpeed-LLM`、`vLLM-Ascend` 或 `SGLang`。

**Q3：昇腾适合训多大的模型？**

目前已验证：
- 稠密模型：135B（盘古 Ultra，8192 NPU）
- MoE 模型：718B（盘古 Ultra MoE，万卡）、1.6T（DeepSeek-V4，910C 集群）

**Q4：QLoRA 在 Ascend 上能用吗？**

目前有限制。`bitsandbytes` 对 NPU 支持不完善，推荐使用 Full / LoRA / Freeze 策略。

**Q5：推理部署选 vLLM-Ascend 还是 MindIE？**

- 追求开源生态、快速迭代：选 **vLLM-Ascend** 或 **SGLang**
- 政企私有化、高合规、需要官方支持：选 **MindIE-LLM**

**Q6：Ascend 950 什么时候能用上？**

- 950PR：2026 Q1
- 950DT：2026 Q4（部分云厂商可能提前到 2026.08）

---

## Contributing

欢迎提交 PR，补充昇腾 NPU 相关的开源项目、框架和学习资源。

1. Fork 本仓库
2. 在对应分类下添加项目（格式：`[repo-name](link) — 一句话说明`）
3. 确保链接可访问
4. 提交 PR
