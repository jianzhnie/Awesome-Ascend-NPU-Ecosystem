# Awesome Ascend NPU Ecosystem

A curated list of awesome Ascend NPU hardware, official repositories, community projects, training/inference frameworks, developer tools, and learning resources.

> Huawei Ascend NPU ecosystem exploded in 2024-2026. This list covers the full landscape from silicon to applications.
>
> 🌐 [中文版 README](./README.md)

---

## Contents

- [Ascend Ecosystem at a Glance](#ascend-ecosystem-at-a-glance)
- [Hardware Roadmap](#hardware-roadmap)
- [Official Repositories](#official-repositories)
  - [Core Compute Platform](#core-compute-platform)
  - [Training Suites](#training-suites)
  - [Inference Engines](#inference-engines)
  - [Operators & Compilers](#operators--compilers)
  - [Recipes](#recipes)
  - [Domain Libraries & Tools](#domain-libraries--tools)
- [Community Repositories](#community-repositories)
  - [Inference Engines](#community-inference-engines)
  - [Training & RL](#training--rl)
  - [Kernel & Compiler](#kernel--compiler)
  - [Agent & Application Frameworks](#agent--application-frameworks)
  - [Serving & Deployment](#serving--deployment)
  - [Quantization](#quantization)
  - [Multimodal](#multimodal)
  - [llama.cpp / GGUF](#llamacpp--gguf)
  - [HuggingFace Ecosystem](#huggingface-ecosystem)
- [Ascend Model Support Matrix](#ascend-model-support-matrix)
- [Large-Scale Training on Ascend](#large-scale-training-on-ascend)
- [Developer Tools](#developer-tools)
- [Community & Learning](#community--learning)
- [Quick Start](#quick-start)
- [Scenario-Based Selection Guide](#scenario-based-selection-guide)
- [Training Framework Comparison](#training-framework-comparison)
- [FAQ](#faq)
- [Contributing](#contributing)

---

## Ascend Ecosystem at a Glance

```text
┌─────────────────────────────────────────────────────────────┐
│  Apps: Dify / LangChain / AgentSDK / FastChat / RAG         │
├─────────────────────────────────────────────────────────────┤
│  Serving: vLLM-Ascend / SGLang / MindIE / Xinference / KServe│
├─────────────────────────────────────────────────────────────┤
│  Frameworks: MindSpeed-LLM / MindSpeed-MM / torchtitan-npu / verl│
├─────────────────────────────────────────────────────────────┤
│  Acceleration: torch_npu / torchair / ops-transformer / Triton│
├─────────────────────────────────────────────────────────────┤
│  Foundation: CANN / AscendNPU-IR / Ascend C / Drivers / FW  │
├─────────────────────────────────────────────────────────────┤
│  Hardware: 910B / 910C / 950PR / 950DT / SuperNode / SuperPoD│
└─────────────────────────────────────────────────────────────┘
```

**Key takeaway**: Ascend now covers the full stack from silicon to applications. The **software optimization headroom is larger than the single-card hardware gap** — cluster-level tuning and native Ascend stacks (MindSpeed, vLLM-Ascend, SGLang) are the keys to unlocking performance.

---

## Hardware Roadmap

| Chip | Status | Released | FP16 | FP8 | HBM | Bandwidth | Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ascend 910B** | Shipping | 2023 Q1 | ~320 TFLOPS | — | 64 GB | ~1.2 TB/s | Training + Inference |
| **Ascend 910C** | Shipping (current) | 2025 Q1 | ~800 TFLOPS | — | 128 GB | ~3.2 TB/s | Training + Inference |
| **Ascend 950PR** | Planned | 2026 Q1 | — | ~1 PFLOPS | 128 GB (HiBL 1.0) | ~1.6 TB/s | Inference (Prefill) |
| **Ascend 950DT** | Planned | 2026 Q4 | — | ~1 PFLOPS (FP4: ~2P) | 144 GB (HiZQ 2.0) | ~4.0 TB/s | Training + Inference (Decode) |
| **Ascend 960** | Planned | 2027 Q4 | — | — | — | — | Next-gen |
| **Ascend 970** | Planned | 2028 Q4 | — | — | — | — | Next-gen |

**Key architectural change**: 950 series introduces SIMD/SIMT dual-mode architecture — 4× discrete memory access efficiency, 30% higher vector compute ratio, 50% lower task scheduling latency vs 910 series.

---

## Official Repositories

Official repos are hosted on **[GitCode (AtomGit)](https://gitcode.com/Ascend)**. In Sept 2025, Huawei announced full CANN open-source: 50+ repos, 800+ operators.

### Core Compute Platform

| Repository | Description |
| :--- | :--- |
| [CANN](https://gitcode.com/cann) | Ascend Compute Architecture — the CUDA equivalent for Ascend NPUs |
| [torch_npu](https://gitcode.com/Ascend/pytorch) | PyTorch Ascend backend plugin (Ascend Extension for PyTorch) |
| [torchair](https://gitcode.com/Ascend/torchair) | PyTorch graph-mode (JIT / `torch.compile`) compiler backend for Ascend |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | Plug-and-play Ascend backend for Meta's torchtitan distributed training framework |
| [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | NVIDIA TransformerEngine ported to Ascend, used with Megatron-LM |

### Training Suites

| Repository | Description |
| :--- | :--- |
| [MindSpeed](https://gitcode.com/Ascend/MindSpeed) | Core large-model acceleration library |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | End-to-end LLM distributed training suite (based on Megatron-LM, 100+ models) |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | Multimodal training suite (Wan2.1, Qwen2.5-VL, InternVL3, HunyuanVideo, GLM4.1V, etc.) |
| [MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL) | RL training framework (GRPO / DAPO / PPO / DPO) |
| [ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) | Official Ascend model zoo with pre-optimized scripts (Flash Attention, operator fusion, etc.) |

### Inference Engines

| Repository | Description |
| :--- | :--- |
| [MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM) | Huawei's native LLM inference engine (Continuous Batching, PageAttention, FlashDecoding) |
| MindIE-SD | Diffusion model inference engine (part of the MindIE suite) |
| MindIE Turbo | Merged into vLLM-Ascend — no separate installation needed |

### Operators & Compilers

| Repository | Description |
| :--- | :--- |
| [ops-nn](https://gitcode.com/cann/ops-nn) | Neural network operator library (MatMul, Activation, Norm, Pooling, etc.) |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | Transformer-specific operators (FlashAttention, FusedAttention, GroupedMatmul) |
| [ascend-transformer-boost](https://gitcode.com/cann/ascend-transformer-boost) | ATB (Ascend Transformer Boost) — fused ops + graph ops + plugin mechanism |
| [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) | MLIR-based intermediate representation for Ascend operator compilation |
| ops-adv | Advanced fusion operator library (FlashAttention, GroupedMatmul) — bundled with CANN toolkit |

### Recipes

| Repository | Description |
| :--- | :--- |
| [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train) | Training recipes: pretrain / SFT / fine-tuning examples |
| [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) | Inference recipes: mainstream model deployment examples |

### Domain Libraries & Tools

| Repository | Description |
| :--- | :--- |
| [fbgemm-ascend](https://gitcode.com/Ascend/fbgemm-ascend) | High-performance PyTorch NPU kernel library for recommendation systems |
| [vision](https://gitcode.com/Ascend/vision) | TorchVision Ascend NPU adaptation |
| [apex](https://gitcode.com/Ascend/apex) | NVIDIA Apex mixed-precision training ported to Ascend |
| [RecSDK](https://gitcode.com/Ascend/RecSDK) | Recommendation SDK with TorchRec NPU adaptation |
| [AMCT](https://gitcode.com/cann/amct) | Ascend Model Compression Toolkit (AWQ / GPTQ / SmoothQuant / FlatQuant; INT8 / INT4 / MXFP8 / MXFP4 / HiF8) |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | Enterprise-grade Agent training & inference framework (v26.0), integrates MindSpeed-RL GRPO |

### Recommended Software Stack

| Component | Recommended Version |
| :--- | :--- |
| CANN | 8.5.0 / 9.0.0 / 9.1.0 |
| PyTorch | 2.7.1 / 2.8.0 / 2.9.0 |
| torch_npu | 2.8.0 / 2.9.0 |
| Python | 3.10 / 3.11 |
| Megatron-LM (Mcore) | core_r0.12.1 |

---

## Community Repositories

### Status Legend

| Symbol | Meaning |
| :--- | :--- |
| ✅ | Officially supported or natively adapted, ready to use |
| ⚠️ | Community-maintained or needs manual adaptation, with some limitations |
| 🚧 | Actively under development, feature incomplete |

### Community Inference Engines

| Repository | Description |
| :--- | :--- |
| [vllm-ascend](https://github.com/vllm-project/vllm-ascend) | The de-facto Ascend inference engine — official vLLM community plugin. Supports DeepSeek-V3/V4, Qwen3, GLM5, Kimi-K2, etc. |
| [SGLang](https://github.com/sgl-project/sglang) | High-performance LLM serving with native Ascend support (since Jul 2025). Structured generation, PD separation, speculative decoding. |
| [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) | SGLang Ascend kernel library: DeepEP-Ascend, MLA, GQA, FlashAttention, SwiGLU, Mamba |
| [openfuyao-vllm-ascend](https://github.com/qsunnyy/openfuyao-vllm-ascend) | Community-maintained vLLM-Ascend fork |

### Training & RL

| Repository | Description |
| :--- | :--- |
| [verl](https://github.com/volcengine/verl) | ByteDance RL training framework — Ascend as first-class citizen (2026 Q1-Q2) |
| [OpenRLHF](https://github.com/jianzhnie/OpenRLHF) | Community Ascend adaptation of OpenRLHF (PPO/GRPO/DPO/KTO/REINFORCE++) |
| [slime-ascend](https://gitcode.com/Ascend/slime-ascend) | Zhipu AI post-training / RL scaling framework (Megatron + SGLang). Supports GLM-5, Qwen3, DeepSeek V3/R1, Llama 3. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Heterogeneous inference/fine-tuning framework; Ascend NPU support since Oct 2025; integrates with LLaMA-Factory for ultra-large MoE models |
| [self-llm](https://github.com/datawhalechina/self-llm) | Datawhale "Open Source LLM Cooking Guide" Ascend edition; covers 50+ models including Qwen3, DeepSeek, Llama |

### Kernel & Compiler

| Repository | Description |
| :--- | :--- |
| [triton-ascend](https://github.com/triton-lang/triton-ascend) | Triton language Ascend backend — 600+ Triton ops, 100% interface compatible |
| [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | Tile-level DSL for Ascend kernels — 300+ TileLang ops; Beginner / Developer / Expert modes |

### Agent & Application Frameworks

| Repository | Description | Ascend Status |
| :--- | :--- | :--- |
| [AgentSDK](https://gitcode.com/Ascend/AgentSDK) | Official Ascend Agent training/inference framework | ✅ Native (v26.0) |
| [LangChain](https://python.langchain.ac.cn/docs/integrations/providers/ascend/) | LLM application framework | ✅ Official `AscendEmbeddings` integration |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph orchestration | ✅ Verified with DeepSeek + Ascend |
| [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | Local RAG + Agent app framework based on LangChain; supports Qwen, ChatGLM, Llama | ✅ |
| [Dify](https://github.com/langgenius/dify) | LLM application platform | ✅ Native `langgenius/dify:latest-gpu-cn` image |
| [FastChat](https://github.com/lm-sys/FastChat) | Conversational serving | ⚠️ Needs `model_adapter.py` patch for multi-NPU |

### Serving & Deployment

| Repository | Description | Ascend Status |
| :--- | :--- | :--- |
| [Xinference](https://github.com/xorbitsai/inference) | Heterogeneous compute inference framework | ✅ Transformers backend; enterprise edition supports MindIE |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA inference server | ⚠️ Ascend adaptation available (community reference) |
| [KServe](https://github.com/kserve/kserve) | K8s-native model serving | ✅ Via vLLM/SGLang runtime |

### Quantization

| Framework | Description | Supported Formats |
| :--- | :--- | :--- |
| [AMCT](https://gitcode.com/cann/amct) | Official Ascend compression toolkit | INT8, INT4, MXFP8, MXFP4, HiF8 (AWQ/GPTQ/SmoothQuant/FlatQuant) |
| [msModelSlim](https://www.hiascend.com) | MindStudio quantization component | W8A8, W4A8, W8A16, W4A4 (LLM + multimodal + diffusion) |
| SGLang Quant | Built into [SGLang](https://github.com/sgl-project/sglang) | AWQ, GPTQ, AutoRound, W4A16, W8A16 |

> **Recommendation**: AWQ (W4A16) is the default for ≥7B models on Ascend — near lossless. GPTQ is the runner-up. W4A4 shows logic degradation on long-context reasoning.

### Multimodal

| Framework | Supported Models | Key Optimizations |
| :--- | :--- | :--- |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | Qwen2-VL, InternVL2/2.5, GLM-4.1V, LLaVA, CogVideoX, HunyuanVideo, Wan2.1, SD3.5/SDXL, Flux | DiTCache 1.6× video speedup; Ulysses parallel 3.75×; fused RoPE +15% |
| vLLM-Ascend (vision) | InternVL2, Qwen-VL, Qwen2-VL, LLaVA, MiniCPM | OpenAI-compatible Vision API |
| [mllm](https://github.com/UbiquitousLearning/mllm) | Qwen3-VL, Qwen2-VL (W8A8) | Edge Ascend NPU inference (2026.05) |

### llama.cpp / GGUF

| Project | Description |
| :--- | :--- |
| [BitCPM-CANN](https://huggingface.co/openbmb/BitCPM-CANN-0.5B-gguf) | First native Ascend 1.58-bit ternary quantized model (OpenBMB, 2026) — 6× memory savings, ≥95.7% FP performance |
| llama.cpp CANN | Community adaptation in progress ([ggml-org/llama.cpp#11698](https://github.com/ggml-org/llama.cpp/discussions/11698)) |

### HuggingFace Ecosystem

| Component | Ascend Support | Min Version |
| :--- | :--- | :--- |
| [transformers](https://github.com/huggingface/transformers) | ✅ Auto-adaptation via `torch_npu` | ≥ 4.32.0 |
| [accelerate](https://github.com/huggingface/accelerate) | ✅ Via `torch_npu` | ≥ 0.22.0 |
| [peft](https://github.com/huggingface/peft) | ✅ LoRA / Prefix Tuning | ≥ 0.5.0 |
| [trl](https://github.com/huggingface/trl) | ✅ RL training | ≥ 0.5.0 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ✅ Via `torch_npu` | Latest |

---

## Ascend Model Support Matrix

A summary of mainstream open-source models adapted to Ascend NPU, based on the [vllm-ascend supported models list](https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_models.html) and [Datawhale self-llm Ascend tutorials](https://github.com/datawhalechina/self-llm/blob/master/support_model_Ascend.md).

### Text Generation Models

| Model Family | vLLM-Ascend | SGLang | MindIE | Training Frameworks | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek V3 / V3.1 / V3.2 / R1 / Distill** | ✅ | ✅ | ✅ | MindSpeed-LLM / torchtitan-npu / verl | Day-0 support, mature MoE optimization |
| **DeepSeek-V4-Pro / Flash** | ✅ | ✅ | ✅ | torchtitan-npu / verl | First trillion-param full-stack Ascend adaptation |
| **Qwen3 / Qwen3-MoE / Qwen3-Next / Qwen3-Coder** | ✅ | ✅ | ✅ | MindSpeed-LLM / ms-swift / LLaMA-Factory | Official priority support from Alibaba |
| **Qwen2 / Qwen2.5 / QwQ-32B** | ✅ | ✅ | ✅ | MindSpeed-LLM / ms-swift / LLaMA-Factory | One of the most widely used families |
| **Llama 2 / 3 / 3.1 / 3.2 / 4** | ✅ | ✅ | ⚠️ | MindSpeed-LLM / LLaMA-Factory | Llama-4 experimental support |
| **GLM-4 / GLM-4.1V / GLM-5** | ✅ | ✅ | ⚠️ | MindSpeed-LLM / slime-ascend | Native Zhipu support |
| **Kimi-K2 / MiniMax-M2 / MiniMax-M3** | ✅ | ✅ | ⚠️ | slime-ascend / verl | Day-0 support improving |
| **Gemma-2 / Gemma-3 / Phi-3 / Phi-4 / Mistral** | ✅ | 🟡 | ⚠️ | — | Some need testing |
| **Baichuan / ChatGLM / InternLM** | ✅ | 🟡 | ⚠️ | MindSpeed-LLM | Good domestic model adaptation |

### Multimodal Models

| Model | vLLM-Ascend | SGLang | MindIE-SD | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL / Qwen2.5-VL / Qwen3-VL** | ✅ | ✅ | — | Vision-Language understanding |
| **Qwen2.5-Omni / QVQ / Qwen2-Audio** | ✅ | 🟡 | — | Audio / multimodal |
| **LLaVA 1.5/1.6 / InternVL2 / InternVL2.5** | ✅ | ✅ | — | Visual dialogue |
| **MiniCPM-V / Gemma-3 (MM)** | ✅ | 🟡 | — | Edge multimodal |
| **Wan2.1 / HunyuanVideo / CogVideoX** | — | — | ✅ | Video generation (MindSpeed-MM / MindIE-SD) |
| **SD3.5 / SDXL / Flux** | — | — | ✅ | Image generation |

> **Legend**: ✅ official/native support; 🟡 community verified/partial support; ⚠️ needs adaptation or limited support; — not applicable or to be verified.

---

## Large-Scale Training on Ascend

### Pangu Series (Ascend-native)

| Model | Params | Cluster | MFU / Key Metrics | References |
| :--- | :--- | :--- | :--- | :--- |
| **Pangu Ultra** | 135B Dense | 8192 Ascend NPUs | MFU **52%**; 13.2T tokens | [arXiv:2504.07866](https://arxiv.org/abs/2504.07866) |
| **Pangu Ultra MoE** | 718B (39B active) | 10K+ CloudMatrix 384 | MFU: 18.9% → 30% → **41%** at 10K; 256 routed experts, 8 active | [arXiv:2505.04519](https://arxiv.org/abs/2505.04519) |
| **Pangu Pro MoE** | 72B (16B active) | 4000 Ascend NPUs | 13T tokens (open-sourced Jun 2025) | [arXiv:2505.21411](https://arxiv.org/abs/2505.21411) |
| **Pangu Embedded** | 7B | Ascend NPU | Edge model, open-sourced | — |

### DeepSeek Series

| Model | Ascend Training Status | References |
| :--- | :--- | :--- |
| **DeepSeek-V3** | Trained on 2048× H800; later adapted & validated on Ascend 910B | [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) |
| **DeepSeek-V4** | First frontier model publicly confirmed with full-stack Ascend adaptation (V4-Pro 1.6T, V4-Flash 284B); trained & SFT+RLHF on 910B/910C clusters, MFU 34.9% | [Model Card (PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) / [Huxiu-910C training](https://www.huxiu.com/ainews/12966.html) / [Sohu-trillion-param training](https://www.sohu.com/a/1032474134_114760) |
| **DeepSeek-R1** | Trained on H800; Ascend handles inference adaptation | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |

### Other Vendors

| Vendor / Model | Ascend Training | Key Info | References |
| :--- | :--- | :--- | :--- |
| **Meituan LongCat-2.0** | Pretrained on 50-60K domestic accelerators | Trillion-param (1T+), 1M context | [LongCat-2.0 Preview](https://www.longcat.ai/) |
| **iFlytek Spark V4.0/X2** | Wan-card Ascend clusters | Training efficiency improved from A100 30-50% → 85-95% | [iFlytek Spark](https://xinghuo.xfyun.cn/) |
| **Baidu ERNIE 4.5** | 300B MoE (A47B); PaddlePaddle Ascend support | H800 MFU 47% (Ascend adaptation in progress) | [ERNIE GitHub](https://github.com/PaddlePaddle/ERNIE) |
| **Zhipu GLM-4.5** | 355B MoE; slime framework on Ascend | 23T tokens | [Zhipu AI](https://www.zhipuai.cn/) |
| **Zhipu GLM-Image** | First SOTA multimodal model trained entirely on Ascend + MindSpore | Full domestic workflow | [Zhipu GLM-Image](https://www.zhipuai.cn/) |
| **Alibaba Qwen Series** | ModelArts official Ascend training support | SFT / LoRA / GRPO full workflow | [Qwen GitHub](https://github.com/QwenLM/Qwen) / [ModelArts](https://www.huaweicloud.com/product/modelarts.html) |

### Training Summary

| Dimension | Status |
| :--- | :--- |
| **Scale** | 1K→10K+ NPU clusters; 100B→trillion-param models |
| **Dense MFU** | Pangu Ultra 52% (on par with H100 45-55%) |
| **MoE MFU** | Pangu Ultra MoE 41% (10K NPUs); A2A communication is the bottleneck |
| **vs. NVIDIA** | Single-card BF16: 910B ~32% of H100, 910C ~80%; cluster training efficiency 35%-70% of H100/H800 |
| **Key challenges** | Ultra-long-run training stability, MoE All-to-All communication, FP8 optimization, large-scale RL |

### Related Papers & HuggingFace Repos

| Model | HuggingFace Repo | Technical Report / Paper |
| :--- | :--- | :--- |
| **Pangu Ultra** | Not yet publicly released on HuggingFace | [arXiv:2504.07866](https://arxiv.org/abs/2504.07866) |
| **Pangu Ultra MoE** | Not yet publicly released on HuggingFace | [arXiv:2505.04519](https://arxiv.org/abs/2505.04519) |
| **Pangu Pro MoE** | Not yet publicly released on HuggingFace | [arXiv:2505.21411](https://arxiv.org/abs/2505.21411) |
| **DeepSeek-V3** | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3) |
| **DeepSeek-V4** | [deepseek-ai/DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4) | [Model Card (PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) |
| **DeepSeek-R1** | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| **Qwen3** | [Qwen/Qwen3](https://huggingface.co/Qwen/Qwen3) | [Qwen GitHub](https://github.com/QwenLM/Qwen3) |
| **GLM-4 / GLM-4.1V / GLM-5** | [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) / [THUDM/GLM-4.1V](https://huggingface.co/THUDM/GLM-4.1V) | [ChatGLM GitHub](https://github.com/THUDM/GLM-4) |
| **ERNIE 4.5** | Baidu Wenxin mainly on ModelScope / Baidu platform | [ERNIE GitHub](https://github.com/PaddlePaddle/ERNIE) |

### Related News

- [Shenzhen domestic chips successfully trained trillion-parameter AI model on Huawei Ascend 910C](https://www.sohu.com/a/1032474134_114760) — Sohu, Jun 2026
- [Ascend 910C cluster completed full-parameter training of 1.6T model](https://post.smzdm.com/p/ae6pl3g4) — SMZDM, Jun 2026
- [Domestic AI computing breakthrough: Huawei Ascend 910C trained 1.6T model](https://k.sina.com.cn/article_7096020433_1a6f4add106801ifzw.html) — Sina, Jun 2026
- [Huawei Ascend 950 PR Is Powering China’s Push to Break CUDA Dependence](https://www.trendforce.com/news/2026/04/07/news-decoding-deepseek-v4-how-huaweis-ascend-950-pr-is-powering-chinas-push-to-break-cuda-dependence/) — TrendForce, Apr 2026
- [Huawei Cloud Model-as-a-Service on the CloudMatrix384 SuperPod](https://arxiv.org/abs/2508.02520) — arXiv, Aug 2025

---

## Developer Tools

### MindStudio

Huawei Ascend full-stack IDE (fully open-sourced Dec 2025). Three tool chains:

| Tool Chain | Components | Purpose |
| :--- | :--- | :--- |
| **msTT (Training)** | msProf, msprof-analyze, msMemScope, msProbe | Performance profiling, auto-analysis (advisor/compare/cluster), memory diagnostics, precision debugging |
| **msIT (Inference)** | msServiceProfiler, msModelSlim, msPrechecker | Inference serving profiling (vLLM/SGLang/MindIE), quantization compression, env pre-check |
| **msOT (Operator)** | msKPP, msOpGen, msProf, msDebug, msSanitizer | Performance modeling, operator generation, tuning, debugging, anomaly detection |
| **Shared** | msInsight, msMonitor, msPTI | Visual analysis (Timeline/Operator/Memory/Communication), online cluster monitoring, Profiling API |

### MindStudio Insight (formerly Ascend Insight)

Visual performance tuning tool supporting 100-1000 card cluster analysis, handling up to 20 GB cluster profiling data.

**Core views**: Timeline (pipeline/overlap/communication-computation overlay), Operator (op timing breakdown), Memory (usage curves), Communication (full network link/slow node analysis), Jupyter (online notebook).

### CANNBot Agent

AI-assisted coding tool supporting Claude Code, Cursor, Trae — operator development 5× faster.

---

## Community & Learning

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Ascend (GitCode)** | Official repo collection | [gitcode.com/Ascend](https://gitcode.com/Ascend) |
| **CANN (GitCode)** | Core CANN open-source hub | [gitcode.com/cann](https://gitcode.com/cann) |
| **Ascend (Gitee)** | Some official repo mirrors | [gitee.com/ascend](https://gitee.com/ascend) |
| **HiAscend** | Official docs, SDK, tech articles, events | [hiascend.com](https://www.hiascend.com) |
| **cann-learning-hub** | 4-level learning path (beginner → expert) | [gitcode.com/cann/cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) |
| **cann-community** | CANN community governance & SIGs | [gitcode.com/cann/community](https://gitcode.com/cann/community) |
| **HiDevLab** | Free Ascend NPU compute (100 card-hours) | [hiascend.com](https://www.hiascend.com) |
| **Docker Base Image** | CANN pre-installed | `quay.io/ascend/cann:8.5.0-a3-openeuler24.03-py3.11` |

---

## Quick Start

```bash
# 1. Install PyTorch Ascend backend
pip install torch_npu

# 2. MindSpeed acceleration library
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && pip install -e .

# 3. MindSpeed-LLM training suite
git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM && pip install -e .

# 4. vLLM Ascend inference
pip install vllm-ascend

# 5. Reference recipes
git clone https://gitcode.com/cann/cann-recipes-infer.git   # inference
git clone https://gitcode.com/cann/cann-recipes-train.git   # training
```

---

## Training Framework Comparison

| Framework | Parallelism | Best For | Ascend Entry Point |
| :--- | :--- | :--- | :--- |
| [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) | FSDP2 / TP / CP / PP / EP | PyTorch-native, DeepSeek-V4 / Qwen3 continued training | `pip install -e .` |
| [MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) | TP / PP / DP / EP / CP + ZeRO | 100+ models end-to-end, Megatron ecosystem | `import mindspeed.megatron_adaptor` |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO-1/2/3/Infinity + Ulysses SP | ZeRO-Infinity CPU/NVMe offload, ultra-long sequences (1M+) | Native via `torch_npu` |
| PyTorch FSDP2 | FSDP2 + Hybrid Shard | Small-medium models (<70B), community standard | Native via `torch_npu` |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | PP / TP / VPP / CP / Encoder-DP | Multimodal (Video/Image/VL) | 19-24% faster than generic solutions |

### torchtitan-npu Benchmarks (Atlas 800T A3, 64 cards)

| Model | Precision | Parallel Strategy | Tokens/p/s | MFU |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V4-Flash | BF16 | FSDP128+EP128 | 1056 | 27.67% |
| DeepSeek-V3-671B | BF16 | FSDP32+TP4+EP128 | 546 | — |
| DeepSeek-V3-671B + compile(AutoFuse) | BF16 | FSDP32+TP4+EP128 | 576 | — |

---

## Scenario-Based Selection Guide

| Your Need | Recommended Solution | Why |
| :--- | :--- | :--- |
| Existing Megatron scripts, migrate to NPU | `MindSpeed` | One-line import adaptation |
| 7B–70B model pretraining / fine-tuning | `MindSpeed-LLM` or `torchtitan-npu` | Ready to use, complete toolchain |
| PyTorch-native training (FSDP2) | `torchtitan-npu` | Aligns with community standard, AutoFuse optimization |
| RLHF / GRPO post-training | `verl` + `vLLM-Ascend` or `MindSpeed-RL` | Training-inference integration, high throughput |
| Production LLM inference service | `vLLM-Ascend` or `SGLang` | Native community support, active ecosystem |
| Government/enterprise high-compliance inference | `MindIE-LLM` | Huawei native, mature private deployment |
| Multimodal training / inference | `MindSpeed-MM` + `MindIE-SD` | Full video/image/VL pipeline |
| Quickly build RAG / Agent apps | `Dify` + `LangChain` | Native images, low integration cost |
| Ultra-long sequences (32k–1M tokens) | `torchtitan-npu` or `DeepSpeed-Ulysses` | Mature long-sequence optimization |
| Recommendation / advertising large models | `RecSDK` / `fbgemm-ascend` | Specialized search/recommendation kernels |
| Edge / on-device deployment | `mllm` / lightweight `MindIE` | Low power, W8A8 quantization |

---

## FAQ

**Q1: I have a CUDA PyTorch model. How much code change is needed to migrate to Ascend?**

Usually just:
1. Install `torch_npu`;
2. Change `.to("cuda")` to `.to("npu")`;
3. Replace CUDA-specific ops (`torch.cuda` → `torch.npu`).

For complex models, use `MindSpeed` or `torchtitan-npu` for better performance.

**Q2: Why can HuggingFace models run on Ascend but with poor performance?**

HF + `torch_npu` provides "GPU-equivalent capability" without Ascend-specific kernel optimizations. For peak performance, use `MindSpeed-LLM`, `vLLM-Ascend`, or `SGLang`.

**Q3: How large a model can Ascend train?**

Verified:
- Dense models: 135B (Pangu Ultra, 8192 NPUs)
- MoE models: 718B (Pangu Ultra MoE, 10K+ NPUs), 1.6T (DeepSeek-V4, 910C cluster)

**Q4: Does QLoRA work on Ascend?**

Currently limited. `bitsandbytes` NPU support is incomplete; prefer Full / LoRA / Freeze strategies.

**Q5: Should I choose vLLM-Ascend or MindIE for inference deployment?**

- Open-source ecosystem, fast iteration: **vLLM-Ascend** or **SGLang**
- Enterprise private deployment, high compliance, official support: **MindIE-LLM**

**Q6: When will Ascend 950 be available?**

- 950PR: 2026 Q1
- 950DT: 2026 Q4 (some cloud providers may offer earlier)

---

## Contributing

PRs welcome! Please add new Ascend NPU projects, frameworks, and resources.
