# M3 · 熟悉推理引擎

目标：真正跑通一个推理引擎示例，理解其工作原理，而非停留在概念。

## ⚠️ 平台说明

- **vLLM 官方支持 Linux + NVIDIA GPU**，在原生 Windows 上较难直接安装。
- 在 Windows 上的可行路径（任选其一）：
  1. **WSL2 + Ubuntu**（推荐，本地有 NVIDIA GPU 时）。
  2. **云 GPU**（Colab / AutoDL / RunPod / Lambda），租一张卡跑 vLLM。
  3. **先用 transformers 跑通流程**（CPU 也能跑小模型），理解 baseline，再上 vLLM 对比。

## 文件索引

| 文件 | 内容 | 运行环境 |
| --- | --- | --- |
| [baseline_transformers.py](baseline_transformers.py) | HuggingFace `transformers.generate` 朴素推理 + 计时 | 任意（CPU 可跑小模型） |
| [vllm_quickstart.py](vllm_quickstart.py) | vLLM 离线批量推理 + 指标 | Linux/WSL + GPU |
| [vllm_server.md](vllm_server.md) | 启动 OpenAI 兼容 server 并压测 | Linux/WSL + GPU |

## 学习路径

1. 先跑 `baseline_transformers.py`，建立"朴素逐请求生成"的基准印象。
2. 在 GPU 环境装 vLLM，跑 `vllm_quickstart.py`，对比同样 prompts 的总耗时/吞吐。
3. 读 [vllm_server.md](vllm_server.md) 起 server，用 M2 的指标概念压测。
4. 在 notes 里写下：vLLM 比 baseline 快在哪？（continuous batching + PagedAttention）

## Definition of Done

- [ ] baseline 跑通并记录耗时
- [ ] vLLM demo 跑通并记录 QPS / TTFT
- [ ] 能用自己的话解释 PagedAttention 与 continuous batching 为什么提升吞吐
