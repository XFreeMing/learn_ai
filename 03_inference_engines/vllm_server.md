# vLLM OpenAI 兼容 Server + 压测

> 环境：Linux / WSL2 + NVIDIA GPU。

## 1. 启动 server

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --port 8000
```

启动后会暴露一个 OpenAI 兼容接口（`/v1/completions`、`/v1/chat/completions`）。

## 2. 发一个请求

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "The future of AI inference is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

## 3. 压测吞吐 / 延迟

vLLM 自带 benchmark 脚本（在其仓库 `benchmarks/` 下），或用 `vllm bench`：

```bash
# 用官方 benchmark 客户端并发打流量，观察 TTFT / TPOT / throughput
vllm bench serve \
    --model facebook/opt-125m \
    --num-prompts 200 \
    --request-rate 10
```

记录三个指标：

| 指标 | 来源 |
| --- | --- |
| Throughput (tokens/s) | benchmark 汇总 |
| TTFT (P50/P99) | benchmark 汇总 |
| TPOT (P50/P99) | benchmark 汇总 |

## 4. 该理解什么（面试要点）

- **Continuous batching**：请求一旦完成立刻让出 slot，新请求即时填入，不必等整批结束 → GPU 不空转。
- **PagedAttention**：把 KV Cache 切成固定大小的"页"，按需分配，减少显存碎片 → 同样显存能放更多并发序列。
- **结果**：相同硬件下，vLLM 的吞吐通常是朴素 `transformers.generate` 的数倍。

## 5. 没有 GPU 怎么办

- 用云 GPU：Google Colab（免费 T4）、AutoDL、RunPod、Lambda。
- 在 Colab 里 `!pip install vllm` 后直接跑 [vllm_quickstart.py](vllm_quickstart.py) 的逻辑。
