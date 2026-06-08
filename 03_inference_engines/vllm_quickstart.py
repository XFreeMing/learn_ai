"""M3: vLLM 离线批量推理快速上手。

环境要求: Linux / WSL2 + NVIDIA GPU。
安装: pip install vllm

vLLM 的 LLM.generate 会自动对这一批 prompts 做 continuous batching，
配合 PagedAttention 管理 KV Cache，吞吐通常远高于逐请求的 transformers baseline。
跑完用 total tokens / wall time 估算吞吐，与 baseline_transformers.py 对比。
"""

import time

# 注意：import vllm 仅在已安装 vllm 的环境可用
from vllm import LLM, SamplingParams

MODEL = "facebook/opt-125m"  # 小模型，方便快速验证；可换成任意 HF causal LM

PROMPTS = [
    "The future of AI inference is",
    "PagedAttention works by",
    "In one sentence, a neural network is",
    "Throughput and latency trade off because",
] * 8  # 复制成 32 条，凸显 batching 优势


def main():
    llm = LLM(model=MODEL)  # 首次会下载权重
    params = SamplingParams(temperature=0.0, max_tokens=32)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print("=== vLLM (continuous batching) ===")
    print(f"requests        = {len(PROMPTS)}")
    print(f"total new tokens= {total_tokens}")
    print(f"wall time       = {elapsed:.2f} s")
    print(f"throughput      = {total_tokens / elapsed:.1f} tokens/s")

    # 展示前两条结果
    for o in outputs[:2]:
        print(f"---\nPROMPT: {o.prompt}\nGEN: {o.outputs[0].text}")


if __name__ == "__main__":
    main()
