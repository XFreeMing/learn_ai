"""M3 baseline: 用 HuggingFace transformers 做朴素推理，作为 vLLM 的对照组。

特点：逐请求 generate，没有 continuous batching / PagedAttention。
目的：先建立 baseline 直觉，再用 vLLM 对比体会优化收益。

CPU 即可运行（用很小的模型，如 sshleifer/tiny-gpt2 或 distilgpt2）。
"""

import time

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"  # 小模型，CPU 友好；有 GPU 可换更大模型

PROMPTS = [
    "The future of AI inference is",
    "PagedAttention works by",
    "In one sentence, a neural network is",
    "Throughput and latency trade off because",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.perf_counter()
    total_new_tokens = 0
    for p in PROMPTS:
        inputs = tok(p, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        total_new_tokens += new_tokens
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"---\n{text}")

    elapsed = time.perf_counter() - t0
    print("\n=== baseline (transformers, 逐请求) ===")
    print(f"requests        = {len(PROMPTS)}")
    print(f"total new tokens= {total_new_tokens}")
    print(f"wall time       = {elapsed:.2f} s")
    print(f"throughput      = {total_new_tokens / elapsed:.1f} tokens/s")
    print("\n对比项：vLLM 会把这些请求做 continuous batching，吞吐通常显著更高。")


if __name__ == "__main__":
    main()
