"""M2: 吞吐量 vs 延迟 权衡基准实验。

用一个小 Transformer-ish 前向（或任意固定模型）模拟推理，
扫描不同 batch size，测量:
    - per-request latency: 单个请求平均耗时
    - throughput: 每秒处理的请求数 (req/s)
观察经典权衡: batch 越大吞吐越高，但单请求延迟越高。

无需 GPU，CPU 即可跑出趋势。生成 tradeoff.png（若装了 matplotlib）。
"""

import time

import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    """模拟一层推理计算的小模型（线性 + 注意力近似）。"""

    def __init__(self, dim=512, depth=4):
        super().__init__()
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(depth)]
        )

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


def benchmark(model, batch_sizes, dim=512, repeats=20, warmup=3):
    results = []
    for bs in batch_sizes:
        x = torch.randn(bs, dim)
        # 预热
        for _ in range(warmup):
            model(x)
        # 计时
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = time.perf_counter() - t0

        per_batch = elapsed / repeats          # 一个 batch 的耗时
        per_request = per_batch / bs           # 单请求延迟
        throughput = (bs * repeats) / elapsed  # req/s
        results.append((bs, per_request * 1000, throughput))
    return results


def main():
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = TinyBlock().eval()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = benchmark(model, batch_sizes)

    print(f"{'batch':>6} | {'per-req latency (ms)':>22} | {'throughput (req/s)':>20}")
    print("-" * 56)
    for bs, lat_ms, thr in results:
        print(f"{bs:>6} | {lat_ms:>22.3f} | {thr:>20.1f}")

    # 找甜点：吞吐增益明显放缓的拐点
    print("\n观察：随 batch 增大，throughput 上升但 per-request latency 也上升。")
    print("甜点 = 在可接受延迟内吞吐最高的那个 batch size。")

    # 可选画图
    try:
        import matplotlib.pyplot as plt

        bss = [r[0] for r in results]
        lats = [r[1] for r in results]
        thrs = [r[2] for r in results]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.set_xlabel("batch size")
        ax1.set_ylabel("per-request latency (ms)", color="tab:red")
        ax1.plot(bss, lats, "o-", color="tab:red", label="latency")
        ax1.set_xscale("log", base=2)
        ax2 = ax1.twinx()
        ax2.set_ylabel("throughput (req/s)", color="tab:blue")
        ax2.plot(bss, thrs, "s-", color="tab:blue", label="throughput")
        plt.title("Throughput vs Latency trade-off")
        fig.tight_layout()
        fig.savefig("tradeoff.png", dpi=120)
        print("\n已保存权衡曲线: tradeoff.png")
    except ImportError:
        print("\n(未安装 matplotlib，跳过画图；pip install matplotlib 后可生成 tradeoff.png)")


if __name__ == "__main__":
    main()
