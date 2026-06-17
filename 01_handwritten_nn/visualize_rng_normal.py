"""可视化 rng.normal(loc, scale, size) 中各参数的作用。

运行后会生成一张图 rng_normal_demo.png，包含三块对比：
  1. 改变 loc（均值）→ 分布左右平移
  2. 改变 scale（标准差）→ 分布变胖/变瘦
  3. 改变 size → 同一规律下采样点数量不同（直方图越来越接近理论曲线）

直观回答：这几个参数分别控制「中心在哪、有多分散、生成多少个」。
"""

import numpy as np
import matplotlib.pyplot as plt

# 让中文标题正常显示（macOS 常见可用字体）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """正态分布概率密度函数，用于画理论曲线。"""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def main():
    rng = np.random.default_rng(0)
    n = 20000  # 采样点数量，越多直方图越接近理论曲线

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- 子图 1：改变 loc（均值），固定 scale=1 ----
    ax = axes[0]
    for loc in (-3, 0, 3):
        samples = rng.normal(loc, 1.0, size=n)
        ax.hist(samples, bins=80, density=True, alpha=0.4, label=f"loc={loc}")
        xs = np.linspace(loc - 4, loc + 4, 200)
        ax.plot(xs, gaussian_pdf(xs, loc, 1.0), linewidth=2)
    ax.set_title("改变 loc(均值): 分布左右平移")
    ax.set_xlabel("采样值")
    ax.set_ylabel("概率密度")
    ax.legend()

    # ---- 子图 2：改变 scale（标准差），固定 loc=0 ----
    ax = axes[1]
    for scale in (0.1, 0.5, 1.0, 2.0):
        samples = rng.normal(0.0, scale, size=n)
        ax.hist(samples, bins=120, density=True, alpha=0.35, label=f"scale={scale}")
        xs = np.linspace(-6, 6, 300)
        ax.plot(xs, gaussian_pdf(xs, 0.0, scale), linewidth=2)
    ax.set_title("改变 scale(标准差): 越大越分散(越胖)")
    ax.set_xlabel("采样值")
    ax.set_xlim(-6, 6)
    ax.legend()

    # ---- 子图 3：改变 size（采样数量），固定 loc=0, scale=0.1 ----
    ax = axes[2]
    xs = np.linspace(-0.4, 0.4, 300)
    for count, alpha in ((50, 0.5), (500, 0.45), (50000, 0.4)):
        samples = rng.normal(0.0, 0.1, size=count)
        ax.hist(samples, bins=40, density=True, alpha=alpha, label=f"size={count}")
    ax.plot(xs, gaussian_pdf(xs, 0.0, 0.1), "k--", linewidth=2, label="理论曲线 σ=0.1")
    ax.set_title("改变 size: 采样越多越逼近理论分布")
    ax.set_xlabel("采样值")
    ax.legend()

    fig.suptitle(
        "rng.normal(loc, scale, size): loc=中心  scale=分散度  size=数量/形状",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = "rng_normal_demo.png"
    fig.savefig(out, dpi=120)
    print(f"已保存图像: {out}")

    # ---- 顺便用文字打印「权重初始化」场景的直观差异 ----
    print("\n不同 scale 下，模拟权重矩阵 (3x2) 的数值范围:")
    for scale in (0.1, 1.0):
        w = rng.normal(0.0, scale, size=(3, 2))
        print(f"  scale={scale}: 取值约 [{w.min():+.3f}, {w.max():+.3f}]\n{w}\n")


if __name__ == "__main__":
    main()
