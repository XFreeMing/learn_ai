"""M1-02: 纯 numpy 实现 2 层 MLP，完成一次完整训练闭环。

结构: Linear(2->16) -> ReLU -> Linear(16->1) -> Sigmoid -> BCE loss
数据: 二维同心圆/异或式玩具二分类。
看到 loss 持续下降即代表前向+反向+梯度下降闭环正确。
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(p, y, eps=1e-7):
    """二元交叉熵及其对 logits 之前 sigmoid 输出 p 的梯度。"""
    p = np.clip(p, eps, 1 - eps)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = (p - y) / (p * (1 - p)) / y.shape[0]
    return loss, grad


def make_xor_data(n=400, seed=0):
    """生成 XOR 样式的非线性可分数据，必须靠隐藏层才能学会。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64).reshape(-1, 1)
    return X, y


def main():
    rng = np.random.default_rng(1)
    X, y = make_xor_data()

    # 参数
    W1 = rng.normal(0, 0.5, size=(2, 16))
    b1 = np.zeros(16)
    W2 = rng.normal(0, 0.5, size=(16, 1))
    b2 = np.zeros(1)
    lr = 0.1

    for epoch in range(2001):
        # ---- 前向 ----
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        p = sigmoid(z2)

        loss, dp = bce_loss(p, y)

        # ---- 反向 ----
        dz2 = dp * (p * (1 - p))        # 通过 sigmoid
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)       # 通过 ReLU
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # ---- 梯度下降 ----
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        if epoch % 400 == 0:
            acc = ((p > 0.5) == y).mean()
            print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
