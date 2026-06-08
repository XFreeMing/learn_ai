"""M1-01: 纯 numpy 手写线性层 Y = WX + B。

不调用任何深度学习框架，手写前向、反向，并用数值梯度校验正确性。
运行后看到 "gradient check passed" 即代表反向传播推导正确。
"""

import numpy as np


class Linear:
    """全连接层：Y = X @ W + B。

    形状约定:
        X: (N, in_features)
        W: (in_features, out_features)
        B: (out_features,)
        Y: (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # 小随机初始化，避免一开始就饱和
        self.W = rng.normal(0, 0.1, size=(in_features, out_features))
        self.B = np.zeros(out_features)
        self._cache_X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache_X = X
        return X @ self.W + self.B

    def backward(self, dY: np.ndarray):
        """给定上游梯度 dL/dY，返回 dL/dX，并计算参数梯度。

        链式法则:
            dL/dW = X^T @ dL/dY
            dL/dB = sum over batch of dL/dY
            dL/dX = dL/dY @ W^T
        """
        X = self._cache_X
        self.dW = X.T @ dY
        self.dB = dY.sum(axis=0)
        dX = dY @ self.W.T
        return dX


def mse_loss(pred: np.ndarray, target: np.ndarray):
    """均方误差及其对 pred 的梯度。"""
    diff = pred - target
    loss = np.mean(diff**2)
    # d(mean(diff^2))/d(pred) = 2*diff / N_total
    grad = 2 * diff / diff.size
    return loss, grad


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分法估计 f 对 x 的梯度，用于校验手写反向传播。"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fpos = f()
        x[idx] = orig - eps
        fneg = f()
        x[idx] = orig
        grad[idx] = (fpos - fneg) / (2 * eps)
        it.iternext()
    return grad


def main():
    rng = np.random.default_rng(42)
    N, in_f, out_f = 4, 3, 2
    X = rng.normal(size=(N, in_f))
    target = rng.normal(size=(N, out_f))

    layer = Linear(in_f, out_f)

    # 前向 + 反向（解析梯度）
    pred = layer.forward(X)
    loss, dY = mse_loss(pred, target)
    layer.backward(dY)

    # 数值梯度校验 dW
    def loss_only():
        return mse_loss(layer.forward(X), target)[0]

    num_dW = numerical_gradient(loss_only, layer.W)
    num_dB = numerical_gradient(loss_only, layer.B)

    err_W = np.max(np.abs(num_dW - layer.dW))
    err_B = np.max(np.abs(num_dB - layer.dB))
    print(f"initial loss      = {loss:.6f}")
    print(f"max |dW err|      = {err_W:.2e}")
    print(f"max |dB err|      = {err_B:.2e}")

    assert err_W < 1e-5 and err_B < 1e-5, "gradient check FAILED"
    print("gradient check passed ✅  (反向传播推导正确)")


if __name__ == "__main__":
    main()
