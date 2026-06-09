"""Tests for 02_mlp_numpy.py — activations, BCE loss, data generation, training loop."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("mlp", ROOT / "01_handwritten_nn/02_mlp_numpy.py")
mlp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_mod)

relu = mlp_mod.relu
relu_grad = mlp_mod.relu_grad
sigmoid = mlp_mod.sigmoid
bce_loss = mlp_mod.bce_loss
make_xor_data = mlp_mod.make_xor_data


class TestReLU:
    def test_positive_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(relu(x), x)

    def test_negative_zeroed(self):
        x = np.array([-1.0, -2.0, -3.0])
        assert np.array_equal(relu(x), np.zeros(3))

    def test_mixed(self):
        x = np.array([-1.0, 0.0, 1.0])
        assert np.array_equal(relu(x), [0.0, 0.0, 1.0])

    def test_2d(self):
        x = np.array([[-1, 2], [3, -4]])
        assert np.array_equal(relu(x), [[0, 2], [3, 0]])


class TestReLUGrad:
    def test_positive_ones(self):
        x = np.array([1.0, 2.0])
        assert np.array_equal(relu_grad(x), np.ones(2))

    def test_negative_zeros(self):
        x = np.array([-1.0, -2.0])
        assert np.array_equal(relu_grad(x), np.zeros(2))

    def test_zero_is_zero(self):
        x = np.array([0.0])
        assert relu_grad(x)[0] == 0


class TestSigmoid:
    def test_zero_is_half(self):
        assert np.isclose(sigmoid(0.0), 0.5)

    def test_range(self):
        x = np.array([-100.0, 0.0, 100.0])
        s = sigmoid(x)
        # sigmoid 输出严格在 [0, 1] 区间内（浮点极端输入可能等于 0 或 1）
        assert np.all(s >= 0) and np.all(s <= 1)
        assert np.isclose(s[0], 0.0, atol=1e-10)
        assert np.isclose(s[2], 1.0, atol=1e-10)


class TestBCELoss:
    def test_perfect_prediction_zero_loss(self):
        p = np.array([[1.0], [0.0]])
        y = np.array([[1.0], [0.0]])
        loss, _ = bce_loss(p, y)
        assert loss < 1e-6

    def test_worst_prediction_high_loss(self):
        p = np.array([[0.0], [1.0]])
        y = np.array([[1.0], [0.0]])
        loss, _ = bce_loss(p, y)
        assert loss > 10  # very high loss

    def test_gradient_direction(self):
        """预测过高时梯度应为正（推动降低预测），过低时为负。"""
        p = np.array([[0.9]])
        y = np.array([[0.0]])
        loss, grad = bce_loss(p, y)
        assert grad[0, 0] > 0  # push prediction down

        p = np.array([[0.1]])
        y = np.array([[1.0]])
        loss, grad = bce_loss(p, y)
        assert grad[0, 0] < 0  # push prediction up


class TestMakeXORData:
    def test_shapes(self):
        X, y = make_xor_data(n=100)
        assert X.shape == (100, 2)
        assert y.shape == (100, 1)

    def test_balanced(self):
        X, y = make_xor_data(n=1000, seed=0)
        ratio = y.mean()
        assert 0.4 < ratio < 0.6  # roughly half positive

    def test_reproducible(self):
        _, y1 = make_xor_data(n=50, seed=42)
        _, y2 = make_xor_data(n=50, seed=42)
        assert np.array_equal(y1, y2)


class TestTrainingLoop:
    def test_loss_decreases(self):
        """训练 200 epoch 后 loss 应显著低于初始值。"""
        rng = np.random.default_rng(1)
        X, y = make_xor_data()

        W1 = rng.normal(0, 0.5, size=(2, 16))
        b1 = np.zeros(16)
        W2 = rng.normal(0, 0.5, size=(16, 1))
        b2 = np.zeros(1)
        lr = 0.1

        # 记录初始 loss
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        p_init = sigmoid(z2)
        initial_loss, _ = bce_loss(p_init, y)

        # 训练 200 步
        for _ in range(200):
            z1 = X @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            p = sigmoid(z2)
            _, dp = bce_loss(p, y)
            dz2 = dp * (p * (1 - p))
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * relu_grad(z1)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        p_final = sigmoid(z2)
        final_loss, _ = bce_loss(p_final, y)

        assert final_loss < initial_loss * 0.5, (
            f"loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_accuracy_improves(self):
        """训练后准确率应显著高于随机（> 0.7）。"""
        rng = np.random.default_rng(1)
        X, y = make_xor_data()

        W1 = rng.normal(0, 0.5, size=(2, 16))
        b1 = np.zeros(16)
        W2 = rng.normal(0, 0.5, size=(16, 1))
        b2 = np.zeros(1)
        lr = 0.1

        for _ in range(500):
            z1 = X @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            p = sigmoid(z2)
            _, dp = bce_loss(p, y)
            dz2 = dp * (p * (1 - p))
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * relu_grad(z1)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        acc = ((p > 0.5) == y).mean()
        assert acc > 0.7, f"accuracy too low: {acc:.3f}"
