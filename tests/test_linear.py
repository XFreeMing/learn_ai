"""Tests for 01_linear_y_wx_b.py — Linear layer forward, backward, gradient check."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """Load a module whose path contains digits (not valid Python identifiers)."""
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


linear_mod = _load("01_handwritten_nn/01_linear_y_wx_b.py")
Linear = linear_mod.Linear
mse_loss = linear_mod.mse_loss
numerical_gradient = linear_mod.numerical_gradient


class TestLinearInit:
    def test_weight_shape(self):
        layer = Linear(3, 5, seed=42)
        assert layer.W.shape == (3, 5)
        assert layer.B.shape == (5,)

    def test_bias_is_zero(self):
        layer = Linear(3, 5, seed=42)
        assert np.allclose(layer.B, 0)

    def test_reproducible_seed(self):
        a = Linear(4, 2, seed=7)
        b = Linear(4, 2, seed=7)
        assert np.array_equal(a.W, b.W)
        assert np.array_equal(a.B, b.B)

    def test_different_seed_different_weights(self):
        a = Linear(4, 2, seed=1)
        b = Linear(4, 2, seed=2)
        assert not np.array_equal(a.W, b.W)


class TestLinearForward:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        layer = Linear(3, 2)
        X = rng.normal(size=(4, 3))
        assert layer.forward(X).shape == (4, 2)

    def test_identity_mapping(self):
        """当 W=I, B=0 时，Y = X。"""
        layer = Linear(3, 3, seed=0)
        layer.W = np.eye(3)
        layer.B = np.zeros(3)
        X = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(layer.forward(X), X)

    def test_bias_addition(self):
        """当 W=0, B=b 时，Y = b（每个样本相同）。"""
        layer = Linear(3, 2, seed=0)
        layer.W = np.zeros((3, 2))
        layer.B = np.array([1.0, -1.0])
        X = np.ones((5, 3))
        out = layer.forward(X)
        assert np.allclose(out, [[1.0, -1.0]] * 5)


class TestLinearBackward:
    def test_gradient_shapes(self):
        rng = np.random.default_rng(0)
        layer = Linear(3, 2)
        X = rng.normal(size=(4, 3))
        layer.forward(X)
        dY = rng.normal(size=(4, 2))
        dX = layer.backward(dY)
        assert dX.shape == (4, 3)
        assert layer.dW.shape == (3, 2)
        assert layer.dB.shape == (2,)

    def test_zero_input_zero_dY(self):
        layer = Linear(3, 2, seed=0)
        X = np.zeros((2, 3))
        layer.forward(X)
        dY = np.zeros((2, 2))
        dX = layer.backward(dY)
        assert np.allclose(dX, 0)
        assert np.allclose(layer.dW, 0)
        assert np.allclose(layer.dB, 0)


class TestGradientCheck:
    def test_numerical_vs_analytical(self):
        """用数值梯度校验手写反向传播，误差 < 1e-5。"""
        rng = np.random.default_rng(42)
        N, in_f, out_f = 4, 3, 2
        X = rng.normal(size=(N, in_f))
        target = rng.normal(size=(N, out_f))
        layer = Linear(in_f, out_f)

        pred = layer.forward(X)
        loss, dY = mse_loss(pred, target)
        layer.backward(dY)

        def loss_only():
            return mse_loss(layer.forward(X), target)[0]

        num_dW = numerical_gradient(loss_only, layer.W)
        num_dB = numerical_gradient(loss_only, layer.B)

        err_W = np.max(np.abs(num_dW - layer.dW))
        err_B = np.max(np.abs(num_dB - layer.dB))
        assert err_W < 1e-5, f"dW error {err_W:.2e} exceeds threshold"
        assert err_B < 1e-5, f"dB error {err_B:.2e} exceeds threshold"
