"""Tests for 02_inference_principles/benchmark_throughput_vs_latency.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("bench", ROOT / "02_inference_principles/benchmark_throughput_vs_latency.py")
bench_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_mod)

TinyBlock = bench_mod.TinyBlock
benchmark = bench_mod.benchmark


class TestTinyBlock:
    def test_output_shape(self):
        model = TinyBlock(dim=64, depth=2)
        x = torch.randn(8, 64)
        assert model(x).shape == (8, 64)

    def test_single_sample(self):
        model = TinyBlock(dim=128, depth=1)
        x = torch.randn(1, 128)
        assert model(x).shape == (1, 128)


class TestBenchmark:
    def test_throughput_increases_with_batch(self):
        """随 batch 增大，总吞吐量（req/s）应上升或持平。"""
        model = TinyBlock(dim=256, depth=2)
        batch_sizes = [1, 2, 4, 8, 16]
        results = benchmark(model, batch_sizes, dim=256, repeats=3, warmup=1)

        # 提取吞吐量
        thrs = [r[2] for r in results]
        # 至少前几个 batch 的吞吐应该递增
        assert thrs[-1] >= thrs[0], (
            f"throughput decreased from bs=1 ({thrs[0]:.1f}) to bs=16 ({thrs[-1]:.1f})"
        )

    def test_result_format(self):
        """每条结果应为 (batch_size, batch_latency_ms, throughput_req_per_s)。"""
        model = TinyBlock(dim=128, depth=1)
        results = benchmark(model, [1, 2], dim=128, repeats=2, warmup=1)
        assert len(results) == 2
        for bs, lat, thr in results:
            assert isinstance(bs, int) and bs > 0
            assert lat > 0
            assert thr > 0
