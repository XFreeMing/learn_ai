# learn_ai · 硅谷 AI 面试三项硬通货训练营

> 在 AI 时代，能写代码的人遍地都是，**真正理解代码背后原理的人才稀缺。**
> 本仓库把这句话拆成三个可演示的能力，逐个动手攻克。

## 目标

详见 [GOAL.md](GOAL.md) · 计划详见 [ROADMAP.md](ROADMAP.md)

| 能力 | 目录 | 一句话验收 |
| --- | --- | --- |
| ① 手写神经网络 | [01_handwritten_nn/](01_handwritten_nn/) | 不靠 `nn.Linear` 实现 `Y=WX+B` 并训练收敛 |
| ② 理解推理原理 | [02_inference_principles/](02_inference_principles/) | 画出吞吐↔延迟权衡曲线并解释拐点 |
| ③ 熟悉推理引擎 | [03_inference_engines/](03_inference_engines/) | 跑通 vLLM demo 并记录 QPS/TTFT |

## 快速开始

```pwsh
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装基础依赖（M1 + M2 在 CPU 即可跑）
pip install -r requirements.txt

# 3. 按顺序动手
python 01_handwritten_nn\01_linear_y_wx_b.py
python 01_handwritten_nn\02_mlp_numpy.py
python 01_handwritten_nn\03_mlp_pytorch.py
python 02_inference_principles\benchmark_throughput_vs_latency.py
python 03_inference_engines\baseline_transformers.py
# vLLM 需 Linux/WSL + GPU，见 03_inference_engines/README.md
```

## 周末冲刺

照着 [ROADMAP.md](ROADMAP.md) 的"周末冲刺"表格走一遍，两天打完地基。

## 工作纪律（贯穿全程）

- **真结果或没发生**：每个结论都要有运行输出，不靠"看起来对"。
- **一次一个函数一个测试**：逐个实现、逐个验证。
- **代码即文档**：markdown 只做索引，细节在代码里。
