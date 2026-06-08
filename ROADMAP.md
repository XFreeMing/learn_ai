# ROADMAP — 三项硬通货能力训练计划

整体节奏：**1 个周末打地基 + 之后每周 2–3 个晚上深化**。
每个模块都给出"做什么 / 怎么验证 / 面试怎么讲"。

---

## 周末冲刺（Day 0–1）：先把地基砸实

| 时段 | 任务 | 产出 |
| --- | --- | --- |
| 周六上午 | M1 手写线性层 + 手写反向传播 | `01_handwritten_nn/01_linear_y_wx_b.py` 跑通 |
| 周六下午 | M1 用纯 numpy 实现 2 层 MLP 并训练 | loss 曲线下降 |
| 周六晚上 | M1 用 PyTorch 重写 3–5 层 MLP | 在 MNIST/玩具集收敛 |
| 周日上午 | M2 训练 vs 推理差异笔记 + KV Cache 原理 | `02_inference_principles/notes.md` |
| 周日下午 | M2 跑吞吐量 vs 延迟基准实验 | 权衡曲线图 |
| 周日晚上 | M3 安装一个推理引擎并跑通 hello world | 记录 3 个指标 |

---

## M1 · 手写神经网络（基础运算 → PyTorch 小网络）

**做什么**
1. 手写 `Y = WX + B` 前向（numpy）。
2. 手写 MSE/交叉熵 loss 与梯度，自己推导 `dL/dW`、`dL/dB`。
3. 手写 2 层 MLP（含一次反向传播闭环），在玩具数据上训练。
4. 用 PyTorch 重写 3–5 层 MLP，训练 MNIST 或 `sklearn` 玩具集。

**怎么验证**
- 纯手写版 loss 单调下降（允许抖动）。
- 用 `torch.autograd` 数值梯度校验手写梯度，误差 < 1e-5。

**面试怎么讲**
- "我能不靠 `nn.Linear` 实现一层全连接，并解释反向传播每一步的链式法则。"

> 代码目录：[01_handwritten_nn/](01_handwritten_nn/)

---

## M2 · 理解推理原理（throughput ↔ latency）

**做什么**
1. 写清楚训练 vs 推理差异：梯度、batch 来源、显存占用、KV Cache。
2. 理解核心指标：**TTFT**（首 token 延迟）、**TPOT**（每 token 延迟）、**Throughput**（tokens/s）。
3. 做一个基准实验：固定模型，改变 batch size，画出 throughput 与 per-request latency 的权衡曲线。

**怎么验证**
- 能解释为什么增大 batch 提升吞吐但抬高单请求延迟。
- 能指出曲线拐点对应的"甜点"batch size。

**面试怎么讲**
- "在线服务我会盯 TTFT 和 P99 延迟；离线批处理我会最大化 throughput。两者通过 batching 策略权衡。"

> 代码目录：[02_inference_principles/](02_inference_principles/)

---

## M3 · 熟悉推理引擎（vLLM / TensorRT-LLM）

**做什么**
1. 选一个引擎（推荐先 vLLM，安装门槛低）。
2. 跑通官方 hello-world：离线批量推理 + OpenAI 兼容 server。
3. 读懂三个关键机制：**PagedAttention**、**continuous batching**、**KV Cache 复用**。
4. 用 M2 的指标脚本压测引擎，对比朴素 `transformers.generate`。

**怎么验证**
- 引擎 demo 能返回结果。
- 记录引擎 vs 朴素实现的 QPS / TTFT 差异。

**面试怎么讲**
- "我实际用 vLLM 跑过服务，知道 PagedAttention 通过分页管理 KV Cache 降低显存碎片，从而支撑更大并发。"

> 代码目录：[03_inference_engines/](03_inference_engines/)

---

## 进度看板

在每个模块 README 勾选 checklist，全部勾完即达成 [GOAL.md](GOAL.md) 的 Definition of Done。
