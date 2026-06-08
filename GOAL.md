# GOAL

> 在 AI 时代，能写代码的人遍地都是，真正理解代码背后原理的人才稀缺。
> 本项目的唯一目标：把"理解原理"变成可演示的硬实力。

## 北极星目标

面试时能现场做到这三件事，而不是只会背概念：

1. **手写神经网络** — 不依赖框架封装，亲手实现 `Y = WX + B`、反向传播，再用 PyTorch 搭 3–5 层小网络并训练收敛。
2. **理解推理原理** — 讲清训练 vs 推理的本质差异，并能量化 **吞吐量（throughput）** 与 **响应延迟（latency）** 的取舍。
3. **熟悉推理引擎** — 真正跑通 vLLM / TensorRT-LLM 等引擎的示例，能解释 KV Cache、PagedAttention、continuous batching 为什么有效。

## 成功判定（Definition of Done）

每个模块只有满足下面条件才算"掌握"，否则不算：

- [ ] M1：手写 NN 在不调用 `torch.nn.Linear` 的情况下能在玩具数据上 loss 持续下降；PyTorch 版准确率可复现。
- [ ] M2：能产出一张 throughput–latency 权衡曲线图，并用自己的话解释拐点。
- [ ] M3：本地或云端跑通一个推理引擎 demo，记录 QPS / TTFT / TPOT 三个指标。

## 工作原则（来自 RULES）

- **Rule 13 真结果或没发生**：每个结论都要有运行输出佐证，不靠"看起来对"。
- **Rule 16 一次一个函数一个测试**：逐个实现、逐个验证，不堆叠改动。
- **Rule 14 代码即文档**：markdown 只做索引和关系图，细节写进代码。

## 阶段路线

详见 [ROADMAP.md](ROADMAP.md)。
