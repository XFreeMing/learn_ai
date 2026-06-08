# M2 · 理解推理原理

目标：讲清训练 vs 推理差异，并能量化 **吞吐量 ↔ 延迟** 的取舍。

## 文件索引

| 文件 | 内容 |
| --- | --- |
| [notes.md](notes.md) | 训练 vs 推理、KV Cache、核心指标定义（面试速记） |
| [benchmark_throughput_vs_latency.py](benchmark_throughput_vs_latency.py) | 用一个小模型模拟不同 batch size 下的吞吐/延迟权衡曲线 |

## 运行

```pwsh
python benchmark_throughput_vs_latency.py   # 打印权衡表，并生成 tradeoff.png
```

## Definition of Done

- [ ] 能口述 TTFT / TPOT / Throughput 三个指标定义
- [ ] 产出 throughput–latency 权衡曲线，并指出甜点 batch size
- [ ] 能解释"为什么 batch 越大吞吐越高但单请求越慢"
