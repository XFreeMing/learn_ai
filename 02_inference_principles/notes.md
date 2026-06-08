# 推理原理速记（面试可直接讲）

## 1. 训练 vs 推理：本质差异

| 维度 | 训练 (Training) | 推理 (Inference) |
| --- | --- | --- |
| 目标 | 更新权重，最小化 loss | 固定权重，生成输出 |
| 反向传播 | 需要（计算梯度） | 不需要 |
| 显存大头 | 权重 + 梯度 + 优化器状态 + 激活 | 权重 + **KV Cache** |
| 计算特征 | 大 batch、吞吐导向 | 自回归、延迟敏感 |
| 数据来源 | 静态数据集 | 用户实时请求 |

一句话：**训练是"学"，推理是"用"；推理没有梯度，瓶颈从算力转向显存带宽与 KV Cache。**

## 2. 自回归推理的两个阶段

1. **Prefill（预填充）**：一次性并行处理整个 prompt，计算密集（compute-bound）。决定 **TTFT**。
2. **Decode（解码）**：逐 token 生成，每步只算 1 个 token，访存密集（memory-bound）。决定 **TPOT**。

## 3. KV Cache 是什么、为什么重要

- 自回归生成时，每个新 token 都要和**之前所有 token**做注意力。
- 若不缓存，每步都要重算历史 K/V → O(n²) 浪费。
- KV Cache 把历史的 Key/Value 存下来，每步只算新 token → 把解码降到 O(n)。
- 代价：显存随 **序列长度 × batch × 层数 × 头数** 线性增长 → 显存常是并发瓶颈。
- vLLM 的 **PagedAttention** 用"分页"管理 KV Cache，减少碎片，从而支撑更高并发。

## 4. 三个核心指标

| 指标 | 含义 | 谁在乎 |
| --- | --- | --- |
| **TTFT** (Time To First Token) | 从请求到首个 token 的延迟 | 在线聊天体验 |
| **TPOT** (Time Per Output Token) | 生成每个后续 token 的平均耗时 | 流式输出流畅度 |
| **Throughput** | 系统每秒总 token 数 (tokens/s) | 成本 / 离线批处理 |

端到端延迟 ≈ `TTFT + TPOT × 输出长度`。

## 5. 吞吐量 ↔ 延迟的取舍（核心考点）

- **增大 batch size**：GPU 利用率↑ → 总吞吐↑，但每个请求要等更多同伴一起算 → **单请求延迟↑**。
- **减小 batch size**：单请求延迟↓，但 GPU 空转 → 吞吐↓、成本↑。
- 因此存在一个**甜点 batch size**：在满足 P99 延迟 SLA 的前提下，最大化吞吐。
- 工程手段：
  - **Continuous batching**（vLLM）：请求完成即换入新请求，不必等整批结束 → 同时改善吞吐和延迟。
  - **分场景部署**：在线服务限制 batch 保延迟；离线批处理放大 batch 保吞吐。

> 面试金句："在线我守 TTFT 和 P99；离线我最大化 throughput。两者通过 batching 策略和 continuous batching 来平衡。"

## 6. 延伸阅读（建议读源码/官方博客）

- vLLM 论文：PagedAttention（SOSP 2023）
- "LLM Inference 性能指标" — TTFT / TPOT / Goodput
