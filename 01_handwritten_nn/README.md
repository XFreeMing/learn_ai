# M1 · 手写神经网络

目标：不靠框架封装，亲手实现从 `Y = WX + B` 到反向传播，再到 PyTorch 小网络。

## 文件索引

| 文件 | 内容 | 依赖框架 |
| --- | --- | --- |
| [01_linear_y_wx_b.py](01_linear_y_wx_b.py) | 纯 numpy 实现线性层前向 + 反向 + 梯度数值校验 | numpy |
| [02_mlp_numpy.py](02_mlp_numpy.py) | 纯 numpy 2 层 MLP，玩具二分类训练闭环 | numpy |
| [03_mlp_pytorch.py](03_mlp_pytorch.py) | PyTorch 3–5 层 MLP，玩具集训练 | torch, sklearn |

## 运行

```pwsh
python 01_linear_y_wx_b.py     # 看到 "gradient check passed" 即正确
python 02_mlp_numpy.py         # 看到 loss 持续下降
python 03_mlp_pytorch.py       # 看到 accuracy 上升
```

## Definition of Done

- [ ] `01` 数值梯度校验通过（误差 < 1e-5）
- [ ] `02` 训练 loss 从 ~0.7 降到 < 0.1
- [ ] `03` 测试准确率 > 0.95

## 要点（面试会问）

- `Y = WX + B` 的形状约定：`X (N, in)`, `W (in, out)`, `B (out,)` → `Y (N, out)`。
- 反向传播链式法则：`dL/dW = X.T @ dL/dY`，`dL/dB = sum(dL/dY)`，`dL/dX = dL/dY @ W.T`。
- 为什么需要激活函数：没有非线性，多层等价于单层。
