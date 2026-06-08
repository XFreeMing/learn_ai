"""M1-03: PyTorch 实现 3-5 层 MLP 并训练。

数据: sklearn digits (8x8 手写数字, 10 类)，无需下载 MNIST。
目标: 测试准确率 > 0.95。
对照 02_mlp_numpy.py 体会框架帮你自动做了反向传播 (autograd)。
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    """4 层全连接网络。"""

    def __init__(self, in_dim=64, hidden=128, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(Xtr)
    Xtr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yte = torch.tensor(yte, dtype=torch.long)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(101):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()          # autograd 自动反向传播
        opt.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte).argmax(1) == yte).float().mean().item()
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  test_acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        final_acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"final test accuracy = {final_acc:.3f}")
    assert final_acc > 0.95, "accuracy below target"
    print("PASS ✅  test_acc > 0.95")


if __name__ == "__main__":
    main()
