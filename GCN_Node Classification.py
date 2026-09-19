import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# 设置随机种子
torch.manual_seed(42)

# 加载 Cora 数据集
dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

print(f'Dataset: {dataset.name}')
print(f'节点数: {data.num_nodes}, 边数: {data.num_edges}')
print(f'特征维度: {dataset.num_features}, 类别数: {dataset.num_classes}')
print(f'训练/验证/测试: {data.train_mask.sum().item()} / '
      f'{data.val_mask.sum().item()} / {data.test_mask.sum().item()}')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 预热 + cosine 学习率调度
total_epochs = 200
warmup_epochs = 10
warmup_start_factor = 0.1


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # 线性预热: 从 start_factor 升到 1
        return warmup_start_factor + (1 - warmup_start_factor) * (epoch + 1) / warmup_epochs
    # cosine 退火: 从 1 降到 0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        accs.append((pred[mask] == data.y[mask]).float().mean().item())
    return accs


best_val_acc = 0.0
best_test_acc = 0.0

for epoch in range(1, total_epochs + 1):
    loss = train()
    scheduler.step()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d} | Loss {loss:.4f} | LR {lr:.4f} | '
              f'Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}')

print(f'\n最佳验证准确率: {best_val_acc:.4f}, 对应测试准确率: {best_test_acc:.4f}')
