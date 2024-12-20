# 利用图神经网络解决TSP问题
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv


# 定义图神经网络模型
class TSP_GNN(nn.Module):
    def __init__(self, n_cities, hidden_dim=64):
        super(TSP_GNN, self).__init__()
        # 定义图卷积层
        self.conv1 = GCNConv(n_cities, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_cities)  # 最终的输出层

    def forward(self, edge_index, edge_attr, node_features):
        # 图卷积层的前向传播
        x = torch.relu(self.conv1(node_features, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        # 输出层，预测城市间的选择概率
        x = self.fc(x)
        return torch.softmax(x, dim=-1)


def probabilistic_search(prob_matrix, temperature=1.0):
    """
    基于温度的概率选择策略（模拟退火）
    """
    n_cities = prob_matrix.shape[0]
    visited = [False] * n_cities
    path = [0]  # 从第一个城市出发
    visited[0] = True
    current_city = 0

    # 依次选择下一个城市
    for _ in range(n_cities - 1):
        prob = prob_matrix[current_city]  # 当前城市的概率分布
        prob[visited] = 0  # 禁止访问已经访问过的城市

        # 使用softmax温度采样
        prob = torch.exp(prob / temperature)  # 温度调节
        prob = prob / prob.sum()  # 归一化

        next_city = torch.multinomial(prob, 1)  # 从概率分布中选择一个城市
        next_city = next_city.item()

        # 确保选择未被访问的城市
        while visited[next_city]:
            prob[next_city] = 0  # 将已访问的城市的概率设为0
            next_city = torch.multinomial(prob, 1).item()  # 重新选择

        path.append(next_city)
        visited[next_city] = True
        current_city = next_city

    # 最后返回第一个城市
    path.append(path[0])
    return path


def compute_path_length(path, adj_matrix):
    """
    计算路径长度
    """
    length = 0
    for i in range(len(path) - 1):
        length += adj_matrix[path[i], path[i + 1]]
    # 注意计算回到路径起点的长度
    length += adj_matrix[path[-1], path[0]]
    return length


# 根据邻接矩阵构建图的边信息
def build_graph(adj_matrix):
    edge_index = []
    edge_attr = []
    n_cities = adj_matrix.shape[0]

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            if adj_matrix[i, j] > 0:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(adj_matrix[i, j])
                edge_attr.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr


# GNN模型训练函数
def solve_tsp_GNN(adj_matrix, epochs=1000, lr=0.01, temperature=1.0, lr_scheduler=None):
    n_cities = adj_matrix.shape[0]
    model = TSP_GNN(n_cities)  # 初始化GNN模型
    optimizer = optim.Adam(model.parameters(), lr=lr)
    node_features = torch.eye(n_cities)  # 每个节点的特征是单位矩阵（每个城市的特征）

    # 构建图的边信息
    edge_index, edge_attr = build_graph(adj_matrix)

    best_path_length = float('inf')  # 保存当前最优路径长度
    best_path = None  # 保存当前最优路径
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 计算路径概率分布
        prob_matrix = model(edge_index, edge_attr, node_features)

        # 使用温度调节的策略选择路径
        path = probabilistic_search(prob_matrix, temperature)

        # 计算路径长度
        path_length = compute_path_length(path, adj_matrix)

        # 如果当前路径长度更短，则更新最优路径
        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path

        # 使用强化学习的策略梯度更新网络
        reward = -path_length  # 奖励是路径长度的负值（越短越好）

        # 损失函数：路径长度作为损失
        loss = torch.tensor(path_length, dtype=torch.float32, requires_grad=True)

        loss.backward()
        optimizer.step()

        # 动态调整温度
        temperature = max(temperature * 0.99, 0.1)  # 温度逐渐降低，增加探索性

        # 学习率调度
        if lr_scheduler:
            lr_scheduler.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Path Length: {best_path_length}")

    return model, best_path, best_path_length


# 使用自适应学习率调度器
def create_lr_scheduler(optimizer, initial_lr=0.01, decay_rate=0.9, decay_steps=100):
    return optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
