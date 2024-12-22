# 利用图神经网络解决TSP问题
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

from modeling import *


# 定义图神经网络模型
class TSP_GNN(nn.Module):
    def __init__(self, n_cities, hidden_dim=64):
        super(TSP_GNN, self).__init__()
        # 定义图卷积层
        self.conv1 = GCNConv(n_cities, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_cities)  # 最终的输出层

    def forward(self, edge_index, edge_attr, node_features):
        # 图卷积层的前向传播，使用Relu函数作为激活函数
        x = torch.relu(self.conv1(node_features, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        # 输出层，预测城市间的选择概率
        x = self.fc(x)
        return torch.softmax(x, dim=-1)


# 基于最短距离构建GNN模型
def distance_gnn(adj_matrix, epochs, lr=0.01, temperature=1.0):
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

        # 损失函数：路径长度作为损失
        loss = torch.tensor(path_length, dtype=torch.float32, requires_grad=True)

        loss.backward()
        optimizer.step()

        # 动态调整温度
        temperature = max(temperature * 0.99, 0.1)  # 温度逐渐降低，增加探索性

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"当前Epoch：{epoch}, 当前最短路径长度为: {best_path_length}")

    return model, best_path, best_path_length


# 基于时间和花费构建GNN模型
def time_cost_gnn(sub_time_matrix, sub_cost_matrix, cost_weight, epochs, lr=0.01, temperature=1.0):
    n_cities = sub_time_matrix.shape[0]
    model = TSP_GNN(n_cities)  # 初始化GNN模型
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 初始化每个节点的特征，这里的特征维度是hidden_dim
    node_features = torch.eye(n_cities)  # 每个节点的特征是单位矩阵（每个城市的特征）

    # 以时间和成本先验构建图的边信息
    adjusted_edge_attr = sub_time_matrix + (cost_weight * sub_cost_matrix)
    edge_index, edge_attr = build_graph(adjusted_edge_attr)

    best_path_total_cost = float('inf')  # 保存当前最优路径长度
    best_time = float('inf')  # 保存当前最优总时间
    best_cost = float('inf')  # 保存当前最优总花费
    best_path = None  # 保存当前最优路径

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 计算路径概率分布
        prob_matrix = model(edge_index, edge_attr, node_features)

        # 使用温度调节的策略选择路径
        path = probabilistic_search(prob_matrix, temperature)

        # 计算路径的总时间和总花费
        total_time, total_cost = compute_time_cost(path, sub_time_matrix, sub_cost_matrix)

        # 将total_time 和 total_cost 转换为torch.tensor
        total_time = torch.tensor(total_time, dtype=torch.float32)
        total_cost = torch.tensor(total_cost, dtype=torch.float32)

        # 计算总的目标函数(规划时间+权重×规划成本)
        total_cost_time = total_time + cost_weight * total_cost

        # 如果当前路径目标函数更小，则更新最优路径
        if total_cost_time < best_path_total_cost:
            best_path_total_cost = total_cost_time
            best_time = total_time
            best_cost = total_cost
            best_path = path

        # 添加正则化项，保证时间和成本的反向关系
        reg_loss = torch.abs(total_time - total_cost)  # 计算正则化损失

        # 使用clone().detach()来避免丢失梯度信息，并确保参与梯度计算
        total_loss = (total_cost_time + reg_loss).clone().detach().requires_grad_(True)  # 目标函数 + 正则化损失

        total_loss.backward()
        optimizer.step()

        # 动态调整温度
        temperature = max(temperature * 0.99, 0.1)  # 温度逐渐降低，增加探索性

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"当前Epoch：{epoch}, 当前最短路径总目标函数值为: {best_path_total_cost}")

    return model, best_path, best_time, best_cost
