""" 利用神经网络求解建模的基础函数部分 """
import torch


# 根据邻接矩阵构建图的边信息
def build_graph(adj_matrix):
    edge_index = []
    edge_attr = []
    n_cities = adj_matrix.shape[0]

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(adj_matrix[i, j])
            edge_attr.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr


# 基于温度的概率选择策略(模拟退火方法)
def probabilistic_search(prob_matrix, temperature=1.0):
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


# 计算当前路径长度
def compute_path_length(path, adj_matrix):
    length = 0
    for i in range(len(path) - 1):
        length += adj_matrix[path[i], path[i + 1]]
    # 注意计算回到路径起点的长度
    length += adj_matrix[path[-1], path[0]]
    return length


# 计算当前耗费时间与花费代价的加权和
def compute_time_cost(path, adj_matrix_time, adj_matrix_cost):
    time, cost = 0, 0
    for i in range(len(path) - 1):
        time += adj_matrix_time[path[i], path[i + 1]]
        cost += adj_matrix_cost[path[i], path[i + 1]]
    # 同样考虑计算回到路径起点的耗费时间与花费代价
    time += adj_matrix_time[path[-1], path[0]]
    cost += adj_matrix_cost[path[-1], path[0]]
    return time, cost
