"""
Function：Hopfiled神经网络接口
"""
import torch.nn as nn

from modeling import *


# Hopfield神经网络类
class HopfieldTSP(nn.Module):
    def __init__(self, n_cities, adj_matrix):
        super(HopfieldTSP, self).__init__()
        self.n_cities = n_cities
        self.adj_matrix = adj_matrix  # 邻接矩阵（城市间的距离）
        self.w = self.init_weights()  # 初始化权重

    def init_weights(self):
        """ 初始化Hopfield网络的权重矩阵 """
        w = torch.zeros((self.n_cities, self.n_cities))

        # 构造能量函数中的第一项：路径长度
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    w[i, j] = -self.adj_matrix[i, j]

        # 构造能量函数中的第二项：合法性约束（确保每个城市只访问一次）
        for i in range(self.n_cities):
            w[i, i] = 2 * torch.sum(self.adj_matrix[i])  # 加上路径的权重

        return w

    def forward(self, x):
        """ 在Hopfield网络中进行能量最小化的迭代更新 """
        x = x.float()  # 确保输入为浮点类型，以便与权重矩阵类型匹配
        for _ in range(1000):  # 增加迭代次数，确保收敛
            x_new = torch.sign(torch.matmul(self.w, x))
            x = x_new
        return x

    def distance_hopfield(self):
        """ 使用Hopfield神经网络解决TSP问题，返回最优路径和路径长度 """
        # 初始状态（随机生成，表示城市访问的初始状态）
        x = torch.randint(0, 2, (self.n_cities,)) * 2 - 1  # 随机初始化，-1和1之间的值

        # 训练网络
        x = self.forward(x)

        # 提取路径
        path = []
        visited = [False] * self.n_cities
        for i in range(self.n_cities):
            if x[i] == 1 and not visited[i]:  # 只选择访问过的城市
                path.append(i)
                visited[i] = True

        # 如果路径没有遍历所有城市，尝试重新选择路径
        attempts = 0
        while len(path) != self.n_cities and attempts < 10:
            x = torch.randint(0, 2, (self.n_cities,)) * 2 - 1  # 重新初始化
            x = self.forward(x)
            path = []
            visited = [False] * self.n_cities
            for i in range(self.n_cities):
                if x[i] == 1 and not visited[i]:
                    path.append(i)
                    visited[i] = True
            attempts += 1

        if len(path) != self.n_cities:
            raise ValueError("Path does not visit all cities. Something went wrong!")

        # 最后加入起点，确保路径回到出发城市
        path.append(path[0])

        # 计算最终路径的长度
        path_length = compute_path_length(path, self.adj_matrix)

        return path, path_length
