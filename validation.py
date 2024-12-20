import itertools
import numpy as np
from tqdm import tqdm  # 导入tqdm

from attraction_data import *  # 假设此模块定义了 adj_matrix


class TSPValidation:
    def __init__(self, adj_matrix):
        self.adj_matrix = np.array(adj_matrix)
        self.n_cities = len(adj_matrix)

    def compute_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.adj_matrix[path[i], path[i + 1]]
        length += self.adj_matrix[path[-1], path[0]]  # 回到起点
        return length

    def find_shortest_path(self):
        # 获取所有城市的排列（不包括起点）
        cities = list(range(self.n_cities))

        # 生成所有可能的路径排列（不包括回到起点）
        all_permutations = itertools.permutations(cities[1:])

        # 初始化最短路径和最短路径长度
        min_path = None
        min_length = float('inf')

        # 使用tqdm显示进度条
        for perm in tqdm(all_permutations, total=np.math.factorial(self.n_cities - 1), desc="Calculating paths"):
            path = [0] + list(perm)  # 从第一个城市出发
            path_length = self.compute_path_length(path)

            if path_length < min_length:
                min_length = path_length
                min_path = path

        return min_path, min_length


path, length = TSPValidation(adj_matrix).find_shortest_path()
print(f"Shortest path length: {length}")
