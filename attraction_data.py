"""
Function：处理原始数据，为神经网络模型提供先验知识
"""

# 将旅游景点的邻接矩阵保存为numpy数组,作为其他文件调用时的数据源
import numpy as np
import pandas as pd

'''距离规划的数据处理部分'''
# 读取excel文件(南京市旅游景点的邻接矩阵)
file_distance = 'origin_data/attraction_distance.xlsx'
df_distance = pd.read_excel(file_distance)

# 读取邻接矩阵信息
df_subset_distance = df_distance.iloc[0:25, 1:26]

# 将excel表中的邻接矩阵转换成numpy数组
adj_matrix_distance = np.array(df_subset_distance)

# 存储旅游景点与序列的映射关系
df_attribute = df_distance.iloc[0:25, 0]
attr_map = {i: df_attribute.iloc[i] for i in range(len(df_attribute))}

'''时间规划的数据处理部分'''
# 读取excel文件(南京市旅游景点之间打车/步行时间的邻接矩阵)
file_time = 'origin_data/attraction_time.xlsx'
df_time = pd.read_excel(file_time)

# 读取excel文件(南京市旅游景点之间打车花销的邻接矩阵)
file_cost = 'origin_data/attraction_cost.xlsx'
df_cost = pd.read_excel(file_cost)

# 读取邻接矩阵信息
df_subset_time = df_time.iloc[0:25, 1:26]
df_subset_cost = df_cost.iloc[0:25, 1:26]

# 将上述邻接矩阵转化成numpy数组
adj_matrix_time = np.array(df_subset_time)
adj_matrix_cost = np.array(df_subset_cost)
