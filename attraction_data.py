# 将旅游景点的邻接矩阵保存为numpy数组,作为其他文件调用时的数据源
import numpy as np
import pandas as pd

# 读取excel文件(南京市旅游景点的邻接矩阵)
file_path = './attraction_distance.xlsx'
df = pd.read_excel(file_path)

# 读取邻接矩阵信息
df_subset = df.iloc[0:25, 1:26]

# 将excel表中的邻接矩阵转换成numpy数组
adj_matrix = np.array(df_subset)

# 存储旅游景点与序列的映射关系
df_attribute = df.iloc[0:25, 0]
attr_map = {i: df_attribute.iloc[i] for i in range(len(df_attribute))}