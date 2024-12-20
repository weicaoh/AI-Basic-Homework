# 设置可视化界面
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from GNN import *  # 图神经网络算法
from attraction_data import *  # 已拥有 attr_map 和 adj_matrix 数据
from hopfield import *


class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP 景点选择")

        # 设置窗口大小，确保一开始就足够显示所有内容
        self.root.geometry("1100x700")  # 可以稍微增大窗口高度，确保可以显示所有内容

        # 用于存储用户选择的景点
        self.selected_attractions = []

        # 创建 25 个按钮，按钮的宽度和高度适中，适应布局
        self.buttons = {}
        for i, name in attr_map.items():
            button = tk.Button(root, text=name, width=25, height=3, command=lambda i=i: self.toggle_selection(i))
            button.grid(row=i // 5, column=i % 5, padx=10, pady=10)  # 按钮网格布局
            self.buttons[i] = button

        # 提交按钮放置到按钮群的下方，增加一些间距
        self.submit_button = tk.Button(root, text="提交", width=20, height=2, command=self.submit_selection)
        self.submit_button.grid(row=5, column=2, columnspan=3, pady=20)

        # 创建显示最佳路径和最短距离的标签和文本框
        self.best_path_label = tk.Label(root, text="最佳路径：", font=("System", 10))
        self.best_path_label.grid(row=6, column=0, columnspan=1, sticky="e", padx=0, pady=5)

        # 调整 best_path_text 的高度，保证与 entry 高度一致
        self.best_path_text = tk.Text(root, width=70, height=3, font=("System", 10))
        self.best_path_text.grid(row=6, column=1, columnspan=3, sticky="w", padx=0, pady=5)

        self.best_distance_label = tk.Label(root, text="最短距离：", font=("System", 10))
        self.best_distance_label.grid(row=7, column=0, columnspan=1, sticky="e", padx=0, pady=5)

        # 设置 best_distance_text 的字体与 Text 组件一致，并确保两者高度一致
        self.best_distance_text = tk.Entry(root, width=70, font=("System", 10))  # 保持一致的字体大小
        self.best_distance_text.grid(row=7, column=1, columnspan=3, sticky="w", padx=0, pady=5)

        # 调整整个布局，使其位于页面中心
        self.root.grid_rowconfigure(6, weight=1)  # 设置第6行（最佳路径和文本框所在的行）的权重，使其占满空间
        self.root.grid_rowconfigure(7, weight=1)  # 设置第7行（最短距离和文本框所在的行）的权重，使其占满空间
        self.root.grid_columnconfigure(0, weight=1)  # 设置第0列（标签所在列）的权重，使其占满空间
        self.root.grid_columnconfigure(1, weight=4)  # 设置第1列（文本框所在列）的权重，使其占满空间
        self.root.grid_columnconfigure(2, weight=4)  # 设置第2列（文本框所在列的其他列）的权重，使其占满空间
        self.root.grid_columnconfigure(3, weight=1)  # 设置第3列（文本框所在列）的权重，使其占满空间
        self.root.grid_columnconfigure(4, weight=1)  # 设置第4列（文本框所在列的其他列）的权重，使其占满空间

        # 创建进度条
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
        self.progress_bar.grid(row=8, column=1, columnspan=3, pady=10)
        self.progress_bar.grid_forget()  # 隐藏进度条，直到计算开始

        # 配置列和行的权重，使得界面布局更灵活
        for i in range(5):
            self.root.grid_columnconfigure(i, weight=1, uniform="equal")
        self.root.grid_rowconfigure(5, weight=1, uniform="equal")
        self.root.grid_rowconfigure(8, weight=1, uniform="equal")

    def toggle_selection(self, index):
        """点击按钮切换景点的选择状态"""
        if index in self.selected_attractions:
            self.selected_attractions.remove(index)
            self.buttons[index].config(bg="SystemButtonFace")  # 恢复默认背景色
        else:
            self.selected_attractions.append(index)
            self.buttons[index].config(bg="lightblue")  # 选中状态的背景色

    def submit_selection(self):
        """提交选择的景点并计算子矩阵，注意这里对用户提交有限制"""
        if len(self.selected_attractions) < 2:
            messagebox.showwarning("选择错误", "请至少选择两个景点！")
            return

        # 获取子邻接矩阵的索引，保证索引一致性
        selected_indices = self.selected_attractions

        # 获取子邻接矩阵
        sub_adj_matrix = adj_matrix[self.selected_attractions, :][:, self.selected_attractions]

        # 获取选中的景点名称
        selected_names = [attr_map[i] for i in self.selected_attractions]

        # 输出选中的景点及子邻接矩阵
        print("您选择的景点是:", selected_names)
        print("这些景点的邻接矩阵是:")
        print(sub_adj_matrix)

        # 显示进度条，开始加载
        self.progress_bar.grid()  # 显示进度条
        self.progress_bar.start()  # 启动进度条动画

        # 在后台线程中运行GNN和Hopfield神经网络，并更新进度条
        self.root.after(100, self.run_both_models, sub_adj_matrix, selected_indices)

    def run_both_models(self, sub_adj_matrix, selected_indices):
        """后台运行GNN和Hopfield算法并更新进度条和显示结果"""
        try:
            # 使用 Hopfield 神经网络求解 TSP 问题
            hopfield_net = HopfieldTSP(len(sub_adj_matrix), torch.tensor(sub_adj_matrix, dtype=torch.float32))
            hopfield_path, hopfield_length = hopfield_net.solve_tsp()  # 使用solve_tsp方法
            hopfield_valid = True  # 假设Hopfield有效
        except Exception as e:
            hopfield_valid = False
            print(f"Hopfield failed: {e}")
            hopfield_path, hopfield_length = [], float('inf')  # 如果Hopfield失败，返回无效路径

        # 使用 GNN 算法求解 TSP 问题
        model, gnn_best_path, gnn_best_distance = solve_tsp_GNN(sub_adj_matrix, epochs=2000, lr=0.01)

        # 比较两个路径的长度，选择更短的路径
        if hopfield_valid and hopfield_length < gnn_best_distance:
            best_path, best_distance = hopfield_path, hopfield_length
        else:
            best_path, best_distance = gnn_best_path, gnn_best_distance

        # 设置进度条的进度为最大值，拉满进度条
        self.progress_bar['value'] = self.progress_bar['maximum']

        # 计算完成，停止进度条
        self.progress_bar.stop()

        # 注意，best_path中的索引是基于子邻接矩阵的索引，需要将其映射回原始矩阵的索引
        original_path = [selected_indices[i] for i in best_path]  # 将子矩阵中的路径索引转换为原始索引

        # 转换路径为景点名称
        path_names = [attr_map[i] for i in original_path]
        path_str = " -> ".join(path_names)

        # 更新界面上的最佳路径和最短距离
        self.best_path_text.delete(1.0, tk.END)  # 清空文本框
        self.best_path_text.insert(tk.END, path_str)  # 显示最佳路径

        self.best_distance_text.delete(0, tk.END)  # 清空文本框
        self.best_distance_text.insert(tk.END, f"{best_distance:.2f}")  # 显示最短路径长度

        messagebox.showinfo("结果", "已提交选择并生成子邻接矩阵，已生成最佳路径")

        # 隐藏进度条
        self.progress_bar.grid_forget()
