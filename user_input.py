""" 设置可视化界面 """
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from TSP_GNN import *  # 图神经网络算法
from hopfield import *
from validation import *


class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("南京市旅游景点选择")

        # 设置窗口大小，确保一开始就足够显示所有内容
        self.root.geometry("1080x720")  # 可以稍微增大窗口高度，确保可以显示所有内容

        # 用于存储用户选择的景点
        self.selected_attractions = []
        self.first_selected = None  # 用来记录第一次点击的景点

        # 创建路径规划、时间规划的选择按钮
        self.instructions = tk.Label(root, text="规划目标选择：", font=("System", 10))
        self.path_planning_button = tk.Button(root, text="路径规划", width=20, height=2, command=self.show_path_planning,
                                              relief="sunken")
        self.time_cost_planning_button = tk.Button(root, text="时间及花费规划", width=20, height=2,
                                                   command=self.show_time_cost_planning, relief="raised")

        # 设置标签位置
        self.instructions.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # 将按钮居中显示
        self.path_planning_button.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
        self.time_cost_planning_button.grid(row=0, column=2, columnspan=2, padx=10, pady=10)

        # 创建景点选择按钮布局
        self.create_attraction_buttons(root)

        # 创建路径规划布局
        self.create_path_planning_layout(root)

        # 创建时间规划布局
        self.create_time_cost_planning_layout(root)

        # 设置默认选中路径规划布局
        self.show_path_planning()

    def create_attraction_buttons(self, root):
        # 创建25个按钮，按钮的宽度和高度适中，适应布局
        self.buttons = {}
        for i, name in attr_map.items():
            button = tk.Button(root, text=name, width=25, height=3, command=lambda i=i: self.toggle_selection(i))
            button.grid(row=(i // 5) + 1, column=i % 5, padx=15, pady=10)  # 按钮网格布局
            self.buttons[i] = button

        # 创建成本参数选项(仅在时间规划下显示)
        self.cost_weight_label = tk.Label(root, text="成本参数(花费相对时间的权重):", font=("System", 10))
        self.cost_weight_knob = ttk.Combobox(root, values=[0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000],
                                             state="readonly", width=10)
        self.cost_weight_knob.set(1)  # 设置默认值

        # 添加“一键清除”按钮
        self.clear_button = tk.Button(root, text="一键清除", width=20, height=2, command=self.clear_selection)
        self.clear_button.grid(row=6, column=2, columnspan=3, pady=10)

        # 添加“提交“按钮
        self.submit_button = tk.Button(root, text="提交", width=20, height=2, command=self.submit_selection)
        self.submit_button.grid(row=6, column=4, columnspan=1, pady=10)

    def create_path_planning_layout(self, root):
        # 创建显示最佳路径的标签和文本框
        self.best_path_label = tk.Label(root, text="最佳路径：", font=("System", 10))
        self.best_path_text = tk.Text(root, width=70, height=3, font=("System", 10))

        # 设置最短距离的标签和文本框
        self.best_distance_label = tk.Label(root, text="最短距离：", font=("System", 10))
        self.best_distance_text = tk.Entry(root, width=70, font=("System", 10))  # 保持一致的字体大小

        # 创建进度条
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
        self.progress_bar.grid_forget()  # 隐藏进度条，直到计算开始

    def create_time_cost_planning_layout(self, root):
        # 创建显示最佳路径的标签和文本框
        self.best_path_time_label = tk.Label(root, text="最佳路径：", font=("System", 10))
        self.best_path_time_text = tk.Text(root, width=70, height=3, font=("System", 10))

        # 创建显示规划时间和规划费用的标签和文本框
        self.best_time_label = tk.Label(root, text="规划时间：", font=("System", 10))
        self.best_time_text = tk.Entry(root, width=70, font=("System", 10))

        self.best_cost_label = tk.Label(root, text="规划费用：", font=("System", 10))
        self.best_cost_text = tk.Entry(root, width=70, font=("System", 10))

        # 创建进度条
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
        self.progress_bar.grid_forget()  # 隐藏进度条，直到计算开始

    # 通用布局设置
    def toggle_layout(self, layout_type):
        # 先隐藏所有布局
        for widget in [self.best_path_label, self.best_path_text, self.best_distance_label, self.best_distance_text,
                       self.best_path_time_label, self.best_path_time_text, self.best_time_label, self.best_time_text,
                       self.best_cost_label, self.best_cost_text, self.cost_weight_label, self.cost_weight_knob]:
            widget.grid_forget()

        # 隐藏进度条
        self.progress_bar.grid_forget()

        if layout_type == "path_planning":
            # 显示路径规划部分
            self.best_path_label.grid(row=7, column=0, columnspan=1, sticky="e", padx=0, pady=5)
            self.best_path_text.grid(row=7, column=1, columnspan=3, sticky="w", padx=0, pady=5)
            self.best_distance_label.grid(row=8, column=0, columnspan=1, sticky="e", padx=0, pady=5)
            self.best_distance_text.grid(row=8, column=1, columnspan=3, sticky="w", padx=0, pady=5)

            # 隐藏时间及花费规划部分
            for widget in [self.best_path_time_label, self.best_path_time_text, self.best_time_label,
                           self.best_time_text,
                           self.best_cost_label, self.best_cost_text]:
                widget.grid_forget()

        elif layout_type == "time_cost_planning":
            # 显示时间及花费规划部分
            self.best_path_time_label.grid(row=7, column=0, columnspan=1, sticky="e", padx=0, pady=5)
            self.best_path_time_text.grid(row=7, column=1, columnspan=3, sticky="w", padx=0, pady=5)
            self.best_time_label.grid(row=8, column=0, columnspan=1, sticky="e", padx=0, pady=5)
            self.best_time_text.grid(row=8, column=1, columnspan=3, sticky="w", padx=0, pady=5)
            self.best_cost_label.grid(row=9, column=0, columnspan=1, sticky="e", padx=0, pady=5)
            self.best_cost_text.grid(row=9, column=1, columnspan=3, sticky="w", padx=0, pady=5)

            # 显示成本参数部分
            self.cost_weight_label.grid(row=6, column=1, padx=0, pady=10)
            self.cost_weight_knob.grid(row=6, column=2, padx=0, pady=10)

    # 展示路径规划布局函数
    def show_path_planning(self):
        self.toggle_layout("path_planning")
        self.path_planning_button.config(relief="sunken", state="disabled")  # 按下状态
        self.time_cost_planning_button.config(relief="raised", state="normal")  # 恢复为可点击状态

    # 展示时间规划布局函数
    def show_time_cost_planning(self):
        self.toggle_layout("time_cost_planning")
        self.time_cost_planning_button.config(relief="sunken", state="disabled")  # 按下状态
        self.path_planning_button.config(relief="raised", state="normal")  # 恢复为可点击状态

    def toggle_selection(self, index):
        """点击按钮切换景点的选择状态"""
        # 若第一次点击景点，设置成初始景点，显示不同颜色
        if index in self.selected_attractions:
            self.selected_attractions.remove(index)
            self.buttons[index].config(bg="SystemButtonFace")  # 恢复默认背景色
        else:
            self.selected_attractions.append(index)
            if self.first_selected is None:
                self.first_selected = index
                self.buttons[index].config(bg="lightgreen")  # 初始景点的背景色为 lightgreen
            else:
                self.buttons[index].config(bg="lightblue")  # 非初始景点的背景色为 lightblue
        # 如果没有景点被选择，重置第一次点击的按钮
        if not self.selected_attractions:
            self.reset_first_selected_button()

    # 重置第一次点击的按钮
    def reset_first_selected_button(self):
        if self.first_selected is not None:
            self.buttons[self.first_selected].config(bg="SystemButtonFace")  # 恢复默认背景色
            self.first_selected = None  # 重置第一次点击的标记

    # 提交选择的景点并计算子矩阵，注意这里对用户提交有限制
    def submit_selection(self):
        if len(self.selected_attractions) < 2:
            messagebox.showwarning("选择错误", "请至少选择两个景点！")
            return

        # 获取用户输入信息并处理
        selected_indices = self.selected_attractions

        # 路径规划调用算法求解
        if self.path_planning_button['relief'] == 'sunken':
            sub_adj_matrix = adj_matrix_distance[self.selected_attractions, :][:, self.selected_attractions]
            selected_names = [attr_map[i] for i in self.selected_attractions]

            # # 输出选中的景点及子邻接矩阵
            # print("您选择的景点是:", selected_names)
            # print("这些景点之间距离的邻接矩阵是:")
            # print(sub_adj_matrix)

            # 显示进度条，开始加载
            self.progress_bar.grid(row=12, column=2, columnspan=3, pady=10)  # 显示进度条
            self.progress_bar.start()  # 启动进度条动画

            # 创建并启动一个后台线程来执行GNN和Hopfield计算
            threading.Thread(target=self.distance_run_both_models, args=(sub_adj_matrix, selected_indices)).start()

            # """注意:这里的验证部分仅用于实验,不对用户开放;在进行实验时,将下面的取消注释"""
            # # 额外的后台线程，利用穷举法执行路径规划验证，结果不对用户界面显示
            # threading.Thread(target=self.verify_tsp_solution, args=(sub_adj_matrix, selected_indices)).start()


        # 时间规划调用算法求解
        elif self.time_cost_planning_button['relief'] == 'sunken':
            # 获取成本参数
            cost_weight = float(self.cost_weight_knob.get())
            sub_time_matrix = adj_matrix_time[self.selected_attractions, :][:, self.selected_attractions]
            sub_cost_matrix = adj_matrix_cost[self.selected_attractions, :][:, self.selected_attractions]
            selected_names = [attr_map[i] for i in self.selected_attractions]

            # 输出选中的景点及子邻接矩阵
            print("您选择的景点是:", selected_names)
            print("这些景点之间打车/步行时间的邻接矩阵是:")
            print(sub_time_matrix)
            print("这些景点之间打车/步行相应花费的邻接矩阵是:")
            print(sub_cost_matrix)

            # 输出成本参数提示
            print("您选择的成本参数是:", cost_weight)

            # 显示进度条
            self.progress_bar.grid(row=12, column=2, columnspan=3, pady=10)
            self.progress_bar.start()  # 启动动画

            # 创建并启动一个后台线程执行GNN计算
            threading.Thread(target=self.time_cost_run_model,
                             args=(sub_time_matrix, sub_cost_matrix, cost_weight, selected_indices)).start()

    # 清除所有已选择的景点
    def clear_selection(self):
        # 清空已选择景点列表
        self.selected_attractions = []
        self.first_selected = None  # 重置第一次点击的景点

        # 重置所有按钮的背景颜色为默认状态
        for button in self.buttons.values():
            button.config(bg="SystemButtonFace")

    # 调用GNN和Hopfield神经网络计算路径规划
    def distance_run_both_models(self, sub_adj_matrix, selected_indices):
        """后台运行GNN和Hopfield算法并更新进度条和显示结果"""
        try:
            # 使用 Hopfield 神经网络求解 TSP 问题
            hopfield_net = HopfieldTSP(len(sub_adj_matrix), torch.tensor(sub_adj_matrix, dtype=torch.float32))
            hopfield_path, hopfield_length = hopfield_net.distance_hopfield()  # 使用solve_tsp方法
            hopfield_valid = True  # 假设Hopfield有效
        except Exception as e:
            hopfield_valid = False
            print(f"使用Hopfield神经网络求解失败: {e}")
            hopfield_path, hopfield_length = [], float('inf')  # 如果Hopfield失败，返回无效路径

        # 使用GNN算法求解最短路径规划问题
        model, gnn_best_path, gnn_best_distance = distance_gnn(sub_adj_matrix, epochs=2000, lr=0.01)

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

    # 调用不同目标下的GNN计算时间和花费规划
    def time_cost_run_model(self, sub_time_matrix, sub_cost_matrix, cost_weight, selected_indices):
        try:
            # 清空文本框中的内容，确保每次运行时不会有旧的结果残留
            self.best_path_time_text.delete(1.0, tk.END)  # 清空路径显示框
            self.best_time_text.delete(0, tk.END)  # 清空时间显示框
            self.best_cost_text.delete(0, tk.END)  # 清空费用显示框

            # 调用 GNN 模型进行路径规划并返回最佳路径、时间和费用
            model, best_path, best_time, best_cost = time_cost_gnn(sub_time_matrix, sub_cost_matrix, cost_weight,
                                                                   epochs=2000)

            # 计算最佳路径的总时间和总费用
            total_time = best_time
            total_cost = best_cost

            # 将最佳路径索引转换为景点名称
            original_path = [selected_indices[i] for i in best_path]
            path_names = [attr_map[i] for i in original_path]
            path_str = " -> ".join(path_names)

            # 更新最佳路径、规划时间和规划费用的文本框
            self.best_path_time_text.insert(tk.END, path_str)  # 显示最佳路径
            self.best_time_text.insert(tk.END, f"{total_time:.2f}")  # 显示规划时间
            self.best_cost_text.insert(tk.END, f"{total_cost:.2f}")  # 显示规划费用

            # 弹出提示框，提示用户计算完成
            messagebox.showinfo("结果", "已生成最佳路径、对应的规划时间和规划费用")

        except Exception as e:
            print(f"运行GNN模型时出现错误: {e}")
            messagebox.showerror("错误", "请检查模型的输入和参数是否正确")

        # 隐藏进度条
        self.progress_bar.grid_forget()

    # 利用穷举法进行局部验证,这里仅对路径规划验证(不对用户开放,在与用户启动页面时关闭)
    def verify_tsp_solution(self, sub_adj_matrix, selected_indices):
        try:
            # 使用TSPValidation来验证结果
            tsp_validator = TSPValidation(sub_adj_matrix)

            # 在后台计算最短路径，并输出结果到终端
            min_path, min_length = tsp_validator.find_shortest_path()

            # 输出验证结果到终端（不显示在用户界面）
            min_path_names = [attr_map[i] for i in min_path]
            path_str = " -> ".join(min_path_names)
            print(f"穷举法验证结果: 最短路径为: {path_str}")
            print(f"最短路径长度为: {min_length:.2f}")

        except Exception as e:
            print(f"验证过程出现错误: {e}")
