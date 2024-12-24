""" 主函数，程序运行的起点 """
from user_input import *

# 创建主窗口
root = tk.Tk()
app = TSPApp(root)
root.mainloop()