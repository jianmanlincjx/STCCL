import os
import numpy as np
import torch


# 读取txt文件
with open("/data2/JM/code/STCCL/test_compute_time_differences.txt", "r") as f:
    lines = f.readlines()

# 初始化每一列的和为0
sums = [0] * len(lines[0].split())

# 计算每一列的和
for line in lines:
    values = [float(val) for val in line.split()]
    for i, val in enumerate(values):
        sums[i] += val

# 计算每一列的均值
num_rows = len(lines)
means = [sum_val / num_rows for sum_val in sums]

print("每一列的均值：", means)
