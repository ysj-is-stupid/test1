import numpy as np
import matplotlib.pyplot as plt

# 参数设置
np.random.seed(42)  # 保证结果可复现
length = 5000       # 时间序列长度

# 生成时间轴
t = np.arange(length)

# 1. 三个周期性成分
# 短周期（高频）
period1 = 50
amplitude1 = 15
cyclic1 = amplitude1 * np.sin(2 * np.pi * t / period1)

# 中周期
period2 = 100
amplitude2 = 10
cyclic2 = amplitude2 * np.cos(2 * np.pi * t / period2)  # 使用cos创造相位差

# 长周期（低频）
period3 = 200
amplitude3 = 7
cyclic3 = amplitude3 * np.sin(2 * np.pi * t / period3 + np.pi/3)  # 添加相位偏移

# 2. 线性趋势成分
slope = 0.15
trend = slope * t

# 3. 高斯白噪声
noise_std = 4  # 噪声强度
noise = np.random.normal(0, noise_std, length)

# 合成时间序列
time_series = cyclic1 + cyclic2 + cyclic3 + trend + noise

# 可视化
plt.figure(figsize=(12, 12))

# 绘制各成分分解图
components = [
    ("Cyclic Component 1 (Short)", cyclic1),
    ("Cyclic Component 2 (Medium)", cyclic2),
    ("Cyclic Component 3 (Long)", cyclic3),
    ("Linear Trend", trend),
    ("Noise", noise),
    ("Combined Signal", time_series)
]

for i, (title, component) in enumerate(components):
    plt.subplot(6, 1, i+1)
    plt.plot(t, component)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)

plt.tight_layout()
plt.show()

# 可选：保存为numpy文件
# np.save("synthetic_timeseries.npy", time_series)
# 可选：保存为CSV
np.savetxt("test_1.csv", time_series, delimiter=",")