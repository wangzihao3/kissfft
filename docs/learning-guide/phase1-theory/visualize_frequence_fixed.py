#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform

# 设置matplotlib后端，避免GUI问题
matplotlib.use('Agg')  # 使用非交互式后端，保存到文件

# 根据不同的操作系统设置中文字体
system = platform.system()

if system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成信号
fs = 1000  # 采样率
t = np.arange(0, 1, 1/fs)  # 时间轴
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# 计算频谱
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/fs)

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 绘制时域信号
ax1.plot(t, signal, 'b-', linewidth=1.5)
ax1.set_title('Time Domain Signal', fontsize=14, pad=20)  # 使用英文标题避免字体问题
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, alpha=0.3)

# 绘制频域频谱
ax2.plot(frequencies[:len(frequencies)//2],
         np.abs(fft_result)[:len(frequencies)//2],
         'r-', linewidth=1.5)
ax2.set_title('Frequency Domain Spectrum', fontsize=14, pad=20)  # 使用英文标题
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Magnitude', fontsize=12)
ax2.grid(True, alpha=0.3)

# 在图上添加中文说明（如果字体支持）
try:
    fig.text(0.02, 0.98, '信号：50Hz正弦波 + 0.5×120Hz正弦波',
             transform=fig.transFigure, fontsize=10,
             verticalalignment='top', fontproperties='SimHei')
except:
    # 如果中文显示失败，使用英文说明
    fig.text(0.02, 0.98, 'Signal: 50Hz sine wave + 0.5×120Hz sine wave',
             transform=fig.transFigure, fontsize=10,
             verticalalignment='top')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部文字留空间

# 保存图像
plt.savefig('frequency_spectrum.png', dpi=150, bbox_inches='tight')
print("图像已保存为 frequency_spectrum.png")

# 尝试显示（在支持GUI的环境中）
try:
    plt.show()
except:
    print("当前环境不支持显示图像，但已保存到文件")