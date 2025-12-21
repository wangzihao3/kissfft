#!/usr/bin/env python3
import numpy as np

# 序列 [1, 1, 0, 0] 的 4 点 DFT
x = np.array([1, 1, 0, 0])

# 使用 numpy 的 FFT 函数计算
X = np.fft.fft(x)

print("序列:", x)
print("DFT 结果:")
for k in range(len(X)):
    print(f"X[{k}] = {X[k]}")

# 手工计算验证
print("\n手工计算:")
N = 4
for k in range(N):
    result = 0
    for n in range(N):
        angle = -2j * np.pi * k * n / N
        result += x[n] * np.exp(angle)
    print(f"X[{k}] = {result}")

# 验证共轭对称性（实数信号的性质）
print("\n共轭对称性验证:")
print(f"X[1] = {X[1]}, X[3]* = {np.conj(X[3])}")
print(f"X[2] = {X[2]}, X[2]* = {np.conj(X[2])}（实数，应该相等）")