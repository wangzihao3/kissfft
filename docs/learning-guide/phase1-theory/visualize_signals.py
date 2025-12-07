#!/usr/bin/env python3
"""
信号可视化工具
用于理解时域和频域的关系
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_signal(fs, duration, signal_type='sine', frequency=10):
    """生成测试信号"""
    t = np.arange(0, duration, 1/fs)

    if signal_type == 'sine':
        signal = np.sin(2 * np.pi * frequency * t)
    elif signal_type == 'cosine':
        signal = np.cos(2 * np.pi * frequency * t)
    elif signal_type == 'square':
        signal = np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == 'sawtooth':
        signal = 2 * (frequency * t % 1) - 1
    elif signal_type == 'triangle':
        signal = 2 * np.abs(2 * (frequency * t % 1) - 1) - 1
    elif signal_type == 'chirp':
        signal = np.sin(2 * np.pi * frequency * t * t / duration)
    elif signal_type == 'composite':
        signal = (np.sin(2 * np.pi * frequency * t) +
                 0.5 * np.sin(2 * np.pi * 3 * frequency * t) +
                 0.25 * np.sin(2 * np.pi * 5 * frequency * t))
    else:
        raise ValueError(f"未知的信号类型: {signal_type}")

    return t, signal

def compute_spectrum(signal, fs):
    """计算信号的频谱"""
    N = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, 1/fs)

    # 只返回正频率部分
    pos_mask = frequencies >= 0
    return frequencies[pos_mask], np.abs(fft_result[pos_mask])

def plot_signal_and_spectrum(t, signal, frequencies, spectrum, signal_name):
    """绘制信号及其频谱"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 时域信号
    ax1.plot(t, signal, 'b-', linewidth=1.5)
    ax1.set_title(f'{signal_name} - 时域信号')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.1)  # 只显示前 0.1 秒

    # 频域频谱
    ax2.stem(frequencies[:len(frequencies)//2],
            spectrum[:len(frequencies)//2],
            'r-', markerfmt='ro', basefmt='r-')
    ax2.set_title(f'{signal_name} - 频域频谱')
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('幅值')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)  # 只显示 0-100 Hz

    plt.tight_layout()
    return fig

def demonstrate_sampling():
    """演示采样定理"""
    fs_original = 1000  # 原始采样率
    fs_low = 25         # 低采样率
    duration = 1

    # 生成高频信号
    t_original, signal_original = generate_signal(
        fs_original, duration, 'sine', frequency=50)

    # 低采样率采样
    t_low, signal_low = generate_signal(
        fs_low, duration, 'sine', frequency=50)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 原始信号
    axes[0].plot(t_original, signal_original, 'b-', linewidth=2, label='原始信号')
    axes[0].set_title('原始信号 (50 Hz)')
    axes[0].set_ylabel('幅值')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 低采样率采样
    axes[1].plot(t_original, signal_original, 'b-', alpha=0.3, label='原始信号')
    axes[1].stem(t_low, signal_low, 'r-', markerfmt='ro', basefmt='r-',
                label='采样点', linefmt='r-')
    axes[1].set_title(f'低采样率采样 (fs = {fs_low} Hz)')
    axes[1].set_ylabel('幅值')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 频谱对比
    freq_orig, spec_orig = compute_spectrum(signal_original, fs_original)
    freq_low, spec_low = compute_spectrum(signal_low, fs_low)

    axes[2].plot(freq_orig[:200], spec_orig[:200], 'b-',
                label=f'原始采样 (fs={fs_original})')
    axes[2].plot(freq_low[:100], spec_low[:100], 'r-',
                label=f'低采样 (fs={fs_low})')
    axes[2].set_title('频谱对比 - 展示混叠效应')
    axes[2].set_xlabel('频率 (Hz)')
    axes[2].set_ylabel('幅值')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0, 100)

    plt.tight_layout()
    return fig

def demonstrate_fft_vs_dft():
    """演示 FFT 相对于 DFT 的计算效率"""
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    dft_times = []
    fft_times = []

    for N in sizes:
        # 生成测试信号
        signal = np.random.randn(N)

        # 计算 DFT 时间
        start_time = np.time.time()
        dft_result = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        dft_time = np.time.time() - start_time
        dft_times.append(dft_time)

        # 计算 FFT 时间
        start_time = np.time.time()
        fft_result = np.fft.fft(signal)
        fft_time = np.time.time() - start_time
        fft_times.append(fft_time)

        print(f"N={N:4d}: DFT={dft_time*1000:7.2f}ms, "
              f"FFT={fft_time*1000:7.4f}ms, "
              f"加速比={dft_time/fft_time:6.1f}x")

    # 绘制性能对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 绝对时间
    ax1.semilogy(sizes, np.array(dft_times)*1000, 'bo-', label='DFT', markersize=8)
    ax1.semilogy(sizes, np.array(fft_times)*1000, 'ro-', label='FFT', markersize=8)
    ax1.set_xlabel('序列长度 N')
    ax1.set_ylabel('计算时间 (ms)')
    ax1.set_title('DFT vs FFT 计算时间')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 加速比
    speedup = np.array(dft_times) / np.array(fft_times)
    ax2.plot(sizes, speedup, 'go-', markersize=8)
    ax2.set_xlabel('序列长度 N')
    ax2.set_ylabel('加速比')
    ax2.set_title('FFT 相对于 DFT 的加速比')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # 添加理论曲线
    theoretical_speedup = sizes / np.log2(sizes)
    ax2.plot(sizes, theoretical_speedup, 'k--', alpha=0.5,
            label='理论加速比 (N/log₂N)')
    ax2.legend()

    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("=== 信号分析可视化工具 ===")
    print("\n1. 基础信号分析")

    # 基本信号类型
    signals = [
        ('sine', '正弦波', 10),
        ('composite', '复合信号', 10),
        ('square', '方波', 10),
        ('sawtooth', '锯齿波', 10)
    ]

    for signal_type, signal_name, freq in signals:
        t, signal = generate_signal(1000, 1, signal_type, freq)
        frequencies, spectrum = compute_spectrum(signal, 1000)

        fig = plot_signal_and_spectrum(t, signal, frequencies, spectrum, signal_name)
        plt.show()
        input("\n按回车键继续...")

    print("\n2. 采样定理演示")
    fig = demonstrate_sampling()
    plt.show()
    input("\n按回车键继续...")

    print("\n3. FFT vs DFT 性能对比")
    fig = demonstrate_fft_vs_dft()
    plt.show()

    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()