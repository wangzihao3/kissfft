# 阶段 1：FFT 理论基础

欢迎进入 FFT 学习的第一个阶段！在本阶段，我们将建立扎实的理论基础，为后续的代码学习和实践应用做准备。

## FFT 可视化说明

### 问题描述
在运行 `visualize_frequence.py` 时，如果中文字符显示为方框（□），这是因为系统缺少合适的中文字体。

### 解决方案

#### 方案 1：安装中文字体（推荐）

运行提供的安装脚本：
```bash
./install_chinese_font.sh
```

或者手动安装：
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install fonts-noto-cjk fonts-wqy-microhei

# 清除字体缓存
sudo fc-cache -fv
```

#### 方案 2：使用修复版本

运行 `visualize_frequence_fixed.py`，该版本：
- 使用英文标签避免字体问题
- 保存图像到文件而不是显示
- 在不支持的环境中也能正常工作

```bash
python3 visualize_frequence_fixed.py
```

#### 方案 3：在 Windows 环境中运行

如果你在 Windows 上使用 Python，可以直接运行原文件，因为 Windows 通常已包含中文字体。

### 文件说明

- `visualize_frequence.py` - 原始版本（需要中文字体支持）
- `visualize_frequence_fixed.py` - 修复版本（英文标签，保存到文件）
- `install_chinese_font.sh` - 中文字体安装脚本

## 学习目标

完成本阶段后，您将能够：
- [ ] 理解数字信号处理的基本概念
- [ ] 掌握傅里叶变换的数学原理
- [ ] 解释 FFT 算法的优化思想
- [ ] 理解 KISS FFT 的设计哲学

## 学习内容

### 1.1 数字信号处理基础

#### 1.1.1 连续信号与离散信号

**核心概念**
- 连续时间信号 vs 离散时间信号
- 采样过程和采样定理
- 奈奎斯特频率和混叠现象

**奈奎斯特频率详解**

奈奎斯特频率（Nyquist Frequency）是数字信号处理中的 fundamental 概念，定义为采样率的一半：
```
f_N = f_s / 2
```
其中 f_N 是奈奎斯特频率，f_s 是采样率。

**奈奎斯特采样定理**：
为了能够从采样信号中无失真地恢复原始信号，采样频率必须至少是信号中最高频率成分的两倍：
```
f_s ≥ 2 * f_max
```

**为什么重要？**
1. **避免混叠（Aliasing）**：当信号频率超过奈奎斯特频率时，高频信号会被"折叠"到低频区域
2. **实际应用**：
   - CD 音质采样率：44.1 kHz，奈奎斯特频率：22.05 kHz
   - 语音通信采样率：8 kHz，奈奎斯特频率：4 kHz

**混叠示例**：
```c
// 采样率 1000 Hz，奈奎斯特频率 500 Hz
#define SAMPLE_RATE 1000
#define NYQUIST_FREQ (SAMPLE_RATE / 2)

// 600 Hz 信号（超过奈奎斯特频率）
float freq_original = 600;  // 原始频率
float freq_aliased = SAMPLE_RATE - freq_original;  // 400 Hz（混叠后）
// 600 Hz 的信号会被误认为是 400 Hz！
```

**实际应用指南**：
- 采样率选择：采样率 = 2.5 ~ 4 × 信号最高频率（留有余量）
- 使用抗混叠滤波器在采样前限制信号带宽
- 过采样可以提高信噪比并简化滤波器设计

**常见误区**：
- 采样率 ≠ 奈奎斯特频率（奈奎斯特频率 = 采样率/2）
- 理论上 2 倍就够了，实际中需要更多余量

**学习任务**
```c
// 采样示例
#include <stdio.h>
#include <math.h>

#define SAMPLE_RATE 44100  // 采样率
#define NYQUIST_FREQ (SAMPLE_RATE / 2)  // 奈奎斯特频率

int main() {
    // 生成一个 10kHz 的正弦波
    float frequency = 10000;  // Hz

    if (frequency > NYQUIST_FREQ) {
        printf("警告：频率 %f Hz 超过奈奎斯特频率 %f Hz\n",
               frequency, NYQUIST_FREQ);
        printf("将产生混叠现象！\n");
    }

    return 0;
}
```

**实践练习**
1. 实现一个简单的采样程序
2. 观察不同采样率对信号的影响
3. 验证奈奎斯特定理

#### 1.1.2 时域与频域

**核心概念**
- 信号的两种表示方法
- 时域特征和频域特征
- 频谱的概念

**可视化工具**
```python
# 使用 Python 可视化时域和频域
import numpy as np
import matplotlib.pyplot as plt

# 生成信号
fs = 1000  # 采样率
t = np.arange(0, 1, 1/fs)  # 时间轴
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# 计算频谱
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/fs)

# 绘制
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('时域信号')
plt.xlabel('时间 (s)')

plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2],
         np.abs(fft_result)[:len(frequencies)//2])
plt.title('频域频谱')
plt.xlabel('频率 (Hz)')

plt.tight_layout()
plt.show()
```

### 1.2 傅里叶变换理论

#### 1.2.1 连续傅里叶变换（CFT）

**数学定义**
```
X(f) = ∫ x(t) * e^(-j2πft) dt
x(t) = ∫ X(f) * e^(j2πft) df
```

**关键性质**
- 线性性
- 时移性质
- 频移性质
- 卷积定理

#### 1.2.2 离散傅里叶变换（DFT）

**数学定义**
```
X[k] = Σ x[n] * e^(-j2πkn/N)  for n = 0 to N-1
x[n] = (1/N) Σ X[k] * e^(j2πkn/N)  for k = 0 to N-1
```

**手工计算示例**
计算序列 [1, 0, 1, 0] 的 DFT：

```
N = 4
X[0] = 1*e^0 + 0*e^0 + 1*e^0 + 0*e^0 = 2
X[1] = 1*e^0 + 0*e^(-jπ/2) + 1*e^(-jπ) + 0*e^(-j3π/2) = 0
X[2] = 1*e^0 + 0*e^(-jπ) + 1*e^(-j2π) + 0*e^(-j3π) = 2
X[3] = 1*e^0 + 0*e^(-j3π/2) + 1*e^(-j3π) + 0*e^(-j9π/2) = 0
```

#### 1.2.3 DFT 的复杂度分析

**直接计算**
- 每个 X[k] 需要 N 次复数乘法
- 总共需要 N^2 次复数乘法
- 时间复杂度：O(N^2)

```c
// 朴素的 DFT 实现
void dft_naive(double* input, double complex* output, int N) {
    for (int k = 0; k < N; k++) {
        output[k] = 0;
        for (int n = 0; n < N; n++) {
            double angle = -2 * M_PI * k * n / N;
            output[k] += input[n] * (cos(angle) + I * sin(angle));
        }
    }
}
```

### 1.3 FFT 算法原理

#### 1.3.1 Cooley-Tukey FFT 算法

**核心思想**
- 分治策略：将大问题分解为小问题
- 利用旋转因子（twiddle factor）的周期性
- 递归分解直到基例

**关键步骤**
1. 将输入序列分成偶数和奇数索引
2. 递归计算两个子序列的 DFT
3. 合并结果（蝶形运算）

**蝶形运算**
```
  X[k]   =   E[k] + W_N^k * O[k]
  X[k+N/2] = E[k] - W_N^k * O[k]

其中：
- E[k] 是偶数索引序列的 DFT
- O[k] 是奇数索引序列的 DFT
- W_N^k = e^(-j2πk/N) 是旋转因子
```

#### 1.3.2 FFT 复杂度分析

**计算复杂度**
- 每级分解：N/2 次蝶形运算
- 总共 log₂N 级
- 总运算量：N/2 * log₂N = O(N log N)

**与 DFT 对比**
```
N = 1024
- DFT: 1024^2 = 1,048,576 次运算
- FFT: 1024 * 10 = 10,240 次运算
- 加速比：约 100 倍！
```

### 1.4 KISS FFT 设计哲学

#### 1.4.1 Keep It Simple, Stupid

**设计原则**
- 优先考虑可读性和可维护性
- 避免过度优化
- 代码自文档化

**对比分析**
| 特性 | KISS FFT | FFTW |
|------|----------|------|
| 代码行数 | ~500 行 | >100,000 行 |
| 性能 | 良好 | 最优 |
| 学习曲线 | 平缓 | 陡峭 |
| 可移植性 | 极好 | 良好 |
| 许可证 | BSD | GPL |

#### 1.4.2 KISS FFT 的特点

**核心特性**
- 混合基数 FFT 实现
- 支持多种数据类型
- 无静态数据（线程安全）
- 简洁的 API 设计

**适用场景**
- 教学和学习
- 嵌入式系统
- 快速原型开发
- 不需要极致性能的应用

## 本周学习任务

### 第 1 周：信号处理基础

**周一/周二**
- [x] 阅读数字信号处理基础材料
- [x] 完成采样定理的理论学习
- [x] 实现采样率转换示例

**周三/周四**
- [x] 理解时域和频域概念
- [x] 运行 Python 可视化脚本
- [x] 分析不同信号的频谱特征

**周五**
- [ ] 总结本周学习内容
- [ ] 完成阶段测验

### 第 2 周：FFT 理论

**周一/周二**
- [ ] 深入学习傅里叶变换数学原理
- [ ] 手工计算简单 DFT 示例
- [ ] 实现 DFT 的朴素算法

**周三/周四**
- [ ] 理解 Cooley-Tukey FFT 算法
- [ ] 分析蝶形运算过程
- [ ] 对比 DFT 和 FFT 的复杂度

**周五**
- [ ] 阅读 KISS FFT README
- [ ] 理解 KISS FFT 的设计选择
- [ ] 准备进入下一阶段

## 评估练习

### 理论测验
1. 解释奈奎斯特采样定理
2. 推导 DFT 和 IDFT 的公式
3. 说明 FFT 相对于 DFT 的优势
4. 分析 KISS FFT 的设计权衡

### 编程练习
1. 实现一个简单的 DFT 函数
2. 验证 DFT 的线性性质
3. 实现一个基-2 的 FFT 算法
4. 比较不同实现的性能

## 常见问题

**Q: 为什么需要复数？**
A: 复数可以同时表示幅度和相位信息，是频域分析的自然表示。

**Q: FFT 只能处理长度为 2 的幂的序列吗？**
A: 不完全是。KISS FFT 支持混合基数，可以处理任意长度，但 2 的幂的效率最高。

**Q: 实数信号的频谱为什么是对称的？**
A: 这是由于实数信号 DFT 的共轭对称性质，一个重要特性可以节省计算量。

## 下一步

完成本阶段所有学习任务后，您已经准备好进入[阶段 2：核心代码探索](../phase2-core/)了！

---

记住：理论是实践的基础，扎实的理论理解将帮助您更好地掌握后续的代码实现。