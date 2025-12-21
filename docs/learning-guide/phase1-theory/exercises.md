# 阶段 1 练习题

## 理论练习

### 练习 1：采样定理

**问题 1.1**
给定一个连续信号：
```
x(t) = 3cos(2π·60t) + 2cos(2π·400t) + sin(2π·1000t)
```

a) 信号的最高频率是多少？
最高频率是 1000 HZ
b) 根据奈奎斯特定理，最小采样率应该是多少？
最小采样率应该是 2000 HZ
c) 如果采样率为 1000 Hz，哪些频率分量会产生混叠？
如果采样率为 1000 Hz，频率分量 1000 Hz 会产生混叠。

**答案提示：**
- 最高频率分量决定了所需的采样率
- 奈奎斯特频率 = fs/2
- 频率 f 在采样后会出现在 |f - n·fs| 位置

### 练习 2：DFT 计算

**问题 2.1**
手工计算序列 [1, 1, 0, 0] 的 4 点 DFT。

X[0] = 1 * e^0 + 1 * e^0 + 0 * e^0 + 0 * e^0 = 2
X[1] = 1 * e^0 + 1 * e^(-jπ/2) + 0 * e^(-jπ) + 0 * e^(-j3π/2) = 1 - 1j
X[2] = 1 * e^0 + 1 * e^(-jπ) + 0 * e^(-j2π) + 0 * e^(-j3π) = 0j
X[3] = 1 * e^0 + 1 * e^(-3jπ/2) + 0 * e^(-j6π/2) + 0 * e^(-j9π/2) = 1 + 1j

**问题 2.2**
证明 DFT 的线性性质：如果 y[n] = ax[n] + bz[n]，那么 Y[k] = aX[k] + bZ[k]。

**证明：**

根据DFT的定义，对于序列 y[n]，其DFT为：
Y[k] = Σ(n=0 to N-1) y[n] · e^(-j2πkn/N)

将 y[n] = ax[n] + bz[n] 代入：
Y[k] = Σ(n=0 to N-1) [ax[n] + bz[n]] · e^(-j2πkn/N)

利用求和的线性性质，展开求和：
Y[k] = Σ(n=0 to N-1) ax[n] · e^(-j2πkn/N) + Σ(n=0 to N-1) bz[n] · e^(-j2πkn/N)

将常数项 a 和 b 提取出来：
Y[k] = a Σ(n=0 to N-1) x[n] · e^(-j2πkn/N) + b Σ(n=0 to N-1) z[n] · e^(-j2πkn/N)

根据DFT的定义，这两个求和分别是 X[k] 和 Z[k]：
X[k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N)
Z[k] = Σ(n=0 to N-1) z[n] · e^(-j2πkn/N)

因此：
Y[k] = aX[k] + bZ[k]

证毕。

**问题 2.3**
解释为什么实数信号的 DFT 具有共轭对称性：X[k] = X*[N-k]。

**解答：**

对于实数信号 x[n]（其中 n = 0, 1, ..., N-1），其 DFT 为：
X[k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N)

考虑 X[N-k]：
X[N-k] = Σ(n=0 to N-1) x[n] · e^(-j2π(N-k)n/N)

展开指数项：
X[N-k] = Σ(n=0 to N-1) x[n] · e^(-j2πn) · e^(j2πkn/N)

由于 e^(-j2πn) = 1（因为 e^(j2π) = 1，且 n 为整数）：
X[N-k] = Σ(n=0 to N-1) x[n] · e^(j2πkn/N)

取 X[N-k] 的共轭：
X*[N-k] = [Σ(n=0 to N-1) x[n] · e^(j2πkn/N)]*

利用共轭的性质 (a·b)* = a*·b* 和 (e^(jθ))* = e^(-jθ)：
X*[N-k] = Σ(n=0 to N-1) x[n]* · e^(-j2πkn/N)

由于 x[n] 是实数，所以 x[n]* = x[n]：
X*[N-k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N)

这正是 X[k] 的定义：
X*[N-k] = X[k]

证毕。

**物理意义：**
1. 实数信号的频谱是共轭对称的
2. 这意味着我们只需要计算一半的频谱点（0 到 N/2），其余可以通过对称性获得
3. 对于实数信号，负频率分量不包含新的信息，它们只是正频率分量的共轭

### 练习 3：FFT 算法

**问题 3.1**
对于 N=8 的 FFT，需要多少级分解？每级有多少个蝶形运算？

**问题 3.2**
绘制基-2 FFT 的信号流图，标出旋转因子。

**问题 3.3**
比较 N=1024 时 DFT 和 FFT 的运算次数，计算加速比。

## 编程练习

### 练习 4：实现 DFT

**参考实现：**
```c
void my_dft(double *input, double complex *output, int N) {
    for (int k = 0; k < N; k++) {
        output[k] = 0;
        for (int n = 0; n < N; n++) {
            // 计算旋转因子: exp(-j*2*pi*k*n/N)
            double complex twiddle = cexp(-2.0 * PI * I * k * n / N);
            output[k] += input[n] * twiddle;
        }
    }
}
```

**测试用例：**
```c
double input[] = {1, 2, 3, 4};
double complex output[4];

my_dft(input, output, 4);

// 预期输出（近似值）：
// X[0] = 10 + 0i
// X[1] = -2 + 2i
// X[2] = -2 + 0i
// X[3] = -2 - 2i
```

### 练习 5：IDFT 实现

**参考实现：**
```c
void my_idft(double complex *input, double *output, int N) {
    for (int n = 0; n < N; n++) {
        double complex sum = 0;
        for (int k = 0; k < N; k++) {
            // 计算旋转因子: exp(j*2*pi*k*n/N)
            double complex twiddle = cexp(2.0 * PI * I * k * n / N);
            sum += input[k] * twiddle;
        }
        // 注意 1/N 的缩放因子
        output[n] = creal(sum) / N;
    }
}
```

**关键要点：**
- IDFT 与 DFT 的区别在于旋转因子的符号相反
- IDFT 需要乘以 1/N 的缩放因子
- 对于实数输入，IDFT 的输出也应该是实数（忽略数值误差）
- DFT 和 IDFT 组合使用应该能够完美重构原始信号

### 练习 6：频率检测

编写一个程序检测信号中的主要频率成分：

```c
#include <stdio.h>
#include <math.h>
#include <complex.h>

// 检测信号中的主要频率
void detect_frequencies(double *signal, int N, int fs) {
    // 1. 计算 DFT
    // 2. 找到幅值最大的几个频率
    // 3. 打印结果
}

int main() {
    // 测试信号：50 Hz + 120 Hz
    double fs = 1000;  // 采样率 1 kHz
    int N = 1024;
    double signal[N];

    for (int i = 0; i < N; i++) {
        double t = (double)i / fs;
        signal[i] = sin(2*PI*50*t) + 0.5*sin(2*PI*120*t);
    }

    detect_frequencies(signal, N, fs);
    return 0;
}
```

## 扩展练习

### 练习 7：窗函数

不同的窗函数会影响频谱的泄漏：

```c
// 实现汉明窗
void hamming_window(double *signal, int N) {
    // TODO: 应用汉明窗
    // w[n] = 0.54 - 0.46 * cos(2πn/(N-1))
}
```

**实验：**
1. 对同一个信号加窗和不加窗
2. 比较频谱的差异
3. 解释为什么加窗可以减少频谱泄漏

### 练习 8：实时频谱分析器

创建一个简单的命令行频谱分析器：

```c
#include <ncurses.h>  // 用于实时显示

// 实时显示频谱
void display_spectrum(double *spectrum, int num_bins) {
    // TODO: 使用字符绘制频谱柱状图
}

int main() {
    // 模拟实时音频输入
    while (1) {
        // 获取音频帧
        // 计算频谱
        // 显示结果
    }
    return 0;
}
```

## 项目练习

### 练习 9：音频均衡器

使用 DFT/FFT 实现一个简单的 3 段均衡器：

```c
typedef struct {
    double low_gain;    // 低频增益 (dB)
    double mid_gain;    // 中频增益 (dB)
    double high_gain;   // 高频增益 (dB)
} Equalizer;

void apply_equalizer(double *signal, int N, int fs, Equalizer *eq) {
    // 1. 计算频谱
    // 2. 调整不同频段的增益
    // 3. 逆变换回时域
}
```

### 练习 10：音频效果器

实现以下效果之一：
- 回声效果（延迟和混合）
- 合唱效果（调制）
- 失真效果（非线性变换）

## 实验报告

对于每个编程练习，准备一份简短的实验报告，包括：

1. **问题描述**
   - 清晰说明任务目标

2. **设计方案**
   - 算法选择和理由
   - 数据结构设计

3. **实现细节**
   - 关键代码片段
   - 遇到的问题和解决方案

4. **结果分析**
   - 测试数据
   - 性能分析
   - 与理论对比

5. **改进建议**
   - 可能的优化方向
   - 功能扩展想法

## 评估标准

### 理论理解 (40%)
- 概念的准确性
- 推理的严密性
- 表达的清晰度

### 编程实现 (40%)
- 代码的正确性
- 实现的效率
- 代码的可读性

### 实验分析 (20%)
- 结果的完整性
- 分析的深度
- 结论的合理性

## 参考答案

完成练习后，可以：
1. 对照理论验证结果
2. 与标准实现（如 numpy.fft）对比
3. 向同学或导师请教
4. 查阅相关资料

记住：编程练习的目的是加深理解，不要害怕犯错！