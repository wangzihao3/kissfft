# FFT 算法原理

## 概述

快速傅里叶变换（FFT）是计算 DFT 的高效算法，将计算复杂度从 O(N²) 降低到 O(N log N)。本章节将深入讲解 FFT 的核心思想和实现原理。

## 学习目标

完成本章节后，你应该能够：
- 理解为什么需要 FFT（DFT 的效率问题）
- 掌握 Cooley-Tukey FFT 算法的核心思想
- 理解蝶形运算（butterfly operation）
- 掌握时间抽取（DIT）和频率抽取（DIF）方法
- 理解混合基数 FFT 的优势
- 了解 FFT 的各种变体和应用

## 目录

1. [为什么需要 FFT](#1-为什么需要-fft)
2. [Cooley-Tukey FFT 算法](#2-cooley-tukey-fft-算法)
3. [蝶形运算](#3-蝶形运算)
4. [时间抽取 FFT（DIT）](#4-时间抽取-fft）
5. [频率抽取 FFT（DIF）](#5-频率抽取-fft)
6. [混合基数 FFT](#6-混合基数-fft)
7. [FFT 的计算复杂度](#7-fft-的计算复杂度)
8. [实际应用示例](#8-实际应用示例)
9. [练习题](#练习题)

---

## 1. 为什么需要 FFT

### 1.1 DFT 的计算问题

**DFT 定义：**
```
X[k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N),  k = 0, 1, ..., N-1
```

**计算量分析：**
- 每个 X[k] 需要 N 次复数乘法和 N-1 次复数加法
- 共有 N 个输出 X[k]
- 总计算量：**O(N²)** 次复数运算

**实际问题：**
```
N = 1024:    约 100 万次运算
N = 4096:    约 1600 万次运算
N = 65536:   约 40 亿次运算！
```

对于实时信号处理（如音频、视频），O(N²) 的复杂度是不可接受的。

### 1.2 FFT 的突破

**核心思想：** 利用 DFT 的对称性和周期性，将长序列 DFT 分解为短序列 DFT。

**关键观察：**
```
旋转因子的周期性：
W_N^(kN) = 1
W_N^(k + N/2) = -W_N^k

旋转因子的对称性：
W_N^(k + N/2) = -W_N^k
W_N^(2k) = W_(N/2)^k
```

**结果：** 计算复杂度从 O(N²) 降低到 **O(N log N)**

```
N = 1024:   FFT 约 10,000 次运算（比 DFT 快 100 倍！）
N = 65536:  FFT 约 200 万次运算（比 DFT 快 20,000 倍！）
```

### 1.3 FFT 的发展历史

- **1965**: Cooley 和 Tukey 发表经典 FFT 算法
- **1805**: Gauss 早就发现了类似算法（但未发表）
- **1964**: 为了研究地震数据，Cooley 和 Tukey 重新发明
- **现代**: FFT 被评为 20 世纪最重要的算法之一

---

## 2. Cooley-Tukey FFT 算法

### 2.1 基本思想：分而治之

将 N 点 DFT 分解为多个较小规模的 DFT：

```
N 点 DFT → N/2 点 DFT × 2
         → N/4 点 DFT × 4
         → ...
         → 2 点 DFT × N/2
```

### 2.2 前提条件

**N 必须是合数**（最好是 2 的幂）：
- N = 2^m：基-2 FFT（最常见）
- N = 4^m：基-4 FFT
- N = 2^a · 3^b · 5^c：混合基数 FFT

### 2.3 核心推导

假设 N 是偶数，将 x[n] 分成偶数索引和奇数索引：

```
x[n] → {x[0], x[2], x[4], ..., x[N-2]}  (偶数序列)
     → {x[1], x[3], x[5], ..., x[N-1]}  (奇数序列)
```

定义：
```
g[m] = x[2m]    (偶数序列，长度 N/2)
h[m] = x[2m+1]  (奇数序列，长度 N/2)
```

DFT 可以写成：
```
X[k] = Σ(n=0 to N-1) x[n] · W_N^(kn)
     = Σ(m=0 to N/2-1) x[2m] · W_N^(k·2m) + Σ(m=0 to N/2-1) x[2m+1] · W_N^(k(2m+1))

     = Σ(m=0 to N/2-1) g[m] · W_N^(2km) + W_N^k · Σ(m=0 to N/2-1) h[m] · W_N^(2km)

     = Σ(m=0 to N/2-1) g[m] · W_(N/2)^(km) + W_N^k · Σ(m=0 to N/2-1) h[m] · W_(N/2)^(km)
```

注意到：
```
W_N^(2km) = e^(-j2π·2km/N) = e^(-j2πkm/(N/2)) = W_(N/2)^(km)
```

因此：
```
X[k] = G[k] + W_N^k · H[k]

其中：
G[k] = DFT_{N/2}{g[m]}  (N/2 点 DFT)
H[k] = DFT_{N/2}{h[m]}  (N/2 点 DFT)
```

### 2.4 利用周期性

G[k] 和 H[k] 的周期是 N/2：
```
G[k + N/2] = G[k]
H[k + N/2] = H[k]
```

因此：
```
X[k + N/2] = G[k + N/2] + W_N^(k+N/2) · H[k + N/2]
          = G[k] + W_N^k · W_N^(N/2) · H[k]
          = G[k] + W_N^k · (-1) · H[k]
          = G[k] - W_N^k · H[k]
```

### 2.5 蝶形运算公式

```
X[k]     = G[k] + W_N^k · H[k]
X[k+N/2] = G[k] - W_N^k · H[k]
```

这就是**蝶形运算**（Butterfly Operation）！

---

## 3. 蝶形运算

### 3.1 基本蝶形

蝶形运算是 FFT 的基本计算单元：

```
        G[k] ────┬──── X[k] = G[k] + W·H[k]
                  │
                  ├─ W_N^k
                  │
        H[k] ────┴──── X[k+N/2] = G[k] - W·H[k]
```

**符号表示：**
```
     A ────┬──── A + W·B
            │
            ├─ W
            │
     B ────┴──── A - W·B
```

### 3.2 蝶形运算的硬件实现

```c
// 基本蝶形运算（C 语言）
void butterfly(kiss_fft_cpx *a, kiss_fft_cpx *b, kiss_fft_cpx twiddle) {
    kiss_fft_cpx t;

    // t = W · b
    t.r = twiddle.r * b->r - twiddle.i * b->i;
    t.i = twiddle.r * b->i + twiddle.i * b->r;

    // a' = a + t
    // b' = a - t
    {
        kiss_fft_cpx a_new = {a->r + t.r, a->i + t.i};
        kiss_fft_cpx b_new = {a->r - t.r, a->i - t.i};
        *a = a_new;
        *b = b_new;
    }
}
```

### 3.3 蝶形运算的数量

**N 点 FFT 需要：**
- 第 1 级：N/2 个蝶形
- 第 2 级：N/2 个蝶形
- ...
- 第 log₂N 级：N/2 个蝶形

**总计：** (N/2) · log₂N 个蝶形

**每个蝶形：**
- 1 次复数乘法（W·B）
- 2 次复数加法（A+t, A-t）

**总计算量：**
- 复数乘法：N/2 · log₂N
- 复数加法：N · log₂N

---

## 4. 时间抽取 FFT（DIT）

### 4.1 算法流程

DIT（Decimation-In-Time）将输入序列按时间索引分解：

```
步骤 1: 位反转排列（输入重排）
步骤 2: log₂N 级蝶形运算
步骤 3: 输出按自然顺序
```

### 4.2 位反转（Bit Reversal）

**为什么需要位反转？**

DIT FFT 将输入不断分成偶数和奇数索引，最终导致输入顺序需要位反转。

**示例：N = 8**

| 自然索引 n | 二进制 | 位反转 | 反转后索引 |
|-----------|--------|--------|-----------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

**输入重排：**
```
原序: {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]}
重排: {x[0], x[4], x[2], x[6], x[1], x[5], x[3], x[7]}
```

**位反转代码（C 语言）：**
```c
unsigned int bit_reverse(unsigned int n, unsigned int num_bits) {
    unsigned int reversed = 0;
    for (unsigned int i = 0; i < num_bits; i++) {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    return reversed;
}
```

### 4.3 DIT FFT 的信号流图（N=8）

```
          第 1 级        第 2 级        第 3 级
x[0] ────────○─────────────○─────────────○── X[0]
             │             │             │
x[4] ────────○─────────────│─────────────○── X[1]
             │             │             │
x[2] ────────○─────────────○─────────────│── X[2]
             │             │             │
x[6] ────────○─────────────│─────────────│── X[3]
             │             │             │
x[1] ────────○─────────────○─────────────│── X[4]
             │             │             │
x[5] ────────○─────────────│─────────────│── X[5]
             │             │             │
x[3] ────────○─────────────○─────────────│── X[6]
             │             │             │
x[7] ────────○─────────────│─────────────│── X[7]

其中 ○ 代表一个蝶形运算
```

### 4.4 DIT FFT 算法步骤

```c
void fft_dit(kiss_fft_cpx *x, int N) {
    // 1. 位反转排列输入
    bit_reverse(x, N);

    // 2. 执行 log₂N 级蝶形运算
    for (int stage = 0; stage < log2(N); stage++) {
        int butterfly_size = 1 << stage;  // 2^stage
        int num_groups = N / (2 * butterfly_size);

        for (int group = 0; group < num_groups; group++) {
            for (int k = 0; k < butterfly_size; k++) {
                int idx1 = 2 * butterfly_size * group + k;
                int idx2 = idx1 + butterfly_size;

                // 计算旋转因子
                int twiddle_idx = k * (N / (2 * butterfly_size));
                kiss_fft_cpx twiddle = W_N[twiddle_idx];

                // 蝶形运算
                butterfly(&x[idx1], &x[idx2], twiddle);
            }
        }
    }
}
```

### 4.5 DIT 特点

✅ **优点：**
- 输入重排（位反转）可以预先完成
- 蝶形运算结构规则，易于硬件实现
- 同址计算（in-place）节省内存

❌ **缺点：**
- 输入需要位反转
- 不如 DIF 直观

---

## 5. 频率抽取 FFT（DIF）

### 5.1 算法思想

DIF（Decimation-In-Frequency）将输出序列（频域）按频率索引分解：

```
X[k] 的偶数索引 → 一个 N/2 点 DFT
X[k] 的奇数索引 → 另一个 N/2 点 DFT
```

### 5.2 数学推导

从 DFT 定义出发，将 X[k] 分成偶数和奇数索引：

```
偶数索引：X[2k]
奇数索引：X[2k+1]
```

**偶数索引：**
```
X[2k] = Σ(n=0 to N-1) x[n] · W_N^(2kn)
      = Σ(n=0 to N/2-1) x[n] · W_N^(2kn) + Σ(n=N/2 to N-1) x[n] · W_N^(2kn)

令 m = n - N/2：
      = Σ(n=0 to N/2-1) x[n] · W_N^(2kn) + Σ(m=0 to N/2-1) x[m+N/2] · W_N^(2k(m+N/2))

      = Σ(n=0 to N/2-1) [x[n] + x[n+N/2]] · W_(N/2)^(kn)
```

**奇数索引：**
```
X[2k+1] = Σ(n=0 to N-1) x[n] · W_N^((2k+1)n)
        = Σ(n=0 to N/2-1) [x[n] - x[n+N/2]] · W_N^n · W_(N/2)^(kn)
```

### 5.3 DIF 蝶形

```
输入无需位反转
   │
   ├── 前半部分 + 后半部分 → 偶数频率 DFT
   │
   └── (前半部分 - 后半部分) × W_N^n → 奇数频率 DFT
```

**蝶形公式：**
```
a[n] = x[n] + x[n+N/2]
b[n] = (x[n] - x[n+N/2]) · W_N^n
```

### 5.4 DIF FFT 算法步骤

```c
void fft_dif(kiss_fft_cpx *x, int N) {
    // 1. 执行 log₂N 级蝶形运算（无需位反转）
    for (int stage = 0; stage < log2(N); stage++) {
        int butterfly_size = 1 << (log2(N) - stage - 1);

        for (int k = 0; k < N/2; k++) {
            int idx1 = k / butterfly_size * 2 * butterfly_size + k % butterfly_size;
            int idx2 = idx1 + butterfly_size;

            // 计算旋转因子
            int twiddle_idx = k % butterfly_size * (1 << stage);
            kiss_fft_cpx twiddle = W_N[twiddle_idx];

            // DIF 蝶形运算
            kiss_fft_cpx sum, diff;
            sum.r = x[idx1].r + x[idx2].r;
            sum.i = x[idx1].i + x[idx2].i;

            diff.r = x[idx1].r - x[idx2].r;
            diff.i = x[idx1].i - x[idx2].i;

            // 旋转因子乘法
            x[idx1] = sum;
            x[idx2].r = diff.r * twiddle.r - diff.i * twiddle.i;
            x[idx2].i = diff.r * twiddle.i + diff.i * twiddle.r;
        }
    }

    // 2. 位反转排列输出
    bit_reverse(x, N);
}
```

### 5.5 DIF 特点

✅ **优点：**
- 输入按自然顺序
- 数学推导更直观
- 适合并行实现

❌ **缺点：**
- 输出需要位反转
- 旋转因子位置不如 DIT 规则

---

## 6. 混合基数 FFT

### 6.1 为什么需要混合基数？

**问题：** N 不一定是 2 的幂怎么办？

**解决方案：** 使用混合基数 FFT

**常见情况：**
- N = 2^a · 3^b（基-2 和基-3 混合）
- N = 4^a（纯基-4，比基-2 快）
- N = 任意合数（Cooley-Tukey 通用算法）

### 6.2 Cooley-Tukey 通用算法

设 N = N₁ · N₂：

```
将二维索引 (n₁, n₂) 映射到一维索引 n：

n = n₁ + N₁ · n₂

其中：
0 ≤ n₁ < N₁
0 ≤ n₂ < N₂
```

**DFT 变换为：**
```
X[k₁ + N₁·k₂] = Σ(n₁=0 to N₁-1) Σ(n₂=0 to N₂-1)
                 x[n₁ + N₁·n₂] · W_N^((n₁+N₁·n₂)(k₁+N₁·k₂))
```

**分解为：**
1. N₁ 个 N₂ 点 DFT
2. 旋转因子相乘
3. N₂ 个 N₁ 点 DFT

### 6.3 基-4 FFT

对于 N = 4^m，基-4 FFT 比基-2 FFT 更高效：

**计算量对比：**
- 基-2: N/2 · log₄N 次复数乘法（每级 N/2 个蝶形）
- 基-4: N/4 · log₄N 次复数乘法（每级 N/4 个蝶形）

**优势：** 减少 25% 的复数乘法

**蝶形公式：**
```
X[k]     = A + W·B + W²·C + W³·D
X[k+N/4] = A - j·W·B - W²·C + j·W³·D
X[k+N/2] = A - W·B + W²·C - W³·D
X[k+3N/4] = A + j·W·B - W²·C - j·W³·D
```

### 6.4 KISS FFT 的混合基数策略

KISS FFT 支持任意 N（合数）：

```c
// KISS FFT 的分解策略
kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, 0, 0);

// 内部会：
// 1. 找出 N 的最小因子 p（基数）
// 2. 分解为 p 个 N/p 点 DFT
// 3. 递归处理直到 N 为素数
```

**示例：** N = 60 = 2² · 3 · 5

```
60 点 DFT → 2 个 30 点 DFT
         → 2 × [2 个 15 点 DFT]
         → 2 × 2 × [3 个 5 点 DFT]
         → 最终：8 个 5 点 DFT（小 N 可以直接计算）
```

---

## 7. FFT 的计算复杂度

### 7.1 复杂度对比

| 算法 | 复数乘法 | 复数加法 | 总运算量（N=1024） |
|------|---------|---------|------------------|
| 直接 DFT | N² | N(N-1) | ~1,000,000 |
| 基-2 FFT | (N/2)log₂N | N log₂N | ~5,120 (快 200 倍！) |
| 基-4 FFT | (N/4)log₄N | N log₄N | ~3,840 (快 260 倍！) |

### 7.2 实际性能因素

**影响性能的因素：**
1. **N 的大小**：越大，FFT 优势越明显
2. **N 的形式**：2 的幂最快
3. **硬件支持**：SIMD、GPU 加速
4. **实现优化**：旋转因子表、内存对齐

**经验法则：**
- N < 32：直接计算 DFT 更快
- 32 ≤ N ≤ 128：差距不大
- N > 128：FFT 明显更快
- N > 1024：FFT 必须使用

---

## 8. 实际应用示例

### 8.1 音频频谱分析

```c
// 采样率：44100 Hz
// FFT 长度：N = 1024
// 频率分辨率：Δf = 44100/1024 ≈ 43 Hz

float audio[1024];  // 音频样本
kiss_fft_cpx cx_in[1024], cx_out[1024];

// 1. 准备输入
for (int i = 0; i < 1024; i++) {
    cx_in[i].r = audio[i];
    cx_in[i].i = 0;
}

// 2. 执行 FFT
kiss_fft_cfg cfg = kiss_fft_alloc(1024, 0, 0, 0);
kiss_fft(cfg, cx_in, cx_out);

// 3. 提取幅度谱
float magnitude[512];  // 只需要前 N/2 个（实信号）
for (int i = 0; i < 512; i++) {
    magnitude[i] = sqrt(cx_out[i].r * cx_out[i].r +
                        cx_out[i].i * cx_out[i].i) / 512;
}

// 4. 映射到频率
float freq = i * 43.0f;  // 第 i 个频率分量的频率
```

### 8.2 卷积定理应用

```
时域卷积 = 频域乘法

y[n] = x[n] * h[n]
Y[k] = X[k] · H[k]

步骤：
1. FFT{x[n]} → X[k]
2. FFT{h[n]} → H[k]
3. Y[k] = X[k] · H[k]
4. IFFT{Y[k]} → y[n]

复杂度：
- 直接卷积：O(N²)
- FFT 卷积：O(N log N)
```

---

## 练习题

### 基础练习

1. **蝶形计算**
   已知 G[0] = 3+j2, H[0] = 1-j, W_8^0 = 1，计算 X[0] 和 X[4]。

2. **位反转**
   对于 N = 16，计算索引 n = 13 的位反转值。

3. **旋转因子**
   计算 W_16^3 的值（用 a+jb 形式表示）。

### 进阶练习

4. **DIT vs DIF**
   比较 8 点 DIT FFT 和 DIF FFT 的信号流图，找出它们的区别。

5. **复杂度计算**
   推导基-4 FFT 的复数乘法次数，与基-2 FFT 对比。

6. **设计题**
   设计一个 N = 12 点 FFT 的分解方案（12 = 3 × 4）。

### 挑战练习

7. **混合基数实现**
   实现 N = 18 = 2 × 3² 的混合基数 FFT。

8. **优化问题**
   如何减少 FFT 中的旋转因子乘法次数？

---

## 总结

本章节介绍了 FFT 算法的核心原理：

1. **为什么需要 FFT**：DFT 的 O(N²) 复杂度太高
2. **Cooley-Tukey FFT**：分而治之，将长 DFT 分解为短 DFT
3. **蝶形运算**：FFT 的基本计算单元
4. **DIT FFT**：时间抽取，输入需位反转
5. **DIF FFT**：频率抽取，输出需位反转
6. **混合基数 FFT**：支持任意合数 N
7. **计算复杂度**：O(N log N)，比 DFT 快几个数量级

**下一步**：学习 KISS FFT 的具体实现，看看这些理论如何应用到实际代码中。

---

## 参考资源

- 《快速傅里叶变换及其应用》- Brigham
- Cooley, J.W., & Tukey, J.W. (1965). "An algorithm for the machine calculation of complex Fourier series"
- [FFT Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- [Understanding the FFT](https://www.youtube.com/watch?v=iTMn0Kt18tg)
