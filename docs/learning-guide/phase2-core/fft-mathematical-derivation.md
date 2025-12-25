# FFT 数学公式完整推导

## 目录

1. [从傅里叶级数到 DFT](#1-从傅里叶级数到-dft)
2. [DFT 的计算复杂度分析](#2-dft-的计算复杂度分析)
3. [Cooley-Tukey FFT 算法推导](#3-cooley-tukey-fft-算法推导)
4. [蝶形运算公式推导](#4-蝶形运算公式推导)
5. [数值考虑](#5-数值考虑)
6. [练习题](#练习题)

---

## 1. 从傅里叶级数到 DFT

### 1.1 傅里叶级数回顾

**傅里叶级数** 是傅里叶分析的起点，它告诉我们：**任何周期函数都可以表示为一系列正弦和余弦函数的加权和**。

对于一个周期为 $$T$$ 的连续周期信号 $$x(t)$$，其傅里叶级数表示为：

$$x(t) = a_0 + \sum_{n=1}^{\infty} \left[a_n \cos\left(\frac{2\pi nt}{T}\right) + b_n \sin\left(\frac{2\pi nt}{T}\right)\right]$$

其中系数为：

$$a_0 = \frac{1}{T}\int_0^T x(t)\,dt$$

$$a_n = \frac{2}{T}\int_0^T x(t)\cos\left(\frac{2\pi nt}{T}\right)\,dt$$

$$b_n = \frac{2}{T}\int_0^T x(t)\sin\left(\frac{2\pi nt}{T}\right)\,dt$$

**复数形式的傅里叶级数**更为简洁：

$$x(t) = \sum_{n=-\infty}^{\infty} c_n \cdot e^{j2\pi nt/T}$$

其中：

$$c_n = \frac{1}{T}\int_0^T x(t) \cdot e^{-j2\pi nt/T}\,dt$$

> **关键洞察**：傅里叶级数告诉我们，周期信号可以分解为不同频率的复指数分量。每个 $c_n$ 表示频率为 $\frac{n}{T}$ 的成分的强度。

---

### 1.2 连续时间傅里叶变换 (CTFT)

当周期 $T \to \infty$ 时，周期信号变为非周期信号，离散的频率谱变为连续的频率谱。

定义 **角频率**：$\Omega = \frac{2\pi}{T}$

当 $T \to \infty$ 时：
- $\Omega \to 0$（频率间隔趋于零）
- $n\Omega \to \omega$（离散频率变为连续频率）

**连续时间傅里叶变换 (CTFT)** 定义为：

$$X(\omega) = \int_{-\infty}^{\infty} x(t) \cdot e^{-j\omega t}\,dt$$

**逆变换**：

$$x(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} X(\omega) \cdot e^{j\omega t}\,d\omega$$

> **关键洞察**：非周期信号包含连续的频率成分，而非离散的谐波频率。

---

### 1.3 离散时间傅里叶变换 (DTFT)

对于**离散时间信号** $x[n]$（序列），我们需要对连续信号进行**采样**。

设采样周期为 $T_s$，采样频率为 $f_s = \frac{1}{T_s}$，角频率为 $\Omega_s = \frac{2\pi}{T_s}$。

将 $t = nT_s$ 代入 CTFT 的积分（求和近似积分）：

$$X(\Omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot e^{-j\Omega n T_s}$$

令 $\omega = \Omega T_s$（归一化角频率），得到：

**DTFT 正变换**：

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] \cdot e^{-j\omega n}$$

**DTFT 逆变换**：

$$x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega}) \cdot e^{j\omega n}\,d\omega$$

> **重要性质**：DTFT 的频谱 $X(e^{j\omega})$ 是周期的，周期为 $2\pi$。
> 这是因为 $e^{-j(\omega+2\pi)n} = e^{-j\omega n} \cdot e^{-j2\pi n} = e^{-j\omega n}$

---

### 1.4 离散傅里叶变换 (DFT)

DTFT 处理无限长序列和连续频率，但在计算机中我们只能处理：
1. **有限长序列**：$x[n]$ 仅在 $0 \leq n \leq N-1$ 有定义
2. **离散频率**：在频域也需要采样

#### 1.4.1 频域采样

对 DTFT 在频域均匀采样 $N$ 个点：

$$\omega_k = \frac{2\pi k}{N}, \quad k = 0, 1, \ldots, N-1$$

代入 DTFT 公式：

$$X(e^{j\omega_k}) = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi kn}{N}}$$

记为 $X[k]$：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi kn}{N}}$$

这就是 **N 点 DFT 的定义**！

#### 1.4.2 DFT 定义

**N 点离散傅里叶变换 (DFT)** 定义为：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot W_N^{kn}, \quad k = 0, 1, \ldots, N-1$$

其中 **旋转因子**（Twiddle Factor）：

$$W_N = e^{-j\frac{2\pi}{N}}$$

因此 $W_N^{kn} = e^{-j\frac{2\pi kn}{N}}$

**N 点离散傅里叶逆变换 (IDFT)** 定义为：

$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k] \cdot W_N^{-kn}, \quad n = 0, 1, \ldots, N-1$$

#### 1.4.3 IDFT 推导

证明 IDFT 确实能恢复原信号：

$$\begin{aligned}
x[n] &= \frac{1}{N}\sum_{k=0}^{N-1} X[k] \cdot W_N^{-kn} \\
&= \frac{1}{N}\sum_{k=0}^{N-1} \left(\sum_{m=0}^{N-1} x[m] \cdot W_N^{km}\right) \cdot W_N^{-kn} \\
&= \frac{1}{N}\sum_{m=0}^{N-1} x[m] \left(\sum_{k=0}^{N-1} W_N^{k(m-n)}\right)
\end{aligned}$$

利用**正交性**：

$$\sum_{k=0}^{N-1} W_N^{k(m-n)} = \begin{cases} N, & m-n \text{ 是 } N \text{ 的倍数} \\ 0, & \text{其他} \end{cases}$$

因此，当 $0 \leq m, n \leq N-1$ 时：

$$x[n] = \frac{1}{N}\sum_{m=0}^{N-1} x[m] \cdot N\delta[m-n] = x[n]$$

> **关键洞察**：DFT 将时域的 $N$ 个点映射到频域的 $N$ 个点，是完全可逆的线性变换。

---

## 2. DFT 的计算复杂度分析

### 2.1 直接计算 DFT 的复杂度

考虑直接使用 DFT 定义计算：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot W_N^{kn}$$

**分析**：
- 对于每个 $k$（$N$ 个），需要 $N$ 次复数乘法和 $N-1$ 次复数加法
- 总共需要 $N^2$ 次复数乘法和 $N(N-1)$ 次复数加法
- **计算复杂度**：$\mathcal{O}(N^2)$

**示例**：对于 $N = 1024$
- 复数乘法：$1024^2 = 1,048,576$ 次
- 这对于实时处理来说是不可接受的

### 2.2 DFT 矩阵表示

DFT 可以写成矩阵形式：

$$\begin{bmatrix} X[0] \\ X[1] \\ X[2] \\ \vdots \\ X[N-1] \end{bmatrix} = \begin{bmatrix} W_N^0 & W_N^0 & W_N^0 & \cdots & W_N^0 \\ W_N^0 & W_N^1 & W_N^2 & \cdots & W_N^{N-1} \\ W_N^0 & W_N^2 & W_N^4 & \cdots & W_N^{2(N-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ W_N^0 & W_N^{N-1} & W_N^{2(N-1)} & \cdots & W_N^{(N-1)^2} \end{bmatrix} \begin{bmatrix} x[0] \\ x[1] \\ x[2] \\ \vdots \\ x[N-1] \end{bmatrix}$$

记为：$\mathbf{X} = \mathbf{W}_N \cdot \mathbf{x}$

其中 $\mathbf{W}_N$ 是 $N \times N$ 的 DFT 矩阵。

### 2.3 旋转因子 $W_N^k$ 的性质

旋转因子 $W_N = e^{-j\frac{2\pi}{N}}$ 具有以下重要性质，这些性质是 FFT 优化算法的基础：

#### 2.3.1 周期性

$$W_N^{k+N} = W_N^k$$

**证明**：

$$W_N^{k+N} = e^{-j\frac{2\pi(k+N)}{N}} = e^{-j\frac{2\pi k}{N}} \cdot e^{-j2\pi} = W_N^k \cdot 1 = W_N^k$$

> **直观理解**：在复平面上，旋转 $2\pi$（360°）后回到原点。

#### 2.3.2 对称性（反比性）

$$W_N^{k+N/2} = -W_N^k$$

**证明**：

$$W_N^{k+N/2} = e^{-j\frac{2\pi(k+N/2)}{N}} = e^{-j\frac{2\pi k}{N}} \cdot e^{-j\pi} = W_N^k \cdot (-1) = -W_N^k$$

> **直观理解**：在复平面上，旋转 $\pi$（180°）后方向相反。

#### 2.3.3 共轭对称性

$$W_N^{-k} = W_N^{N-k} = (W_N^k)^*$$

其中 $^*$ 表示复共轭。

**证明**：

$$W_N^{-k} = e^{j\frac{2\pi k}{N}} = \left(e^{-j\frac{2\pi k}{N}}\right)^* = (W_N^k)^*$$

$$W_N^{N-k} = e^{-j\frac{2\pi(N-k)}{N}} = e^{-j2\pi} \cdot e^{j\frac{2\pi k}{N}} = 1 \cdot (W_N^k)^* = (W_N^k)^*$$

#### 2.3.4 可约性

$$W_N^{mk} = W_{N/m}^k$$

**证明**：

$$W_N^{mk} = e^{-j\frac{2\pi mk}{N}} = e^{-j\frac{2\pi k}{N/m}} = W_{N/m}^k$$

> **重要应用**：这是 FFT 分解的关键性质！

### 2.4 优化机会识别

基于旋转因子的性质，我们可以识别以下优化机会：

1. **重复计算**：$W_N^{kn}$ 中有很多值是相同的（周期性）
2. **对称性**：可以利用 $W_N^{k+N/2} = -W_N^k$ 减少计算
3. **可分解性**：可以将大 DFT 分解为小 DFT 的组合

这些洞察引出了 **快速傅里叶变换 (FFT)** 算法。

---

## 3. Cooley-Tukey FFT 算法推导

### 3.1 时间抽取算法 (DIT - Decimation In Time)

**核心思想**：将时域序列 $x[n]$ 按偶数和奇数下标分解，从而将 $N$ 点 DFT 分解为两个 $N/2$ 点 DFT。

#### 3.1.1 算法推导

**步骤 1**：将 DFT 求和按偶奇下标分组

$$\begin{aligned}
X[k] &= \sum_{n=0}^{N-1} x[n] \cdot W_N^{kn} \\
&= \sum_{r=0}^{N/2-1} x[2r] \cdot W_N^{k(2r)} + \sum_{r=0}^{N/2-1} x[2r+1] \cdot W_N^{k(2r+1)} \\
&= \sum_{r=0}^{N/2-1} x[2r] \cdot W_N^{2kr} + \sum_{r=0}^{N/2-1} x[2r+1] \cdot W_N^{k(2r+1)}
\end{aligned}$$

**步骤 2**：利用可约性 $W_N^2 = W_{N/2}$

$$W_N^{2kr} = \left(W_N^2\right)^{kr} = W_{N/2}^{kr}$$

**步骤 3**：提取公因子 $W_N^k$（从奇数项求和中）

$$\sum_{r=0}^{N/2-1} x[2r+1] \cdot W_N^{k(2r+1)} = W_N^k \cdot \sum_{r=0}^{N/2-1} x[2r+1] \cdot W_N^{2kr} = W_N^k \cdot \sum_{r=0}^{N/2-1} x[2r+1] \cdot W_{N/2}^{kr}$$

**步骤 4**：定义两个 $N/2$ 点 DFT

$$G[k] = \sum_{r=0}^{N/2-1} x[2r] \cdot W_{N/2}^{kr} = \text{DFT}_{N/2}\{x[2r]\}$$

$$H[k] = \sum_{r=0}^{N/2-1} x[2r+1] \cdot W_{N/2}^{kr} = \text{DFT}_{N/2}\{x[2r+1]\}$$

**步骤 5**：得到组合公式

$$X[k] = G[k] + W_N^k \cdot H[k], \quad k = 0, 1, \ldots, N-1$$

**问题**：$G[k]$ 和 $H[k]$ 只有 $N/2$ 个值，但我们需要 $N$ 个 $X[k]$ 值。

**解决方案**：利用 $G[k]$ 和 $H[k]$ 的**周期性**（$N/2$ 点 DFT 的周期为 $N/2$）

对于 $k \geq N/2$，令 $k' = k - N/2$：

$$X[k'+N/2] = G[k'] + W_N^{k'+N/2} \cdot H[k']$$

利用对称性 $W_N^{k'+N/2} = -W_N^{k'}$：

$$X[k'+N/2] = G[k'] - W_N^{k'} \cdot H[k']$$

**最终 DIT 蝶形运算公式**：

$$\boxed{\begin{aligned} X[k] &= G[k] + W_N^k \cdot H[k] \\ X[k+N/2] &= G[k] - W_N^k \cdot H[k] \end{aligned}} \quad k = 0, 1, \ldots, \frac{N}{2}-1$$

> **这就是基-2 蝶形运算！** 它将两个 $N/2$ 点 DFT 组合成一个 $N$ 点 DFT。

#### 3.1.2 递归算法

1. **分解**：将 $N$ 点 DFT 分解为两个 $N/2$ 点 DFT
2. **递归**：对每个 $N/2$ 点 DFT 继续分解，直到 $N=2$（基-2）或 $N=4$（基-4）
3. **组合**：使用蝶形运算将小 DFT 组合成大 DFT

**递归关系**：

$$T(N) = 2T(N/2) + \mathcal{O}(N)$$

根据主定理，$T(N) = \mathcal{O}(N\log N)$

#### 3.1.3 计算复杂度

- **乘法**：每级 $N/2$ 次乘以 $W_N^k$，共 $\log_2 N$ 级
  - 总计：$\frac{N}{2}\log_2 N$ 次复数乘法
- **加法**：每级 $N$ 次加法，共 $\log_2 N$ 级
  - 总计：$N\log_2 N$ 次复数加法

**对比**：直接 DFT 需要 $N^2$ 次操作

对于 $N = 1024$：
- 直接 DFT：$1024^2 = 1,048,576$
- FFT：$1024 \times 10 = 10,240$（约 100 倍加速！）

---

### 3.2 频率抽取算法 (DIF - Decimation In Frequency)

**核心思想**：将频域序列 $X[k]$ 按偶数和奇数下标分解，而不是分解时域。

#### 3.2.1 算法推导

**步骤 1**：将 DFT 求和分成两半

$$\begin{aligned}
X[k] &= \sum_{n=0}^{N-1} x[n] \cdot W_N^{kn} \\
&= \sum_{n=0}^{N/2-1} x[n] \cdot W_N^{kn} + \sum_{n=N/2}^{N-1} x[n] \cdot W_N^{kn}
\end{aligned}$$

**步骤 2**：在第二个求和中令 $m = n - N/2$

$$\sum_{n=N/2}^{N-1} x[n] \cdot W_N^{kn} = \sum_{m=0}^{N/2-1} x[m+N/2] \cdot W_N^{k(m+N/2)}$$

利用 $W_N^{k(m+N/2)} = W_N^{km} \cdot W_N^{kN/2} = W_N^{km} \cdot (-1)^k$

**步骤 3**：合并求和

$$X[k] = \sum_{n=0}^{N/2-1} \left(x[n] + (-1)^k \cdot x[n+N/2]\right) \cdot W_N^{kn}$$

**步骤 4**：分别考虑偶数和奇数 $k$

- **偶数** $k = 2r$：$(-1)^{2r} = 1$

$$X[2r] = \sum_{n=0}^{N/2-1} \underbrace{(x[n] + x[n+N/2])}_{x_1[n]} \cdot W_N^{2rn}$$

利用 $W_N^{2rn} = W_{N/2}^{rn}$：

$$X[2r] = \sum_{n=0}^{N/2-1} x_1[n] \cdot W_{N/2}^{rn} = \text{DFT}_{N/2}\{x_1[n]\}$$

- **奇数** $k = 2r+1$：$(-1)^{2r+1} = -1$

$$X[2r+1] = \sum_{n=0}^{N/2-1} \underbrace{(x[n] - x[n+N/2]) \cdot W_N^n}_{x_2[n]} \cdot W_N^{2rn}$$

$$X[2r+1] = \sum_{n=0}^{N/2-1} x_2[n] \cdot W_{N/2}^{rn} = \text{DFT}_{N/2}\{x_2[n]\}$$

**最终 DIF 公式**：

$$\boxed{\begin{aligned} x_1[n] &= x[n] + x[n+N/2] \\ x_2[n] &= (x[n] - x[n+N/2]) \cdot W_N^n \\ X[2r] &= \text{DFT}_{N/2}\{x_1[n]\} \\ X[2r+1] &= \text{DFT}_{N/2}\{x_2[n]\} \end{aligned}}$$

#### 3.2.2 DIT vs DIF

| 特性 | DIT（时间抽取） | DIF（频率抽取） |
|-----|----------------|----------------|
| 分解对象 | 时域 $x[n]$ 按偶奇分组 | 频域 $X[k]$ 按偶奇分组 |
| 输入顺序 | 需要位反转排序 | 自然顺序 |
| 输出顺序 | 自然顺序 | 需要位反转排序 |
| 计算复杂度 | 相同 | 相同 |

---

### 3.3 混合基数推广

Cooley-Tukey 算法不限于基-2，可以推广到任意基数。

设 $N = p \times q$（$p, q$ 为整数）

使用双索引表示：$n = q \cdot n_1 + n_0$，其中 $0 \leq n_1 < p$, $0 \leq n_0 < q$

$$\begin{aligned}
X[k] &= \sum_{n=0}^{N-1} x[n] \cdot W_N^{kn} \\ &= \sum_{n_0=0}^{q-1} \sum_{n_1=0}^{p-1} x[qn_1 + n_0] \cdot W_N^{k(qn_1 + n_0)} \\ &= \sum_{n_0=0}^{q-1} W_N^{kn_0} \sum_{n_1=0}^{p-1} x[qn_1 + n_0] \cdot (W_N^q)^{kn_1} \\ &= \sum_{n_0=0}^{q-1} W_N^{kn_0} \sum_{n_1=0}^{p-1} x[qn_1 + n_0] \cdot W_p^{kn_1}
\end{aligned}$$

内层求和是 $p$ 点 DFT，外层求和组合这些结果。

**常见基数**：
- **基-2**：$N = 2^m$（最常用）
- **基-4**：$N = 4^m$（更高效，更少乘法）
- **混合基**：如 $N = 2^m \times 3^n$（处理任意长度）

---

## 4. 蝶形运算公式推导

### 4.1 基-2 蝶形运算公式

我们已经推导出 DIT 基-2 蝶形运算公式：

$$\boxed{\begin{aligned} X[k] &= G[k] + W_N^k \cdot H[k] \\ X[k+N/2] &= G[k] - W_N^k \cdot H[k] \end{aligned}} \quad k = 0, 1, \ldots, \frac{N}{2}-1$$

其中：
- $G[k]$：偶数下标序列的 $N/2$ 点 DFT
- $H[k]$：奇数下标序列的 $N/2$ 点 DFT
- $W_N^k = e^{-j\frac{2\pi k}{N}}$：旋转因子

**蝶形图表示**：

```
          G[k] ───────┬────────→ X[k]
                       │
          W_N^k       │
          H[k] ────⊗──┴────────→ X[k+N/2]
                   │
                   └──→ (-1)──
```

**代码实现**（对应 `kf_bfly2` 函数）：

```c
// 蝶形运算公式:
// X[k]     = G[k] + W_N^k · H[k]
// X[k+N/2] = G[k] - W_N^k · H[k]

// 代码实现:
C_MUL(t, *Fout2, *tw1);   // t = W_N^k · H[k]
C_SUB(*Fout2, *Fout, t);  // Fout2 = G[k] - t = X[k+N/2]
C_ADDTO(*Fout, t);        // Fout = G[k] + t = X[k]

// 变量对应:
// Fout  ← G[k]
// Fout2 ← H[k]
// tw1   ← W_N^k
// t     ← W_N^k · H[k]
```

### 4.2 基-4 蝶形运算公式

对于 $N = 4^m$，我们可以一次分解为 4 个 $N/4$ 点 DFT。

将序列按 $n \mod 4$ 分组：
- $x[4r]$：下标模 4 余 0
- $x[4r+1]$：下标模 4 余 1
- $x[4r+2]$：下标模 4 余 2
- $x[4r+3]$：下标模 4 余 3

推导得到基-4 蝶形运算公式：

$$\boxed{\begin{aligned} X[k] &= G_0[k] + W_N^k \cdot G_1[k] + W_N^{2k} \cdot G_2[k] + W_N^{3k} \cdot G_3[k] \\ X[k+N/4] &= G_0[k] - j \cdot W_N^k \cdot G_1[k] - W_N^{2k} \cdot G_2[k] + j \cdot W_N^{3k} \cdot G_3[k] \\ X[k+N/2] &= G_0[k] - W_N^k \cdot G_1[k] + W_N^{2k} \cdot G_2[k] - W_N^{3k} \cdot G_3[k] \\ X[k+3N/4] &= G_0[k] + j \cdot W_N^k \cdot G_1[k] - W_N^{2k} \cdot G_2[k] - j \cdot W_N^{3k} \cdot G_3[k] \end{aligned}}$$

其中 $G_0, G_1, G_2, G_3$ 分别是四组的 $N/4$ 点 DFT。

**基-4 优势**：
- 更少的乘法（某些 $W_N^k$ 是 $\pm 1, \pm j$）
- 更好的内存访问模式
- 更少的级数（$\log_4 N$ 级 vs $\log_2 N$ 级）

**代码实现**（对应 `kf_bfly4` 函数）：

```c
// 基-4 蝶形运算需要更多临时变量存储中间结果
C_MUL(scratch[0], Fout[m],  *tw1);  // W_N^k * G_1[k]
C_MUL(scratch[1], Fout[m2], *tw2);  // W_N^{2k} * G_2[k]
C_MUL(scratch[2], Fout[m3], *tw3);  // W_N^{3k} * G_3[k]

// 然后通过加法和减法组合出四个输出
// 具体实现利用了对称性和特殊角度优化
```

### 4.3 旋转因子性质总结

| 性质 | 公式 | 应用 |
|-----|------|------|
| 周期性 | $W_N^{k+N} = W_N^k$ | 减少旋转因子存储 |
| 对称性 | $W_N^{k+N/2} = -W_N^k$ | 蝶形运算利用 |
| 共轭对称 | $(W_N^k)^* = W_N^{-k}$ | 实信号 FFT 优化 |
| 可约性 | $W_N^{mk} = W_{N/m}^k$ | FFT 分解基础 |
| 特殊角度 | $W_4^1 = -j, W_8^2 = -j$ | 基-4/基-8 优化 |

---

## 5. 数值考虑

### 5.1 浮点精度分析

**误差来源**：
1. **舍入误差**：每次浮点运算引入的舍入误差
2. **累积误差**：误差在多级计算中累积

**FFT 误差特性**：
- FFT 是**数值稳定**的，误差增长与 $\sqrt{N}$ 成正比
- 对于 $N = 1024$，典型误差约为 $10^{-12}$（双精度）

**缓解措施**：
- 使用双精度浮点数
- 避免 subtractive cancellation（相近数相减）
- 合理的运算顺序

### 5.2 定点数定标

KISS FFT 支持定点数实现，需要考虑**动态范围**和**精度**。

**定标策略**：
- 每级蝶形运算后可能溢出，需要右移（定标）
- 常用策略：每级右移 1 位（防止溢出）

**代码中的定标**（`C_FIXDIV` 宏）：

```c
C_FIXDIV(*Fout, 2);  // 定点数定标：除以 2（右移 1 位）
```

> **注意**：定点数会损失精度，但在嵌入式系统中节省资源。

### 5.3 误差累积分析

**理论误差界**（对于 $N$ 点 FFT）：

$$|\text{error}| \leq c \cdot \sqrt{N} \cdot \epsilon$$

其中：
- $c$：常数（与具体实现相关）
- $\epsilon$：机器精度（单精度约 $10^{-7}$，双精度约 $10^{-16}$）

**示例**：$N = 1024$，双精度
- 理论误差界：$c \cdot 32 \cdot 10^{-16} \approx 3 \times 10^{-14}$
- 实际测量误差：约 $10^{-13}$ 至 $10^{-14}$

---

## 练习题

### 练习 1：基础计算

给定 $N=4$，序列 $x[n] = \{1, 2, 3, 4\}$，计算：
1. 直接计算 $X[1]$（使用 DFT 定义）
2. 使用基-2 FFT 分解计算 $X[1]$
3. 验证两种方法结果相同

**答案**：

1. **直接计算**：
   $$W_4 = e^{-j\frac{2\pi}{4}} = e^{-j\frac{\pi}{2}} = -j$$
   $$X[1] = \sum_{n=0}^{3} x[n] \cdot W_4^n = 1 \cdot 1 + 2 \cdot (-j) + 3 \cdot (-1) + 4 \cdot j = -2 + 2j$$

2. **FFT 分解**：
   - 偶数序列：$g[n] = \{1, 3\}$
   - 奇数序列：$h[n] = \{2, 4\}$
   - $G[1] = 1 + 3 \cdot W_2^1 = 1 - 3 = -2$
   - $H[1] = 2 + 4 \cdot W_2^1 = 2 - 4 = -2$
   - 蝶形运算：$X[1] = G[1] + W_4^1 \cdot H[1] = -2 + (-j) \cdot (-2) = -2 + 2j$

3. 验证：结果相同 ✓

---

### 练习 2：旋转因子性质

证明以下恒等式：
1. $W_N^0 = 1$
2. $W_N^{N/4} = -j$（假设 $N$ 是 4 的倍数）
3. $W_N^{N/2} = -1$

**答案**：

1. $W_N^0 = e^{-j\frac{2\pi \cdot 0}{N}} = e^0 = 1$

2. $W_N^{N/4} = e^{-j\frac{2\pi \cdot N/4}{N}} = e^{-j\frac{\pi}{2}} = \cos\frac{\pi}{2} - j\sin\frac{\pi}{2} = 0 - j \cdot 1 = -j$

3. $W_N^{N/2} = e^{-j\frac{2\pi \cdot N/2}{N}} = e^{-j\pi} = \cos\pi - j\sin\pi = -1 - j \cdot 0 = -1$

---

### 练习 3：复杂度分析

对于 $N = 2048$，比较：
1. 直接 DFT 的复数乘法次数
2. 基-2 FFT 的复数乘法次数
3. 加速比

**答案**：

1. 直接 DFT：$N^2 = 2048^2 = 4,194,304$ 次复数乘法

2. 基-2 FFT：$\frac{N}{2}\log_2 N = 1024 \times 11 = 11,264$ 次复数乘法

3. 加速比：$\frac{4,194,304}{11,264} \approx 372.5$ 倍

---

### 练习 4：蝶形图绘制

绘制 $N=8$ 的基-2 DIT FFT 第一级蝶形图。

**答案**：

```
输入序列（位反转顺序）：
x[0] ──────┬────────→ (暂存)
            │
x[4] ──────┼── W₈⁰ ──→ 组合成 X[0] 和 X[4]
            │
x[2] ──────┼────────→ (暂存)
            │
x[6] ──────┼── W₈² ──→ 组合成 X[2] 和 X[6]
            │
x[1] ──────┼────────→ (暂存)
            │
x[5] ──────┼── W₈¹ ──→ 组合成 X[1] 和 X[5]
            │
x[3] ──────┼────────→ (暂存)
            │
x[7] ──────┴── W₈³ ──→ 组合成 X[3] 和 X[7]
```

完整蝶形图有 $\log_2 8 = 3$ 级。

---

### 练习 5：基-4 优化

对于 $N=16$，比较：
1. 基-2 FFT 需要的级数和蝶形运算次数
2. 基-4 FFT 需要的级数和蝶形运算次数

**答案**：

1. **基-2**：
   - 级数：$\log_2 16 = 4$ 级
   - 每级蝶形数：$16/2 = 8$ 个
   - 总蝶形运算：$4 \times 8 = 32$ 个

2. **基-4**：
   - 级数：$\log_4 16 = 2$ 级
   - 每级蝶形数：$16/4 = 4$ 个
   - 总蝶形运算：$2 \times 4 = 8$ 个

基-4 减少了级数和总运算量，但每个蝶形更复杂。

---

## 参考资源

### 经典论文
- Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series". *Mathematics of Computation*, 19(90), 297-301.

### 推荐书籍
- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
- Brigham, E. O. (1988). *The Fast Fourier Transform and Its Applications*. Prentice Hall.

### 在线资源
- [KISS FFT GitHub Repository](https://github.com/mborgerding/kissfft)
- [FFTW 文档](http://www.fftw.org/)
- [3Blue1Brown - Fourier Series 可视化](https://www.youtube.com/watch?v=r6sGWTCMz2k)

---

## 相关文档

- [蝶形运算详细说明](./butterfly-operations.md) - 深入讲解蝶形图的绘制和旋转因子
- [kf_bfly 代码分析](./kf_bfly-code-analysis.md) - KISS FFT 源码逐行解析
- [KISS FFT 剖析](./kiss_fft_anatomy.md) - 整体架构概览
