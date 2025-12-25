# kf_bfly 代码分析

## 目录

1. [函数概览](#1-函数概览)
2. [kf_bfly2 详解](#2-kf_bfly2-详解)
3. [kf_bfly4 详解](#3-kf_bfly4-详解)
4. [kf_bfly3 和 kf_bfly5](#4-kf_bfly3-和-kf_bfly5)
5. [kf_bfly_generic](#5-kf_bfly_generic)
6. [宏定义分析](#6-宏定义分析)
7. [代码-公式对照表](#7-代码-公式对照表)
8. [优化技巧总结](#8-优化技巧总结)

---

## 1. 函数概览

### 1.1 kf_bfly 函数族

KISS FFT 实现了一组蝶形运算函数，用于处理不同基数的 FFT：

| 函数 | 文件位置 | 基数 | 用途 |
|-----|---------|------|------|
| `kf_bfly2` | kiss_fft.c:15 | 2 | 基-2 蝶形运算 |
| `kf_bfly3` | kiss_fft.c:86 | 3 | 基-3 蝶形运算 |
| `kf_bfly4` | kiss_fft.c:38 | 4 | 基-4 蝶形运算 |
| `kf_bfly5` | kiss_fft.c:130 | 5 | 基-5 蝶形运算 |
| `kf_bfly_generic` | kiss_fft.c:192 | 任意 | 通用蝶形运算 |

### 1.2 参数含义详解

所有蝶形函数共享相似的参数：

```c
static void kf_bfly2(
    kiss_fft_cpx * Fout,      // 输出/输入数组（原位计算）
    const size_t fstride,     // 旋转因子步长
    const kiss_fft_cfg st,    // FFT 配置（包含旋转因子表）
    int m                      // 当前阶段的蝶形数量
);
```

**参数详解**：

| 参数 | 类型 | 含义 | 示例值 |
|-----|------|------|--------|
| `Fout` | `kiss_fft_cpx*` | 指向待处理数据的指针（原位计算） | 指向复数数组 |
| `fstride` | `size_t` | 旋转因子表索引步长 | 第1级: 1, 第2级: 2, ... |
| `st` | `kiss_fft_cfg` | FFT 配置结构体 | 包含 `twiddles` 表 |
| `m` | `int` | 本阶段需要处理的蝶形数量 | $N/2, N/4, N/8, ...$ |

**fstride 的含义**：

对于 $N$ 点 FFT，旋转因子表有 $N$ 个元素：
- 第 1 级：使用 `twiddles[0], twiddles[fstride], twiddles[2*fstride], ...`
- 第 2 级：`fstride` 翻倍，使用 `twiddles[0], twiddles[2*fstride], ...`

### 1.3 调用关系图

```
kiss_fft_stride()
    ↓
kf_work()  (递归分解)
    ↓
    ├→ kf_bfly2()  (p=2)
    ├→ kf_bfly3()  (p=3)
    ├→ kf_bfly4()  (p=4)
    ├→ kf_bfly5()  (p=5)
    └→ kf_bfly_generic()  (其他)
```

**关键调用点**（kiss_fft.c:293-299）：

```c
switch (p) {
    case 2: kf_bfly2(Fout,fstride,st,m); break;
    case 3: kf_bfly3(Fout,fstride,st,m); break;
    case 4: kf_bfly4(Fout,fstride,st,m); break;
    case 5: kf_bfly5(Fout,fstride,st,m); break;
    default: kf_bfly_generic(Fout,fstride,st,m,p); break;
}
```

---

## 2. kf_bfly2 详解

### 2.1 函数签名和参数

```c
static void kf_bfly2(
    kiss_fft_cpx * Fout,      // 输入/输出数组
    const size_t fstride,     // 旋转因子步长
    const kiss_fft_cfg st,    // FFT 配置
    int m                      // 蝶形数量
)
```

**数学对应**：基-2 时间抽取蝶形运算

$$\begin{aligned} X[k] &= G[k] + W_N^k \cdot H[k] \\ X[k+N/2] &= G[k] - W_N^k \cdot H[k] \end{aligned}$$

### 2.2 逐行代码解析

```c
// kiss_fft.c:15-36
static void kf_bfly2(
    kiss_fft_cpx * Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    int m
)
{
    kiss_fft_cpx * Fout2;           // 指向第二个输入 (H[k])
    kiss_fft_cpx * tw1 = st->twiddles;  // 旋转因子指针
    kiss_fft_cpx t;                 // 临时变量：存储 W_N^k * H[k]
    Fout2 = Fout + m;               // Fout2 = Fout + m (指向 H[k])
```

> **注释说明**：
> - `Fout` 指向 $G[k]$
> - `Fout2` 指向 $H[k]$（与 $G[k]$ 间距 $m$）
> - `tw1` 从旋转因子表起始位置开始

```c
    do{
        C_FIXDIV(*Fout,2);          // 定点数定标：除以 2
        C_FIXDIV(*Fout2,2);         // 定点数定标：除以 2

        C_MUL(t, *Fout2, *tw1);     // t = Fout2 * tw1 = W_N^k * H[k]
        tw1 += fstride;             // 前进到下一个旋转因子
        C_SUB(*Fout2, *Fout, t);    // Fout2 = Fout - t = G[k] - W*H[k] = X[k+N/2]
        C_ADDTO(*Fout, t);          // Fout = Fout + t = G[k] + W*H[k] = X[k]
        ++Fout2;                    // 前进到下一个 H[k]
        ++Fout;                     // 前进到下一个 G[k]
    }while (--m);                   // 循环 m 次
}
```

### 2.3 与蝶形公式对照

| 代码行 | 数学操作 | 含义 |
|-------|---------|------|
| `C_MUL(t, *Fout2, *tw1)` | $t = W_N^k \cdot H[k]$ | 计算旋转后的 $H[k]$ |
| `C_SUB(*Fout2, *Fout, t)` | $X[k+N/2] = G[k] - t$ | 下半个输出 |
| `C_ADDTO(*Fout, t)` | $X[k] = G[k] + t$ | 上半个输出 |

**变量映射**：

```
代码变量      数学符号
─────────────────────────
Fout      ←  G[k]       (偶数下标 DFT)
Fout2     ←  H[k]       (奇数下标 DFT)
tw1       ←  W_N^k      (旋转因子)
t         ←  W_N^k * H[k] (旋转后的 H[k])
```

### 2.4 循环结构分析

**循环目的**：处理 $m$ 个基-2 蝶形运算

**数据布局**：

```
Fout[0]   Fout[1]   ...   Fout[m-1]   Fout[m]   Fout[m+1] ... Fout[2m-1]
  ↓         ↓              ↓            ↓          ↓            ↓
 G[0]      G[1]    ...    G[m-1]       H[0]       H[1]   ...    H[m-1]
   └─────────┬─────────┘               └──────────┬──────────┘
        第一组（偶数）                   第二组（奇数）
```

**每次迭代**：
1. 处理一对 $(G[i], H[i])$
2. 计算 $X[i]$ 和 $X[i+N/2]$
3. 原位更新数据
4. 前进到下一对

---

## 3. kf_bfly4 详解

### 3.1 基-4 优化原理

**基-4 蝶形公式**（简化版）：

$$\begin{aligned} X[k] &= G_0[k] + W_N^k \cdot G_1[k] + W_N^{2k} \cdot G_2[k] + W_N^{3k} \cdot G_3[k] \\ X[k+N/4] &= G_0[k] - j \cdot W_N^k \cdot G_1[k] - W_N^{2k} \cdot G_2[k] + j \cdot W_N^{3k} \cdot G_3[k] \\ X[k+N/2] &= G_0[k] - W_N^k \cdot G_1[k] + W_N^{2k} \cdot G_2[k] - W_N^{3k} \cdot G_3[k] \\ X[k+3N/4] &= G_0[k] + j \cdot W_N^k \cdot G_1[k] - W_N^{2k} \cdot G_2[k] - j \cdot W_N^{3k} \cdot G_3[k] \end{aligned}$$

**优化技巧**：
1. 利用特殊角度（$W_N^k$ 可能是 $\pm 1, \pm j$）
2. 减少复数乘法次数
3. 利用对称性

### 3.2 scratch 数组使用分析

```c
// kiss_fft.c:38-84
static void kf_bfly4(
    kiss_fft_cpx * Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    const size_t m
)
{
    kiss_fft_cpx *tw1,*tw2,*tw3;
    kiss_fft_cpx scratch[6];          // 6 个临时变量
    size_t k=m;
    const size_t m2=2*m;              // 2m
    const size_t m3=3*m;              // 3m

    tw3 = tw2 = tw1 = st->twiddles;   // 三个旋转因子指针

    do {
        // 定标（定点数）
        C_FIXDIV(*Fout,4);
        C_FIXDIV(Fout[m],4);
        C_FIXDIV(Fout[m2],4);
        C_FIXDIV(Fout[m3],4);

        // 计算旋转后的值
        C_MUL(scratch[0], Fout[m],  *tw1);   // scratch[0] = Fout[m] * W_N^k
        C_MUL(scratch[1], Fout[m2], *tw2);   // scratch[1] = Fout[m2] * W_N^{2k}
        C_MUL(scratch[2], Fout[m3], *tw3);   // scratch[2] = Fout[m3] * W_N^{3k}
```

**scratch 数组用途**：

| 索引 | 内容 | 用途 |
|-----|------|------|
| 0 | $Fout[m] \cdot W_N^k$ | 旋转后的 $G_1[k]$ |
| 1 | $Fout[m2] \cdot W_N^{2k}$ | 旋转后的 $G_2[k]$ |
| 2 | $Fout[m3] \cdot W_N^{3k}$ | 旋转后的 $G_3[k]$ |
| 3 | $scratch[0] + scratch[2]$ | 中间和 |
| 4 | $scratch[0] - scratch[2]$ | 中间差 |
| 5 | $Fout[0] - scratch[1]$ | 中间值 |

### 3.3 复数运算优化技巧

```c
        // 组合计算
        C_SUB(scratch[5], *Fout, scratch[1]);     // scratch[5] = Fout - scratch[1]
        C_ADDTO(*Fout, scratch[1]);               // Fout += scratch[1]
        C_ADD(scratch[3], scratch[0], scratch[2]); // scratch[3] = s0 + s2
        C_SUB(scratch[4], scratch[0], scratch[2]); // scratch[4] = s0 - s2
        C_SUB(Fout[m2], *Fout, scratch[3]);       // Fout[m2] = Fout - scratch[3]
        tw1 += fstride;                           // 前进旋转因子
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO(*Fout, scratch[3]);               // Fout += scratch[3]
```

**关键优化**：
- 通过中间变量 `scratch[3]` 和 `scratch[4]` 重用计算结果
- 避免重复计算相同的加法和减法

### 3.4 正/逆变换差异处理

```c
        if(st->inverse) {
            // 逆变换：IDFT
            Fout[m].r  = scratch[5].r - scratch[4].i;
            Fout[m].i  = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            // 正变换：DFT
            Fout[m].r  = scratch[5].r + scratch[4].i;
            Fout[m].i  = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}
```

**为什么正/逆变换不同？**

- **DFT** 使用 $W_N = e^{-j2\pi/N}$
- **IDFT** 使用 $W_N^{-1} = e^{j2\pi/N}$
- 等价于复共轭：符号相反

**代码技巧**：直接修改实部和虚部的符号，避免额外的乘法运算。

---

## 4. kf_bfly3 和 kf_bfly5

### 4.1 非基数 2 的处理

**应用场景**：
- $N$ 不是 2 的幂（如 $N=12, 18, 25$）
- 混合基数分解（如 $60 = 3 \times 4 \times 5$）

### 4.2 特殊角度优化分析

**kf_bfly3 优化**（kiss_fft.c:86-128）：

```c
static void kf_bfly3(
    kiss_fft_cpx * Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    size_t m
)
{
    size_t k=m;
    const size_t m2 = 2*m;
    kiss_fft_cpx *tw1,*tw2;
    kiss_fft_cpx scratch[5];
    kiss_fft_cpx epi3;
    epi3 = st->twiddles[fstride*m];  // 预取特殊角度 W_N^{N/3}
```

**关键优化**：`epi3` 是 $W_N^{N/3}$，这是一个特殊角度：
- $e^{-j2\pi/3} = -\frac{1}{2} - j\frac{\sqrt{3}}{2}$

```c
    do{
        C_FIXDIV(*Fout,3);
        C_FIXDIV(Fout[m],3);
        C_FIXDIV(Fout[m2],3);

        C_MUL(scratch[1], Fout[m],  *tw1);
        C_MUL(scratch[2], Fout[m2], *tw2);

        C_ADD(scratch[3], scratch[1], scratch[2]);
        C_SUB(scratch[0], scratch[1], scratch[2]);
        tw1 += fstride;
        tw2 += fstride*2;

        // 利用特殊性质减少乘法
        Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
        Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

        C_MULBYSCALAR(scratch[0], epi3.i);  // 只乘以虚部

        C_ADDTO(*Fout, scratch[3]);

        Fout[m2].r = Fout[m].r + scratch[0].i;
        Fout[m2].i = Fout[m].i - scratch[0].r;

        Fout[m].r -= scratch[0].i;
        Fout[m].i += scratch[0].r;

        ++Fout;
    }while(--k);
}
```

**优化点**：
1. `HALF_OF` 宏实现移位或乘以 0.5
2. `C_MULBYSCALAR` 只用标量乘法（而非完整复数乘法）

### 4.3 复数乘法减少技巧

**kf_bfly5 类似的优化**（kiss_fft.c:130-189）：

```c
// 预取特殊角度
ya = twiddles[fstride*m];    // W_N^{N/5}
yb = twiddles[fstride*2*m];  // W_N^{2N/5}

// 利用这些角度的对称性和周期性减少运算
scratch[5].r = scratch[0].r + S_MUL(scratch[7].r, ya.r) + S_MUL(scratch[8].r, yb.r);
scratch[5].i = scratch[0].i + S_MUL(scratch[7].i, ya.r) + S_MUL(scratch[8].i, yb.r);

// 只用实部乘法（S_MUL），避免完整复数乘法
```

---

## 5. kf_bfly_generic

### 5.1 通用算法原理

**处理任意基数 $p$** 的蝶形运算。

```c
// kiss_fft.c:192-233
static void kf_bfly_generic(
    kiss_fft_cpx * Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    int m,
    int p    // 任意基数
)
{
    int u,k,q1,q;
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    // 分配临时缓冲区
    kiss_fft_cpx * scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_cpx)*p);
    if (scratch == NULL) {
        KISS_FFT_ERROR("Memory allocation failed.");
        return;
    }
```

### 5.2 双重循环分析

```c
    for (u=0; u<m; ++u) {           // 外层循环：处理 m 个蝶形组
        k=u;
        for (q1=0; q1<p; ++q1) {    // 内层循环：复制 p 个输入
            scratch[q1] = Fout[k];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for (q1=0; q1<p; ++q1) {    // 内层循环：计算 p 个输出
            int twidx=0;
            Fout[k] = scratch[0];
            for (q=1; q<p; ++q) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;  // 模运算
                C_MUL(t, scratch[q], twiddles[twidx]);
                C_ADDTO(Fout[k], t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}
```

**循环结构**：

1. **外层循环**（`u`）：处理 $m$ 个基-$p$ 蝶形
2. **内层循环 1**：复制 $p$ 个输入到 `scratch`
3. **内层循环 2**：计算 $p$ 个输出，使用旋转因子组合

**旋转因子索引计算**：

```c
twidx += fstride * k;
if (twidx>=Norig) twidx-=Norig;  // 模 N 运算
```

这实现了索引的周期性（利用 $W_N^{k+N} = W_N^k$）。

### 5.3 性能权衡讨论

| 特性 | kf_bfly2/3/4/5 | kf_bfly_generic |
|-----|----------------|-----------------|
| 适用范围 | 特定基数 | 任意基数 |
| 代码大小 | 较大（多个函数） | 较小（一个函数） |
| 运算速度 | 快（优化） | 慢（通用） |
| 内存分配 | 无动态分配 | 需要临时缓冲区 |

**使用建议**：
- 优先使用基-2/4（最常见）
- 基-3/5 用于特定长度（如 $N=12, 25$）
- `kf_bfly_generic` 作为后备（处理大质数）

---

## 6. 宏定义分析

### 6.1 C_ADD, C_SUB, C_MUL

**定义位置**：`_kiss_fft_guts.h`

#### C_ADD - 复数加法

```c
#define C_ADD(res, a, b) \
    do { \
        CHECK_OVERFLOW_OP((a).r, +, (b).r) \
        CHECK_OVERFLOW_OP((a).i, +, (b).i) \
        (res).r = (a).r + (b).r; \
        (res).i = (a).i + (b).i; \
    }while(0)
```

**数学对应**：$res = a + b = (a_r + b_r) + j(a_i + b_i)$

#### C_SUB - 复数减法

```c
#define C_SUB(res, a, b) \
    do { \
        CHECK_OVERFLOW_OP((a).r, -, (b).r) \
        CHECK_OVERFLOW_OP((a).i, -, (b).i) \
        (res).r = (a).r - (b).r; \
        (res).i = (a).i - (b).i; \
    }while(0)
```

**数学对应**：$res = a - b = (a_r - b_r) + j(a_i - b_i)$

#### C_MUL - 复数乘法

```c
// 浮点版本
#define C_MUL(m, a, b) \
    do { \
        (m).r = (a).r*(b).r - (a).i*(b).i; \
        (m).i = (a).r*(b).i + (a).i*(b).r; \
    }while(0)

// 定点版本（带舍入）
#define C_MUL(m, a, b) \
    do { \
        (m).r = sround(smul((a).r, (b).r) - smul((a).i, (b).i)); \
        (m).i = sround(smul((a).r, (b).i) + smul((a).i, (b).r)); \
    }while(0)
```

**数学推导**：

$$\begin{aligned} (a_r + ja_i) \cdot (b_r + jb_i) &= a_r b_r + ja_r b_i + ja_i b_r + j^2 a_i b_i \\ &= (a_r b_r - a_i b_i) + j(a_r b_i + a_i b_r) \end{aligned}$$

### 6.2 C_FIXDIV 定标分析

```c
// 定点版本
#define C_FIXDIV(c, div) \
    do { \
        DIVSCALAR((c).r, div); \
        DIVSCALAR((c).i, div); \
    }while(0)

#define DIVSCALAR(x, k) \
    (x) = sround(smul(x, SAMP_MAX/k))

// 浮点版本（空操作）
#define C_FIXDIV(c, div) /* NOOP */
```

**为什么需要定标？**

在定点数 FFT 中，每级蝶形运算可能使数值范围增大（理论上最多翻倍）。为防止溢出：
- 每级右移 1 位（除以 2）
- 累积：$\log_2 N$ 级后，总共右移 $\log_2 N$ 位

**示例**：$N=1024 = 2^{10}$
- 第 1 级：除以 2
- 第 2 级：除以 2
- ...
- 第 10 级：除以 2
- 总计：除以 $2^{10} = 1024$

### 6.3 其他优化宏

#### HALF_OF - 取半

```c
#define HALF_OF(x) ((x)*((kiss_fft_scalar).5))  // 浮点
#define HALF_OF(x) ((x)>>1)                     // 定点
```

#### C_MULBYSCALAR - 标量乘法

```c
#define C_MULBYSCALAR(c, s) \
    do { \
        (c).r *= (s); \
        (c).i *= (s); \
    }while(0)
```

#### C_ADDTO, C_SUBFROM - 原位加/减

```c
#define C_ADDTO(res, a) \
    do { \
        (res).r += (a).r; \
        (res).i += (a).i; \
    }while(0)

#define C_SUBFROM(res, a) \
    do { \
        (res).r -= (a).r; \
        (res).i -= (a).i; \
    }while(0)
```

---

## 7. 代码-公式对照表

### 7.1 kf_bfly2 对照表

| 数学公式 | 代码 | 说明 |
|---------|------|------|
| $X[k] = G[k] + W_N^k \cdot H[k]$ | `C_ADDTO(*Fout, t)` | 上半个输出 |
| $X[k+N/2] = G[k] - W_N^k \cdot H[k]$ | `C_SUB(*Fout2, *Fout, t)` | 下半个输出 |
| $t = W_N^k \cdot H[k]$ | `C_MUL(t, *Fout2, *tw1)` | 旋转因子乘法 |
| $G[k]$ | `*Fout` | 偶数下标 DFT |
| $H[k]$ | `*Fout2` | 奇数下标 DFT |
| $W_N^k$ | `*tw1` | 旋转因子 |

### 7.2 kf_bfly4 对照表

| 数学公式 | 代码变量 | 说明 |
|---------|---------|------|
| $G_0[k]$ | `*Fout` | 模 4 余 0 的 DFT |
| $G_1[k]$ | `Fout[m]` | 模 4 余 1 的 DFT |
| $G_2[k]$ | `Fout[m2]` | 模 4 余 2 的 DFT |
| $G_3[k]$ | `Fout[m3]` | 模 4 余 3 的 DFT |
| $W_N^k$ | `*tw1` | 第 1 个旋转因子 |
| $W_N^{2k}$ | `*tw2` | 第 2 个旋转因子 |
| $W_N^{3k}$ | `*tw3` | 第 3 个旋转因子 |
| $W_N^k \cdot G_1[k]$ | `scratch[0]` | 旋转后的 $G_1$ |
| $W_N^{2k} \cdot G_2[k]$ | `scratch[1]` | 旋转后的 $G_2$ |
| $W_N^{3k} \cdot G_3[k]$ | `scratch[2]` | 旋转后的 $G_3$ |

### 7.3 变量映射关系

```
代码中的数据布局（以 N=8, m=2 为例）：

Fout[0]   Fout[1]   Fout[2]   Fout[3]   Fout[4]   Fout[5]   Fout[6]   Fout[7]
  ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓
 G0[0]     G0[1]     G1[0]     G1[1]     G2[0]     G2[1]     G3[0]     G3[1]
  └────────┴────────┘         └────────┴────────┘         └────────┴────────┘
     第一组 (间距=m=2)            第二组                      第三组
```

---

## 8. 优化技巧总结

### 8.1 循环优化

**技巧 1：do-while 循环**

```c
do {
    // 循环体
} while (--m);
```

**优势**：
- 比 `for` 循环更快（某些编译器）
- 循环计数器递减（检查零比比较快）

**技巧 2：指针运算**

```c
++Fout;
++Fout2;
tw1 += fstride;
```

**优势**：
- 比数组索引 `Fout[i]` 更快
- 编译器容易优化为寄存器操作

### 8.2 内存访问优化

**技巧 1：原位计算**

- 输出覆盖输入，节省内存
- 提高缓存命中率

**技巧 2：scratch 数组**

```c
kiss_fft_cpx scratch[6];  // 栈上分配（可能被优化到寄存器）
```

**优势**：
- 避免重复计算
- 编译器可能优化到寄存器

**技巧 3：旋转因子预取**

```c
kiss_fft_cpx * tw1 = st->twiddles;  // 初始化指针
tw1 += fstride;                      // 每次迭代前进
```

### 8.3 运算优化

**技巧 1：特殊角度处理**

```c
// 避免 W_N^k = 1, -1, j, -j 的完整乘法
if (st->inverse) {
    Fout[m].r = scratch[5].r - scratch[4].i;  // 只需交换和变号
    ...
}
```

**技巧 2：标量乘法代替复数乘法**

```c
C_MULBYSCALAR(scratch[0], epi3.i);  // 只乘一个标量
```

**技巧 3：中间结果重用**

```c
C_ADD(scratch[3], scratch[0], scratch[2]);  // 计算一次
C_SUB(Fout[m2], *Fout, scratch[3]);        // 重用
C_ADDTO(*Fout, scratch[3]);                // 重用
```

### 8.4 定点数特殊处理

**技巧 1：移位代替除法**

```c
#define HALF_OF(x) ((x)>>1)  // 右移代替除以 2
```

**技巧 2：提前定标**

```c
C_FIXDIV(*Fout, 2);  // 每级定标，防止溢出
```

**技巧 3：舍入而非截断**

```c
#define sround(x) (kiss_fft_scalar)(((x) + (1<<(FRACBITS-1))) >> FRACBITS)
```

---

## 参考资源

### 相关文档
- [FFT 数学公式完整推导](./fft-mathematical-derivation.md) - 蝶形公式的数学推导
- [蝶形运算详细说明](./butterfly-operations.md) - 蝶形图和旋转因子
- [KISS FFT 剖析](./kiss_fft_anatomy.md) - 整体架构

### 源代码
- `kiss_fft.c` - 核心实现
- `_kiss_fft_guts.h` - 宏定义和内部函数
- `kiss_fft.h` - 公共接口

---

**下一步**：结合这三个文档，你应该能够：
1. 理解 FFT 的数学原理
2. 绘制和解读蝶形图
3. 逐行分析 KISS FFT 源代码
4. 理解各种优化技巧的原理

祝你学习顺利！
