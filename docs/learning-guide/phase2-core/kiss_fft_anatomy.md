# KISS FFT 源代码解剖

## 概述

本文档将逐行分析 KISS FFT 的核心源代码，深入理解每个函数的实现细节和设计思想。

## 前置知识

在阅读本文档之前，你应该：
- ✅ 已完成阶段 1 的理论学习
- ✅ 理解 FFT 算法的基本原理
- ✅ 掌握 C 语言的基本语法

## 文件组织

```
kiss_fft.h          ← 公共 API 接口（约 100 行）
_kiss_fft_guts.h    ← 内部宏定义（约 150 行）
kiss_fft.c          ← 核心实现（约 500 行）
```

---

## 第一部分：kiss_fft.h - 公共接口

### 1.1 版本和宏定义

```c
#define KISS_FFT_VERSION "1.3.0"
```

**分析：**
- 版本号用于 API 兼容性检查
- 用户可以在运行时检查版本

### 1.2 条件编译：定点数 vs 浮点数

```c
#ifdef FIXED_POINT
# ifdef FIXED_POINT32
    typedef int32_t kiss_fft_scalar;
# else
    typedef int16_t kiss_fft_scalar;
# endif
#else
    typedef float kiss_fft_scalar;  // 默认浮点
#endif
```

**设计思想：**
- 使用条件编译支持多种数据类型
- 定点数（16/32 位）适合嵌入式系统
- 浮点数（32 位）适合通用应用

### 1.3 复数结构体

```c
typedef struct {
    kiss_fft_scalar r;  // 实部
    kiss_fft_scalar i;  // 虚部
} kiss_fft_cpx;
```

**为什么不用 C99 的 complex 类型？**

**答案：**
1. **可移植性**：C99 complex 支持不广泛
2. **控制性**：手动控制运算更灵活
3. **兼容性**：C++ 编译器可能有问题

### 1.4 前向声明

```c
typedef struct kiss_fft_state *kiss_fft_cfg;
```

**不透明指针模式（Opaque Pointer）：**

```c
// 用户只看到指针，看不到内部结构
kiss_fft_cfg cfg = kiss_fft_alloc(1024, 0, NULL, NULL);
//              ^^^^^^^^^^^^ 返回不透明指针

// 用户只能通过 API 访问，不能直接访问成员
kiss_fft(cfg, in, out);
kiss_fft_free(cfg);
```

**优点：**
- 封装性好（信息隐藏）
- 可以改变内部结构而不影响用户代码
- API 清晰明确

### 1.5 API 函数声明

#### 1.5.1 分配配置

```c
kiss_fft_cfg kiss_fft_alloc(
    int nfft,           // FFT 长度
    int inverse_fft,    // 0=正变换, 1=逆变换
    void *mem,          // 可选：用户提供的内存
    size_t *lenmem      // 可选：内存大小
);
```

**灵活性设计：**

**模式 1：动态分配**
```c
kiss_fft_cfg cfg = kiss_fft_alloc(1024, 0, NULL, NULL);
//                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                   内部调用 malloc
```

**模式 2：静态分配**
```c
char buffer[4096];
size_t len = sizeof(buffer);
kiss_fft_cfg cfg = kiss_fft_alloc(1024, 0, buffer, &len);
//                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                   使用用户提供的内存
```

**优势：**
- 嵌入式系统可能没有 malloc
- 静态分配避免碎片
- 给用户更多控制权

#### 1.5.2 执行 FFT

```c
void kiss_fft(
    const kiss_fft_cfg cfg,    // 配置（不变）
    const kiss_fft_cpx *fin,   // 输入（不变）
    kiss_fft_cpx *fout         // 输出
);
```

**const 修饰符的作用：**
```c
const kiss_fft_cfg cfg    // 配置不会被修改
const kiss_fft_cpx *fin   // 输入不会被修改
```

**编译器优化：**
- const 告诉编译器数据不变
- 可以进行更好的优化
- 防止意外修改

#### 1.5.3 释放配置

```c
void kiss_fft_free(kiss_fft_cfg cfg);
```

**对应两种分配方式：**

```c
// 动态分配时释放内部内存
if (使用了 malloc) {
    free(cfg->twiddles);
    free(cfg);
}

// 静态分配时不做任何事
if (用户提供了内存) {
    // 什么都不做，由用户管理
}
```

---

## 第二部分：_kiss_fft_guts.h - 内部定义

### 2.1 复数运算宏

```c
#define C_ADD(c, a, b) \
    ((c).r = (a).r + (b).r, \
     (c).i = (a).i + (b).i)

#define C_SUB(c, a, b) \
    ((c).r = (a).r - (b).r, \
     (c).i = (a).i - (b).i)

#define C_MUL(c, a, b) \
    ((c).r = (a).r * (b).r - (a).i * (b).i, \
     (c).i = (a).r * (b).i + (a).i * (b).r)
```

**为什么用宏而不是函数？**

**答案：**
1. **性能**：避免函数调用开销
2. **内联**：编译器可以直接展开
3. **类型无关**：支持不同 scalar 类型

**复数乘法的推导：**
```
(a + jb) × (c + jd)
= ac + jad + jbc - bd
= (ac - bd) + j(ad + bc)
```

### 2.2 定点数定标宏

```c
#ifdef FIXED_POINT
# if (FIXED_POINT == 32)
#  define SAMP_MAX 2147483647
#  define SAMP_MIN -2147483648
# else
#  define SAMP_MAX 32767
#  define SAMP_MIN -32768
# endif

# define C_FIXDIV(c, div) \
    do { \
        (c).r = (kiss_fft_scalar)(((int64_t)(c).r) >> (div)); \
        (c).i = (kiss_fft_scalar)(((int64_t)(c).i) >> (div)); \
    } while (0)
#else
# define C_FIXDIV(c, div) /* 浮点数不需要定标 */
#endif
```

**定点数为什么需要定标？**

**问题：** 每级蝶形运算后幅度会增长

```
假设输入幅度 = 1
第 1 级后：最大幅度 = 2
第 2 级后：最大幅度 = 4
...
第 k 级后：最大幅度 = 2^k
```

对于 16 位定点数：
- 最大值 = 32767
- log₂(32767) ≈ 15
- 最多支持 15 级（N = 2^15 = 32768）

**解决方案：右移（除以 2）**
```c
// 每级蝶形运算后右移 1 位
C_FIXDIV(result, 1);  // result >>= 1
```

**do-while(0) 技巧：**
```c
#define C_FIXDIV(c, div) \
    do { \
        (c).r = ...; \
        (c).i = ...; \
    } while (0)
```

**为什么需要？**

**错误示例：**
```c
#define BAD_MACRO(x) \
    a = x; \
    b = x;

// 使用 if 语句
if (condition)
    BAD_MACRO(5);  // ❌ 只有 a = 5 被执行
// 正确意图：
if (condition) {
    a = 5;
    b = 5;
}
```

**正确示例：**
```c
#define GOOD_MACRO(x) \
    do { \
        a = x; \
        b = x; \
    } while (0)

// 使用 if 语句
if (condition)
    GOOD_MACRO(5);  // ✓ 两行都被执行
```

### 2.3 内部结构体定义

```c
struct kiss_fft_state {
    int nfft;                  // FFT 长度
    int inverse;               // 逆变换标志
    kiss_fft_cpx *factors;     // 旋转因子表
    kiss_fft_cpx *tmpbuf;      // 临时缓冲区（可选）
    int *factors;              // 分解因子
    int n_factors;             // 因子数量
};
```

**内存布局：**
```
kiss_fft_state (在堆上)
├── nfft: 4 bytes
├── inverse: 4 bytes
├── twiddles: pointer → 旋转因子数组 (N-1) * sizeof(kiss_fft_cpx)
├── tmpbuf: pointer → 临时缓冲区 N * sizeof(kiss_fft_cpx) (可选)
├── factors: pointer → 分解因子数组
└── n_factors: 4 bytes

总内存（不含 tmpbuf）：
  = sizeof(state) + (N-1)*sizeof(cpx) + n_factors*sizeof(int)

对于 N=1024, float:
  ≈ 24 + 1023*8 + 10*4
  ≈ 8.4 KB
```

---

## 第三部分：kiss_fft.c - 核心实现

### 3.1 因子分解

```c
static void kf_factor(int n, int *factors, int *nfactors)
{
    *nfactors = 0;
    int n_orig = n;
    int p = 4;
    int floor_sqrt = (int)sqrt((double)n);

    // 尝试分解 n = p1 × p2 × ... × pk
    while (n > 1) {
        while (n % p != 0) {
            // 找下一个因子
            if (p == 4) p = 2;
            else if (p == 2) p = 3;
            else if (p == 3) p = 5;
            else p += 2;

            if (p > floor_sqrt)
                p = n;  // n 本身是素数
        }

        n /= p;
        factors[(*nfactors)++] = p;
    }

    // 优化：将 4,2 合并为 8
    if (*nfactors > 1 &&
        factors[*nfactors-1] == 2 &&
        factors[*nfactors-2] == 4) {
        factors[*nfactors-2] = 8;
        (*nfactors)--;
    }
}
```

**示例：N = 60**

```
初始: n = 60, p = 4

步骤 1: 60 % 4 == 0 ✓
  factors[0] = 4
  n = 60 / 4 = 15
  p 重置为 4

步骤 2: 15 % 4 != 0, p = 2
       15 % 2 != 0, p = 3
       15 % 3 == 0 ✓
  factors[1] = 3
  n = 15 / 3 = 5
  p 重置为 4

步骤 3: 5 % 4 != 0, p = 2
       5 % 2 != 0, p = 3
       5 % 3 != 0, p = 5
       5 % 5 == 0 ✓
  factors[2] = 5
  n = 5 / 5 = 1

结果: factors = [4, 3, 5]
验证: 60 = 4 × 3 × 5 ✓
```

**为什么优先分解 4？**

**答案：基-4 FFT 更快**
- 基-4 比两次基-2 快（减少复数乘法）
- 优先分解 4 可以利用基-4 优化

**为什么将 4,2 合并为 8？**

**答案：基-8 FFT 更优**
```
N = 4 × 2 × M = 8 × M

方案 1: 分解为 4, 2
  → 第 1 级：基-4 蝶形
  → 第 2 级：基-2 蝶形

方案 2: 合并为 8
  → 第 1 级：基-8 蝶形
  → 更少的旋转因子乘法
```

### 3.2 旋转因子生成

```c
static void kf_twiddles(kiss_fft_cpx *twiddles, int nfft, int inverse)
{
    int i;
    double phase;

    for (i = 0; i < nfft; ++i) {
        phase = -2 * M_PI * i / nfft;
        if (inverse)
            phase *= -1;  // 逆变换改变符号

        twiddles[i].r = (kiss_fft_scalar) cos(phase);
        twiddles[i].i = (kiss_fft_scalar) sin(phase);
    }
}
```

**旋转因子的数学定义：**

正变换：
```
W_N^k = e^(-j2πk/N) = cos(2πk/N) - j·sin(2πk/N)
```

逆变换：
```
W_N^(-k) = e^(j2πk/N) = cos(2πk/N) + j·sin(2πk/N)
```

**示例：N = 8**

| k | phase (rad) | cos(phase) | sin(phase) | W_8^k |
|---|-------------|-------------|-------------|-------|
| 0 | 0 | 1 | 0 | 1 + j0 |
| 1 | -π/4 | 0.707 | -0.707 | 0.707 - j0.707 |
| 2 | -π/2 | 0 | -1 | 0 - j1 |
| 3 | -3π/4 | -0.707 | -0.707 | -0.707 - j0.707 |
| 4 | -π | -1 | 0 | -1 + j0 |
| 5 | -5π/4 | -0.707 | 0.707 | -0.707 + j0.707 |
| 6 | -3π/2 | 0 | 1 | 0 + j1 |
| 7 | -7π/4 | 0.707 | 0.707 | 0.707 + j0.707 |

**周期性验证：**
```
W_8^9 = W_8^(8+1) = W_8^8 × W_8^1 = 1 × W_8^1 = W_8^1
```

### 3.3 核心工作函数

```c
static void kf_work(
    const kiss_fft_cfg cfg,
    kiss_fft_cpx *Fout,
    const kiss_fft_cpx *f,
    const size_t fstride,
    const int in_stride,
    const int *factors
)
{
    // 递归实现的核心逻辑
    // f: 当前处理的数据
    // fstride: 旋转因子的步长
    // in_stride: 输入数据的步长
    // factors: 因子数组
}
```

**参数详解：**

**fstride（频率步长）：**
```
递归层次   fstride          含义
第 0 级    1                使用所有旋转因子
第 1 级    p0               每 p0 个取一个
第 2 级    p0×p1            每 p0×p1 个取一个
...
```

**示例：N = 12 = 3 × 4**

```
factors = [3, 4]

第 0 级（基-3）：
  fstride = 12 / 3 = 4
  使用 W_12^0, W_12^4, W_12^8

第 1 级（基-4）：
  fstride = 12 / 12 = 1
  使用 W_4^0, W_4^1, W_4^2, W_4^3
```

### 3.4 基-2 蝶形运算

```c
static void kf_bfly2(
    kiss_fft_cpx *Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    int m
)
{
    kiss_fft_cpx *Fout2;
    kiss_fft_cpx *tw1 = st->twiddles;
    kiss_fft_cpx t;

    Fout2 = Fout + m;

    do {
        // 定标（防止溢出）
        C_FIXDIV(*Fout, 1);
        C_FIXDIV(*Fout2, 1);

        // 蝶形运算：
        // Fout' = Fout + W·Fout2
        // Fout2' = Fout - W·Fout2
        C_MUL(t, *Fout2, *tw1);
        C_SUB(*Fout2, *Fout, t);
        C_ADDTO(*Fout, t);

        ++tw1;
        ++Fout2;
        ++Fout;
    } while (--m);
}
```

**蝶形运算图解：**

```
     Fout ───────────┬─────────→ Fout' = Fout + W·Fout2
                      │
                      ├─ W (旋转因子)
                      │
     Fout2 ──────────┴─────────→ Fout2' = Fout - W·Fout2
```

**代码对应：**
```c
C_MUL(t, *Fout2, *tw1);   // t = W·Fout2
C_SUB(*Fout2, *Fout, t);  // Fout2' = Fout - t
C_ADDTO(*Fout, t);        // Fout' = Fout + t
```

**为什么用 do-while 而不是 for？**

**答案：处理 m = 0 的情况**
```c
// do-while: 至少执行一次
do {
    // ...
} while (--m);

// for: 如果 m = 0，不执行
for (int i = 0; i < m; i++) {
    // ...
}
```

但在 FFT 中，m >= 1，两者等效。do-while 是风格选择。

### 3.5 基-4 蝶形运算

```c
static void kf_bfly4(
    kiss_fft_cpx *Fout,
    const size_t fstride,
    const kiss_fft_cfg st,
    int m
)
{
    kiss_fft_cpx *tw1, *tw2, *tw3;
    kiss_fft_cpx scratch[6];
    kiss_fft_cpx *Fout1, *Fout2, *Fout3;

    // 四个输出点
    Fout1 = Fout + m;
    Fout2 = Fout + 2*m;
    Fout3 = Fout + 3*m;

    // 基-4 蝶形运算
    // 展开为多个基-2 蝶形
    // ...
}
```

**基-4 蝶形公式：**
```
X[k]     = A + W·B + W²·C + W³·D
X[k+m]   = A - j·W·B - W²·C + j·W³·D
X[k+2m]  = A - W·B + W²·C - W³·D
X[k+3m]  = A + j·W·B - W²·C - j·W³·D

其中：
W = W_N^k
```

**优化技巧：**
- 减少 3 次复数乘法（4 个输出，共需 12 次，优化后只需 9 次）
- 利用 j = √(-1) 的性质简化运算

---

## 总结

KISS FFT 的核心实现展示了：

1. **清晰的结构**：API → 内部定义 → 核心实现
2. **灵活的设计**：支持多种数据类型和分配方式
3. **高效的算法**：混合基数 FFT，优化蝶形运算
4. **简洁的代码**：约 500 行实现完整功能

**关键收获：**
- 不透明指针模式实现封装
- 宏实现高性能复数运算
- 递归实现清晰的算法结构
- 预计算优化运行时性能

**下一步：** 阅读实际源代码，在 IDE 中跟踪执行流程。
