# 阶段 2：核心代码探索

欢迎进入 KISS FFT 的核心世界！在这个阶段，我们将深入分析 KISS FFT 的源代码，理解其实现细节和设计思想。

## 学习目标

完成本阶段后，您将能够：
- [ ] 理解 KISS FFT 的项目结构和模块划分
- [ ] 掌握核心数据结构的设计和使用
- [ ] 跟踪 FFT 算法的实现流程
- [ ] 理解不同数据类型的处理机制
- [ ] 掌握内存管理和性能优化策略

## 项目概览

### 目录结构

```
kissfft/
├── kiss_fft.h           # 核心 API 定义
├── kiss_fft.c           # 核心 FFT 实现
├── _kiss_fft_guts.h     # 内部定义和宏
├── kiss_fftnd.h/c       # 多维 FFT
├── kiss_fftndr.h/c      # 多维实数 FFT
├── kiss_fftr.h/c        # 实数 FFT
├── kfc.h/c              # FFT 缓存工具
├── tools/               # 命令行工具
├── test/                # 测试代码
└── cmake/               # CMake 配置
```

### 核心模块

1. **kiss_fft.h/c** - 核心 1D 复数 FFT
2. **kiss_fftr.h/c** - 实数 FFT 优化
3. **kiss_fftnd.h/c** - 多维 FFT
4. **_kiss_fft_guts.h** - 内部实现细节

## 核心数据结构

### 1. 复数表示

```c
// kiss_fft.h
typedef struct {
    kiss_fft_scalar r;   // 实部
    kiss_fft_scalar i;   // 虚部
} kiss_fft_cpx;
```

**关键点：**
- `kiss_fft_scalar` 根据编译选项定义为 float 或 double
- 简单的结构体设计，易于理解和优化
- 支持不同的数值精度

### 2. FFT 配置结构

```c
// _kiss_fft_guts.h
struct kiss_fft_state{
    int nfft;                 // FFT 长度
    int inverse;              // 是否为逆变换
    kiss_fft_cpx * factors;   // 旋转因子表
    kiss_fft_cpx *tmpbuf;     // 临时缓冲区
    struct kiss_fft_state *substates; // 子 FFT 状态
    int *factors;             // 分解因子
    int n_factors;            // 因子数量
};

typedef struct kiss_fft_state *kiss_fft_cfg;
```

**设计分析：**
- `nfft`：当前 FFT 长度
- `factors`：预计算的旋转因子，避免重复计算
- `substates`：递归分解的子问题
- `factors` 数组：存储分解的因子（如 [2, 2, 2] 表示 8 = 2×2×2）

### 3. 旋转因子

旋转因子是 FFT 的核心，定义为：
```
W_N^k = e^(-j2πk/N)
```

在 KISS FFT 中，所有旋转因子都预计算并存储在 `factors` 数组中。

## 算法实现分析

### 1. 主函数：kiss_fft

```c
void kiss_fft(const kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
    if (cfg->tmpbuf == NULL) {
        // 使用就地计算
        kf_work(cfg, fout, fin, 1, 1, cfg->factors, cfg->tmpbuf);
    } else {
        // 使用临时缓冲区
        kf_work(cfg, fout, fin, 1, 1, cfg->factors, cfg->tmpbuf);
    }
}
```

### 2. 核心工作函数：kf_work

```c
static void kf_work(const kiss_fft_cfg cfg,
                   kiss_fft_cpx *Fout,
                   const kiss_fft_cpx *f,
                   const size_t fstride,
                   const int in_stride,
                   const int *factors,
                   const kiss_fft_cpx *twiddles)
{
    // 递归实现的核心逻辑
}
```

**参数说明：**
- `fstride`：频率步长
- `in_stride`：输入步长
- `factors`：分解因子
- `twiddles`：旋转因子表

### 3. 蝶形运算

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
        C_FIXDIV(*Fout, 2);
        C_FIXDIV(*Fout2, 2);

        C_MUL(t, *Fout2, *tw1);
        C_SUB(*Fout2, *Fout, t);
        C_ADDTO(*Fout, t);
        ++tw1;
        ++Fout2;
        ++Fout;
    } while (--m);
}
```

**宏定义简化了复数运算：**
```c
#define C_ADD(c,a,b) ( (c).r = (a).r + (b).r, (c).i = (a).i + (b).i )
#define C_SUB(c,a,b) ( (c).r = (a).r - (b).r, (c).i = (a).i - (b).i )
#define C_MUL(c,a,b) ( (c).r = (a).r*(b).r - (a).i*(b).i, \
                        (c).i = (a).r*(b).i + (a).i*(b).r )
```

## 内存管理策略

### 1. 配置分配

```c
kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void *mem, size_t *lenmem)
{
    kiss_fft_cfg cfg = NULL;
    size_t memneeded = sizeof(struct kiss_fft_state) +
                      sizeof(kiss_fft_cpx) * (nfft - 1); // twiddle factors

    if (lenmem == NULL) {
        cfg = (kiss_fft_cfg)KISS_FFT_MALLOC(memneeded);
    } else {
        if (*lenmem >= memneeded)
            cfg = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }

    if (cfg) {
        // 初始化配置
        cfg->nfft = nfft;
        cfg->inverse = inverse_fft;
        // ...
    }

    return cfg;
}
```

**特点：**
- 支持静态分配（传入内存指针）
- 支持动态分配（使用 malloc）
- 内存大小精确计算

### 2. 临时缓冲区

```c
// 可以使用 alloca 分配栈内存
#ifdef KISS_FFT_USE_ALLOCA
    void *tmpbuf = alloca(sizeof(kiss_fft_cpx) * nfft);
#else
    void *tmpbuf = KISS_FFT_MALLOC(sizeof(kiss_fft_cpx) * nfft);
#endif
```

## 数据类型处理

### 1. 定点数支持

当编译时定义 `FIXED_POINT` 时，启用定点数运算：

```c
#ifdef FIXED_POINT
# if (FIXED_POINT==32)
    typedef int32_t kiss_fft_scalar;
# else
    typedef int16_t kiss_fft_scalar;
# endif
#else
    typedef float kiss_fft_scalar;
#endif
```

### 2. 定点数运算

```c
#ifdef FIXED_POINT
#define SAMP_MAX 32767
#define SAMP_MIN -32768

#define C_FIXDIV(c,div) \
    do { (c).r = (kiss_fft_scalar)(((int64_t)(c).r) >> (div)); \
         (c).i = (kiss_fft_scalar)(((int64_t)(c).i) >> (div)); } while (0)
#else
#define C_FIXDIV(c,div) /* No scaling for floating point */
#endif
```

## 性能优化技巧

### 1. 编译器优化

```cmake
# CMakeLists.txt 中的优化选项
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-ffast-math -fomit-frame-pointer)
endif()
```

### 2. 循环展开

```c
// 手工展开关键循环
case 4:
    do {
        C_MUL(t, *Fout2, *tw1);
        tw1 += fstride*4;
        C_SUB(*Fout2, *Fout, t);
        C_ADDTO(*Fout, t);
        ++Fout2; ++Fout;
    } while (--m);
    break;
```

### 3. 内存对齐

```c
// 确保数据对齐以提高缓存效率
#ifdef USE_SIMD
    __attribute__((aligned(16))) kiss_fft_cpx buffer[1024];
#endif
```

## 本周学习任务

### 第 3 周：项目结构和数据结构

**周一**
- [ ] 阅读项目文档和 README
- [ ] 理解目录结构和模块划分
- [ ] 编译并运行基本测试

**周二**
- [ ] 分析 `kiss_fft.h` 中的数据类型定义
- [ ] 理解 `kiss_fft_cpx` 结构体设计
- [ ] 实验不同的数值精度设置

**周三**
- [ ] 深入分析 `kiss_fft_cfg` 结构
- [ ] 理解配置参数的含义
- [ ] 跟踪 `kiss_fft_alloc` 的执行流程

**周四**
- [ ] 学习旋转因子的预计算
- [ ] 理解内存分配策略
- [ ] 分析子 FFT 的递归结构

**周五**
- [ ] 总结数据结构的设计思想
- [ ] 完成代码分析练习

### 第 4 周：算法实现

**周一/周二**
- [ ] 逐行分析 `kiss_fft.c` 的主函数
- [ ] 理解参数传递和调用约定
- [ ] 跟踪简单 FFT 的执行流程

**周三/周四**
- [ ] 深入分析 `kf_work` 函数
- [ ] 理解递归分解的逻辑
- [ ] 绘制函数调用树

**周五**
- [ ] 分析蝶形运算的实现
- [ ] 理解优化技巧
- [ ] 性能测试和对比

### 第 5 周：高级主题

**周一**
- [ ] 学习定点数 FFT 的实现
- [ ] 理解溢出处理机制
- [ ] 对比浮点和定点的性能

**周二**
- [ ] 探索内存优化技巧
- [ ] 理解缓存友好设计
- [ ] 实验不同的编译选项

**周三/周四**
- [ ] 添加调试输出，跟踪执行
- [ ] 修改代码观察行为变化
- [ ] 性能分析和优化

**周五**
- [ ] 代码审查和重构
- [ ] 总结实现经验
- [ ] 准备进入下一阶段

## 实践练习

### 练习 1：代码追踪

编写一个调试版本，添加详细的执行日志：

```c
// 添加调试宏
#define DEBUG_PRINT(fmt, ...) \
    printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

// 在关键位置添加日志
void kiss_fft_debug(const kiss_fft_cfg cfg, const kiss_fft_cpx *fin, kiss_fft_cpx *fout)
{
    DEBUG_PRINT("Starting FFT: N=%d, inverse=%d", cfg->nfft, cfg->inverse);

    // 打印输入数据
    DEBUG_PRINT("Input data:");
    for (int i = 0; i < cfg->nfft && i < 8; i++) {
        DEBUG_PRINT("  [%d]: %f + %fi", i, fin[i].r, fin[i].i);
    }

    // 调用原始函数
    kiss_fft(cfg, fin, fout);

    // 打印输出数据
    DEBUG_PRINT("Output data:");
    for (int i = 0; i < cfg->nfft && i < 8; i++) {
        DEBUG_PRINT("  [%d]: %f + %fi", i, fout[i].r, fout[i].i);
    }
}
```

### 练习 2：性能测试

```c
#include <time.h>

void benchmark_fft(int N, int iterations) {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    // 计时测试
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        kiss_fft(cfg, input, output);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("N=%d, iterations=%d, time=%.3f ms\n",
           N, iterations, elapsed * 1000 / iterations);

    free(input);
    free(output);
    kiss_fft_free(cfg);
}
```

### 练习 3：代码修改实验

1. **移除优化**：手工实现朴素的 FFT，观察性能差异
2. **改变因子分解**：尝试不同的分解策略
3. **内存布局实验**：改变数据结构的内存排列

## 深入分析主题

### 1. 混合基数 FFT

KISS FFT 支持任意长度的 FFT，不仅仅是 2 的幂：

```c
static void kf_factor(int n, int *factors, int *nfactors)
{
    // 分解 n 为小因子的乘积
    int n_orig = n;
    int p = 4;
    int floor_sqrt = (int)sqrt(n);

    // 优先分解出 4 的因子
    while (n > 1) {
        while (n % p) {
            if (p == 4) p = 2;
            else if (p == 2) p = 3;
            else if (p == 3) p = 5;
            else p += 2;
            if (p > floor_sqrt)
                p = n;
        }

        n /= p;
        factors[(*nfactors)++] = p;
    }

    // 如果最后一个因子是 2，且前面有因子 4，则合并成 8
    if (nfactors > 1 && factors[nfactors-1] == 2 && factors[nfactors-2] == 4) {
        factors[nfactors-2] = 8;
        nfactors--;
    }
}
```

### 2. 特殊优化

基-4 FFT 的特殊优化：

```c
static void kf_bfly4(kiss_fft_cpx *Fout, const size_t fstride,
                     const kiss_fft_cfg st, int m)
{
    const kiss_fft_cpx *tw1, *tw2, *tw3;
    kiss_fft_cpx scratch[6];

    // 基-4 蝶形运算
    // 展开循环以减少分支
}
```

## 常见问题解析

### Q1: 为什么使用递归而不是迭代？

**A:** 递归实现更清晰，便于理解算法结构：
- 自然对应数学定义
- 易于处理混合基数
- 代码可读性好

### Q2: 如何处理就地计算和需要临时缓冲区的情况？

**A:** KISS FFT 提供了两种方式：
- 检查输入和输出是否重叠
- 自动选择最优的内存使用策略

### Q3: 定点数精度如何保证？

**A:** 通过定标和移位操作：
- 在每级蝶形运算后进行定标
- 防止溢出的同时保持精度

## 下一阶段预告

完成本阶段后，您将具备：
- 深入理解 KISS FFT 的实现原理
- 能够阅读和修改 FFT 代码
- 掌握性能分析和优化技巧

准备好进入[阶段 3：实践应用](../phase3-practical/)，将理论转化为实际应用！

---

记住：阅读源代码是学习的最佳方式，不要害怕修改和实验。