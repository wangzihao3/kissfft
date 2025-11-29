# kiss_fft.c - KISS FFT库C语言实现技术文档

## 概述

`kiss_fft.c` 是KISS FFT库的核心C语言实现文件，提供了高效的快速傅里叶变换算法。该实现采用混合基Cooley-Tukey算法，支持多种基数分解策略，并针对不同硬件平台进行了优化。

## 核心数据结构

### FFT状态结构 (`kiss_fft_cfg`)

```c
struct kiss_fft_state {
    int nfft;              // FFT长度
    int inverse;            // 变换方向标志
    kiss_fft_cpx *twiddles; // 预计算的旋转因子
    int factors[2*MAXFACTORS]; // 因子分解结果
};
```

### 复数类型 (`kiss_fft_cpx`)

```c
typedef struct {
    kiss_fft_scalar r;  // 实部
    kiss_fft_scalar i;  // 虚部
} kiss_fft_cpx;
```

## 因子分解算法 `kf_factor`

### 算法策略

```c
static void kf_factor(int n, int *facbuf)
{
    int p = 4;
    double floor_sqrt = floor(sqrt((double)n));

    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;      // 优先分解4
                case 2: p = 3; break;      // 然后分解2
                default: p += 2; break;   // 最后尝试奇数因子
            }
            if (p > floor_sqrt)
                p = n;                  // 无更多因子
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}
```

### 分解原理

该算法采用以下优先级：
1. **基数4 (Radix-4)**：最有效率，减少25%的乘法运算
2. **基数2 (Radix-2)**：经典Cooley-Tukey算法
3. **奇数因子 (3,5,7,9...)**：处理非2的幂的情况

分解结果存储为：`[p1, m1, p2, m2, p3, m3, ...]`，其中 `pi × mi = n`。

## 核心工作函数 `kf_work`

### 算法架构

```c
static void kf_work(
    kiss_fft_cpx * Fout,
    const kiss_fft_cpx * f,
    const size_t fstride,
    int in_stride,
    int * factors,
    const kiss_fft_cfg st
)
```

### 递归分解过程

1. **基础情况处理** (`m == 1`)
   ```c
   if (m == 1) {
       do {
           *Fout = *f;
           f += fstride * in_stride;
       } while (++Fout != Fout_end);
   }
   ```

2. **递归分解** (`m > 1`)
   ```c
   do {
       // 递归调用：计算p个较小的DFT
       kf_work(Fout, f, fstride * p, in_stride, factors, st);
       f += fstride * in_stride;
   } while ((Fout += m) != Fout_end);
   ```

3. **蝶形运算重组**
   ```c
   switch (p) {
       case 2: kf_bfly2(Fout, fstride, st, m); break;
       case 3: kf_bfly3(Fout, fstride, st, m); break;
       case 4: kf_bfly4(Fout, fstride, st, m); break;
       case 5: kf_bfly5(Fout, fstride, st, m); break;
       default: kf_bfly_generic(Fout, fstride, st, m, p); break;
   }
   ```

## 优化蝶形运算实现

### Radix-2 蝶形运算

```c
static void kf_bfly2(kiss_fft_cpx * Fout, const size_t fstride,
                    const kiss_fft_cfg st, int m)
{
    kiss_fft_cpx * Fout2;
    kiss_fft_cpx * tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;

    do {
        C_FIXDIV(*Fout, 2); C_FIXDIV(*Fout2, 2);

        C_MUL(t, *Fout2, *tw1);
        tw1 += fstride;
        C_SUB(*Fout2, *Fout, t);
        C_ADDTO(*Fout, t);
        ++Fout2;
        ++Fout;
    } while (--m);
}
```

**数学原理**：
$$X[k] = X[k] + W_N^k \cdot X[k+m]$$
$$X[k+m] = X[k] - W_N^k \cdot X[k+m]$$

### Radix-4 蝶形运算

```c
static void kf_bfly4(kiss_fft_cpx * Fout, const size_t fstride,
                    const kiss_fft_cfg st, const size_t m)
{
    kiss_fft_cpx *tw1, *tw2, *tw3;
    kiss_fft_cpx scratch[6];
    size_t k = m;
    const size_t m2 = 2 * m;
    const size_t m3 = 3 * m;

    tw3 = tw2 = tw1 = st->twiddles;

    do {
        // 旋转因子乘法
        C_MUL(scratch[0], Fout[m], *tw1);
        C_MUL(scratch[1], Fout[m2], *tw2);
        C_MUL(scratch[2], Fout[m3], *tw3);

        // 蝶形运算
        C_SUB(scratch[5], *Fout, scratch[1]);
        C_ADDTO(*Fout, scratch[1]);
        C_ADD(scratch[3], scratch[0], scratch[2]);
        C_SUB(scratch[4], scratch[0], scratch[2]);

        // 特殊处理：实部/虚部交换（根据变换方向）
        if (st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        } else {
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }

        // 旋转因子指针更新
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        ++Fout;
    } while (--k);
}
```

**Radix-4 优化原理**：
- 减少25%的复数乘法
- 通过巧妙的加减法组合提高效率
- 特殊的实部虚部交换处理

### Radix-3 蝶形运算

```c
static void kf_bfly3(kiss_fft_cpx * Fout, const size_t fstride,
                    const kiss_fft_cfg st, size_t m)
{
    size_t k = m;
    const size_t m2 = 2 * m;
    kiss_fft_cpx *tw1, *tw2;
    kiss_fft_cpx scratch[5];
    kiss_fft_cpx epi3;
    epi3 = st->twiddles[fstride * m];

    tw1 = tw2 = st->twiddles;

    do {
        C_MUL(scratch[1], Fout[m], *tw1);
        C_MUL(scratch[2], Fout[m2], *tw2);

        C_ADD(scratch[3], scratch[1], scratch[2]);
        C_SUB(scratch[0], scratch[1], scratch[2]);
        tw1 += fstride;
        tw2 += fstride * 2;

        Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
        Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

        C_MULBYSCALAR(scratch[0], epi3.i);

        C_ADDTO(*Fout, scratch[3]);

        Fout[m2].r = Fout[m].r + scratch[0].i;
        Fout[m2].i = Fout[m].i - scratch[0].r;

        Fout[m].r -= scratch[0].i;
        Fout[m].i += scratch[0].r;

        ++Fout;
    } while (--k);
}
```

### Radix-5 蝶形运算

```c
static void kf_bfly5(kiss_fft_cpx * Fout, const size_t fstride,
                    const kiss_fft_cfg st, int m)
{
    kiss_fft_cpx *Fout0, *Fout1, *Fout2, *Fout3, *Fout4;
    int u;
    kiss_fft_cpx scratch[13];
    kiss_fft_cpx *twiddles = st->twiddles;
    kiss_fft_cpx *tw;
    kiss_fft_cpx ya, yb;
    ya = twiddles[fstride * m];
    yb = twiddles[fstride * 2 * m];

    Fout0 = Fout;
    Fout1 = Fout0 + m;
    Fout2 = Fout0 + 2 * m;
    Fout3 = Fout0 + 3 * m;
    Fout4 = Fout0 + 4 * m;

    tw = st->twiddles;

    for (u = 0; u < m; ++u) {
        // 旋转因子乘法
        C_MUL(scratch[1], *Fout1, tw[u * fstride]);
        C_MUL(scratch[2], *Fout2, tw[2 * u * fstride]);
        C_MUL(scratch[3], *Fout3, tw[3 * u * fstride]);
        C_MUL(scratch[4], *Fout4, tw[4 * u * fstride]);

        // 加减法组合
        C_ADD(scratch[7], scratch[1], scratch[4]);
        C_SUB(scratch[10], scratch[1], scratch[4]);
        C_ADD(scratch[8], scratch[2], scratch[3]);
        C_SUB(scratch[9], scratch[2], scratch[3]);

        // 复杂的乘法组合
        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r, ya.r) +
                    S_MUL(scratch[8].r, yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i, ya.r) +
                    S_MUL(scratch[8].i, yb.r);

        scratch[6].r = S_MUL(scratch[10].i, ya.i) + S_MUL(scratch[9].i, yb.i);
        scratch[6].i = -S_MUL(scratch[10].r, ya.i) - S_MUL(scratch[9].r, yb.i);

        C_SUB(*Fout1, scratch[5], scratch[6]);
        C_ADD(*Fout4, scratch[5], scratch[6]);

        // 更多乘法组合...
        ++Fout0; ++Fout1; ++Fout2; ++Fout3; ++Fout4;
    }
}
```

### 通用蝶形运算

```c
static void kf_bfly_generic(kiss_fft_cpx * Fout, const size_t fstride,
                          const kiss_fft_cfg st, int m, int p)
{
    int u, k, q1, q;
    kiss_fft_cpx *twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx *scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(
                                    sizeof(kiss_fft_cpx) * p);
    if (scratch == NULL) {
        KISS_FFT_ERROR("Memory allocation failed.");
        return;
    }

    for (u = 0; u < m; ++u) {
        k = u;
        for (q1 = 0; q1 < p; ++q1) {
            scratch[q1] = Fout[k];
            C_FIXDIV(scratch[q1], p);
            k += m;
        }

        k = u;
        for (q1 = 0; q1 < p; ++q1) {
            int twidx = 0;
            Fout[k] = scratch[0];
            for (q = 1; q < p; ++q) {
                twidx += fstride * k;
                if (twidx >= Norig) twidx -= Norig;
                C_MUL(t, scratch[q], twiddles[twidx]);
                C_ADDTO(Fout[k], t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}
```

## 并行化支持

### OpenMP优化

```c
#ifdef _OPENMP
// 使用OpenMP扩展在顶层进行并行化
if (fstride == 1 && p <= 5 && m != 1) {
    int k;

    // 在不同线程中执行p个工作单元
#   pragma omp parallel for
    for (k = 0; k < p; ++k)
        kf_work(Fout + k * m, f + fstride * in_stride * k,
                 fstride * p, in_stride, factors, st);

    // 所有线程在此点汇合

    switch (p) {
        case 2: kf_bfly2(Fout, fstride, st, m); break;
        case 3: kf_bfly3(Fout, fstride, st, m); break;
        case 4: kf_bfly4(Fout, fstride, st, m); break;
        case 5: kf_bfly5(Fout, fstride, st, m); break;
        default: kf_bfly_generic(Fout, fstride, st, m, p); break;
    }
    return;
}
#endif
```

## 内存管理和接口

### FFT配置分配

```c
kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft,
                        void * mem, size_t * lenmem)
{
    KISS_FFT_ALIGN_CHECK(mem);

    kiss_fft_cfg st = NULL;
    size_t memneeded = KISS_FFT_ALIGN_SIZE_UP(
                          sizeof(struct kiss_fft_state)
                          + sizeof(kiss_fft_cpx) * (nfft - 1));

    if (lenmem == NULL) {
        st = (kiss_fft_cfg)KISS_FFT_MALLOC(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }

    if (st) {
        int i;
        st->nfft = nfft;
        st->inverse = inverse_fft;

        // 预计算旋转因子
        for (i = 0; i < nfft; ++i) {
            const double pi = 3.141592653589793238462643383279502884197169399375105820974944592;
            double phase = -2 * pi * i / nfft;
            if (st->inverse)
                phase *= -1;
            kf_cexp(st->twiddles + i, phase);
        }

        kf_factor(nfft, st->factors);
    }
    return st;
}
```

### 主接口函数

```c
void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx *fin,
               kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg, fin, fout, 1);
}

void kiss_fft_stride(kiss_fft_cfg st, const kiss_fft_cpx *fin,
                  kiss_fft_cpx *fout, int in_stride)
{
    if (fin == fout) {
        // 注意：这并不是真正的就地FFT算法
        // 它只是将结果输出到临时缓冲区中
        if (fout == NULL) {
            KISS_FFT_ERROR("fout buffer NULL.");
            return;
        }

        kiss_fft_cpx *tmpbuf = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(
                                        sizeof(kiss_fft_cpx) * st->nfft);
        if (tmpbuf == NULL) {
            KISS_FFT_ERROR("Memory allocation error.");
            return;
        }

        kf_work(tmpbuf, fin, 1, in_stride, st->factors, st);
        memcpy(fout, tmpbuf, sizeof(kiss_fft_cpx) * st->nfft);
        KISS_FFT_TMP_FREE(tmpbuf);
    } else {
        kf_work(fout, fin, 1, in_stride, st->factors, st);
    }
}
```

## 辅助功能

### 快速大小查找

```c
int kiss_fft_next_fast_size(int n)
{
    while (1) {
        int m = n;
        while ((m % 2) == 0) m /= 2;      // 去除所有2的因子
        while ((m % 3) == 0) m /= 3;      // 去除所有3的因子
        while ((m % 5) == 0) m /= 5;      // 去除所有5的因子
        if (m <= 1)
            break;                        // n完全可分解为2、3、5的因子
        n++;
    }
    return n;
}
```

该函数查找下一个高效的FFT长度，推荐使用2、3、5的组合因子，以获得最佳性能。

## 算法特点总结

### 1. **混合基优化**
- 优先使用radix-4减少乘法运算
- 支持radix-2、radix-3、radix-5等特殊优化
- 通用算法处理任意基数

### 2. **内存效率**
- 原地变换减少内存使用
- 动态内存分配策略
- 缓冲区复用优化

### 3. **性能优化**
- 旋转因子预计算
- 专门的蝶形运算实现
- OpenMP并行化支持

### 4. **数值稳定性**
- 缩放因子处理（`C_FIXDIV`）
- 精度控制机制
- 跨平台兼容性

### 5. **接口灵活性**
- 支持步长访问（stride）
- 多种内存分配方式
- 正向和逆向变换支持

该实现代表了高效的C语言FFT库设计，在保持代码简洁的同时提供了优秀的性能和数值稳定性。