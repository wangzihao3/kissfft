# kissfft.hh - KISS FFT库核心实现技术文档

## 概述

`kissfft.hh` 是KISS FFT（Keep It Simple, Stupid FFT）库的核心C++头文件实现。该文件提供了高效的快速傅里叶变换算法实现，采用混合基（mixed-radix）Cooley-Tukey算法，支持多种数据类型和变换方向。

## 类结构设计

### 模板类定义

```cpp
template <typename scalar_t>
class kissfft
```

- **模板参数**：`scalar_t` - 支持的标量类型（float、double、long double）
- **核心类型**：`typedef std::complex<scalar_t> cpx_t` - 复数类型定义

### 构造函数

```cpp
kissfft( const std::size_t nfft, const bool inverse )
```

#### 构造过程分析

1. **旋转因子预计算**
   ```cpp
   _twiddles.resize(_nfft);
   const scalar_t phinc = (_inverse?2:-2) * std::acos((scalar_t)-1) / _nfft;
   for (std::size_t i=0;i<_nfft;++i)
       _twiddles[i] = std::exp( cpx_t(0,i*phinc) );
   ```

   - 根据变换方向确定相位增量：正向FFT使用-2，逆FFT使用+2
   - 预计算所有旋转因子：$W_N^k = e^{-i2\pi k/N}$（正向）或$e^{i2\pi k/N}$（逆向）

2. **混合基分解算法**
   ```cpp
   std::size_t n = _nfft;
   std::size_t p = 4;
   do {
       while (n % p) {
           switch (p) {
               case 4: p = 2; break;    // 优先分解4
               case 2: p = 3; break;    // 然后分解2
               default: p += 2; break;  // 最后尝试奇数因子
           }
           if (p*p > n) p = n;        // 无更多因子
       }
       n /= p;
       _stageRadix.push_back(p);
       _stageRemainder.push_back(n);
   } while(n>1);
   ```

   该算法采用以下分解策略：
   - 优先提取因子4（radix-4 FFT最有效率）
   - 然后提取因子2（radix-2）
   - 最后尝试奇数因子（3, 5, 7, 9...）

## 核心变换算法

### 主变换函数 `transform`

```cpp
void transform(const cpx_t * fft_in, cpx_t * fft_out,
             const std::size_t stage = 0,
             const std::size_t fstride = 1,
             const std::size_t in_stride = 1) const
```

#### 算法原理

该函数实现递归的混合基Cooley-Tukey算法：

1. **分解阶段**：将长度为 N = p×m 的DFT分解为p个长度为m的较小DFT
2. **递归计算**：计算每个较小DFT
3. **重组阶段**：通过蝶形运算重组结果

#### 实现细节

```cpp
if (m == 1) {
    // 基础情况：长度为1的DFT就是输入本身
    do {
        *fft_out = *fft_in;
        fft_in += fstride*in_stride;
    } while(++fft_out != Fout_end);
} else {
    // 递归调用：计算p个较小的DFT
    do {
        transform(fft_in, fft_out, stage+1, fstride*p, in_stride);
        fft_in += fstride*in_stride;
    } while ((fft_out += m) != Fout_end);
}
```

4. **蝶形运算重组**
   ```cpp
   switch (p) {
       case 2: kf_bfly2(fft_out,fstride,m); break;
       case 3: kf_bfly3(fft_out,fstride,m); break;
       case 4: kf_bfly4(fft_out,fstride,m); break;
       case 5: kf_bfly5(fft_out,fstride,m); break;
       default: kf_bfly_generic(fft_out,fstride,m,p); break;
   }
   ```

## 优化蝶形运算实现

### Radix-2 蝶形运算

```cpp
void kf_bfly2(cpx_t * Fout, const size_t fstride, const std::size_t m) const
{
    for (std::size_t k=0;k<m;++k) {
        const cpx_t t = Fout[m+k] * _twiddles[k*fstride];
        Fout[m+k] = Fout[k] - t;
        Fout[k] += t;
    }
}
```

实现标准的radix-2蝶形运算：
$$X[k] = X[k] + W_N^k \cdot X[k+m]$$
$$X[k+m] = X[k] - W_N^k \cdot X[k+m]$$

### Radix-4 蝶形运算

```cpp
void kf_bfly4(cpx_t * const Fout, const std::size_t fstride, const std::size_t m) const
{
    cpx_t scratch[7];
    const scalar_t negative_if_inverse = _inverse ? -1 : +1;

    for (std::size_t k=0;k<m;++k) {
        scratch[0] = Fout[k+  m] * _twiddles[k*fstride  ];
        scratch[1] = Fout[k+2*m] * _twiddles[k*fstride*2];
        scratch[2] = Fout[k+3*m] * _twiddles[k*fstride*3];
        scratch[5] = Fout[k] - scratch[1];

        Fout[k] += scratch[1];
        scratch[3] = scratch[0] + scratch[2];
        scratch[4] = scratch[0] - scratch[2];
        scratch[4] = cpx_t(scratch[4].imag()*negative_if_inverse,
                          -scratch[4].real()*negative_if_inverse);

        Fout[k+2*m]  = Fout[k] - scratch[3];
        Fout[k    ]+= scratch[3];
        Fout[k+  m] = scratch[5] + scratch[4];
        Fout[k+3*m] = scratch[5] - scratch[4];
    }
}
```

### 通用蝶形运算

```cpp
void kf_bfly_generic(cpx_t * const Fout, const std::size_t fstride,
                   const std::size_t m, const std::size_t p) const
```

处理任意基数的蝶形运算，使用动态分配的缓冲区进行计算。

## 实数FFT优化 `transform_real`

针对实数输入的特殊优化，利用傅里叶变换的对称性：

### 对称性利用

实数序列的DFT具有共轭对称性：
$$X[N-k] = \overline{X[k]}$$

### 优化策略

1. **先进行复数FFT**：将实数序列视为复数序列
2. **后处理优化**：利用对称性减少计算量
3. **特殊处理**：k=0和k=N的特殊情况

```cpp
// post processing for k = 0 and k = N
dst[0] = cpx_t(dst[0].real() + dst[0].imag(),
               dst[0].real() - dst[0].imag());

// post processing for all other k = 1, 2, ..., N-1
const scalar_t half_phi_inc = (_inverse ? pi : -pi) / N;
const cpx_t twiddle_mul = std::exp(cpx_t(0, half_phi_inc));

for (std::size_t k = 1; 2*k < N; ++k) {
    const cpx_t w = (scalar_t)0.5 * cpx_t(
                     dst[k].real() + dst[N-k].real(),
                     dst[k].imag() - dst[N-k].imag());
    const cpx_t z = (scalar_t)0.5 * cpx_t(
                     dst[k].imag() + dst[N-k].imag(),
                    -dst[k].real() + dst[N-k].real());
    // ... 复杂的旋转因子应用
}
```

## 动态重配置 `assign`

支持运行时改变FFT参数：

```cpp
void assign(const std::size_t nfft, const bool inverse)
{
    if (nfft != _nfft) {
        kissfft tmp(nfft, inverse);    // 创建新对象 O(n)
        std::swap(tmp, *this);        // 高效交换 O(1) 或 O(n)
    } else if (inverse != _inverse) {
        // 仅取共轭改变旋转因子
        for (typename std::vector<cpx_t>::iterator it = _twiddles.begin();
             it != _twiddles.end(); ++it)
            it->imag(-it->imag());
    }
}
```

## 数据成员

```cpp
std::size_t _nfft;                           // FFT长度
bool _inverse;                                 // 变换方向
std::vector<cpx_t> _twiddles;                 // 预计算的旋转因子
std::vector<std::size_t> _stageRadix;          // 各阶段的基数
std::vector<std::size_t> _stageRemainder;      // 各阶段的余数
mutable std::vector<cpx_t> _scratchbuf;         // 临时缓冲区
```

## 算法复杂度分析

### 时间复杂度
- **最优情况**：$O(N \log N)$（当N的因子以2和4为主）
- **最坏情况**：$O(N \log N)$（仍然保持对数复杂度）
- **实数FFT**：约减少50%的计算量

### 空间复杂度
- **存储需求**：$O(N)$（旋转因子 + 临时缓冲区）
- **递归栈**：$O(\log N)$（深度为因子分解层数）

## 设计特点

1. **模板化设计**：支持多种数值精度
2. **混合基优化**：优先使用radix-4和radix-2提高效率
3. **预计算优化**：旋转因子预计算避免重复计算
4. **内存效率**：原地变换，最小化内存使用
5. **动态配置**：支持运行时参数修改
6. **实数优化**：针对实数输入的特殊优化路径

该实现代表了现代FFT库的高效设计，在保持代码简洁性的同时提供了优秀的性能表现。