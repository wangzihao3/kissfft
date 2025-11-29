# test/testcpp.cc FFT算法测试程序技术文档

## 概述

`test/testcpp.cc` 是一个用于测试KISS FFT库功能的C++测试程序。该程序通过模板化的方式测试不同精度类型（float、double、long double）的FFT实现，并验证算法的准确性。

## 主要功能

### 1. 时间测量函数

```cpp
static inline double curtime(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec*.000001;
}
```

该函数使用 `gettimeofday()` 系统调用获取高精度时间戳，用于性能测试中的时间测量。

### 2. 模板化测试函数 `dotest<T>`

#### 函数签名
```cpp
template <class T> void dotest(int nfft)
```

#### 功能流程

1. **初始化阶段**
   - 根据模板类型 `T` 实例化 `kissfft<T>` 和 `std::complex<T>`
   - 创建FFT对象：`FFT fft(nfft,false)`（第二个参数 `false` 表示正向FFT）
   - 分配输入和输出缓冲区

2. **测试数据生成**
   ```cpp
   for (int k=0;k<nfft;++k)
       inbuf[k]= cpx_type(
               (T)(rand()/(double)RAND_MAX - .5),
               (T)(rand()/(double)RAND_MAX - .5) );
   ```
   - 生成随机复数数据作为FFT输入
   - 实部和虚部范围在 [-0.5, 0.5] 之间

3. **FFT变换**
   ```cpp
   fft.transform( &inbuf[0] , &outbuf[0] );
   ```
   - 对输入数据执行FFT变换

4. **准确性验证**
   程序通过直接计算DFT（离散傅里叶变换）来验证FFT结果的准确性：

   ```cpp
   const long double pi = acosl(-1);  // 计算π值

   for (int k0=0;k0<nfft;++k0) {
       complex<long double> acc = 0;
       long double phinc = 2*k0* pi / nfft;
       for (int k1=0;k1<nfft;++k1) {
           complex<long double> x(inbuf[k1].real(),inbuf[k1].imag());
           acc += x * exp( complex<long double>(0,-k1*phinc) );
       }
       // 计算总功率和差值功率
   }
   ```

   这里实现了标准的DFT公式：
   $$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N}$$

5. **性能测试**
   ```cpp
   int nits=20e6/nfft;  // 根据FFT大小调整迭代次数
   for (int k=0;k<nits;++k) {
       fft.transform( &inbuf[0] , &outbuf[0] );
   }
   ```
   - 执行多次FFT变换来测量性能
   - 计算MSPS（Million Samples Per Second，每秒百万样本数）

## 程序入口 `main`

### 命令行参数处理

- **有参数模式**：解析命令行参数作为FFT大小进行测试
  ```cpp
  for (int k=1;k<argc;++k) {
      int nfft = atoi(argv[k]);
      dotest<float>(nfft);
      dotest<double>(nfft);
      dotest<long double>(nfft);
  }
  ```

- **无参数模式**：使用预设的FFT大小进行测试
  - 32点FFT（小规模）
  - 1024点FFT（中等规模）
  - 840点FFT（特定规模）

## 输出指标说明

### 1. RMSE（Root Mean Square Error，均方根误差）
```cpp
cout << " RMSE:" << sqrt(difpower/totalpower) << "\t";
```
- 衡量FFT结果与理论DFT结果之间的相对误差
- 数值越小表示算法准确性越高

### 2. MSPS（Million Samples Per Second）
```cpp
cout << " MSPS:" << (nits*nfft)*1e-6/ (t1-t0) << endl;
```
- 衡量FFT算法的处理速度
- 表示每秒能够处理多少百万个样本点
- 数值越大表示算法性能越高

## 算法验证机制

程序采用双重验证机制：

1. **理论对比验证**：通过直接计算DFT公式作为理论基准
2. **统计误差分析**：使用相对均方根误差来量化算法精度

这种验证方法确保了FFT实现的正确性，同时提供了性能基准测试。

## 技术特点

1. **模板化设计**：支持多种数值精度类型
2. **随机测试数据**：避免特殊数据集导致的偏差
3. **高精度验证**：使用 `long double` 进行理论计算以确保精度
4. **性能自适应**：根据FFT大小动态调整测试次数
5. **实时性能监控**：提供准确的算法性能指标

该测试程序为KISS FFT库提供了全面的正确性验证和性能评估工具。