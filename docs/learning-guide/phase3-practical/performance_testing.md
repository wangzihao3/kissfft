# KISS FFT 性能测试指南

本文档介绍如何进行 KISS FFT 的性能测试、分析和优化。

## 目录

1. [性能测试基础](#性能测试基础)
2. [使用 benchkiss](#使用-benchkiss)
3. [自定义性能测试](#自定义性能测试)
4. [性能分析工具](#性能分析工具)
5. [优化技巧](#优化技巧)
6. [对比测试](#对比测试)

---

## 性能测试基础

### 性能指标

**主要指标：**
1. **执行时间**：单次 FFT 的计算时间
2. **吞吐量**：每秒可执行的 FFT 数量
3. **内存使用**：静态和动态内存占用
4. **缓存效率**：缓存命中率
5. **CPU 使用率**：多核利用情况

### 测试方法

```c
// 基础性能测试模板
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kiss_fft.h"

#define DEFAULT_ITERATIONS 10000

double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_fft(int nfft, int iterations) {
    // 分配配置
    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * nfft);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * nfft);

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < nfft; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    // 预热（避免冷启动影响）
    for (int i = 0; i < 100; i++) {
        kiss_fft(cfg, input, output);
    }

    // 计时测试
    double start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        kiss_fft(cfg, input, output);
    }
    double end = get_time_seconds();

    // 计算统计
    double elapsed = end - start;
    double avg_time_ms = elapsed * 1000.0 / iterations;
    double throughput = iterations / elapsed;

    printf("N=%d: %.6f ms/FFT, %.0f FFT/s\n",
           nfft, avg_time_ms, throughput);

    // 清理
    free(input);
    free(output);
    kiss_fft_free(cfg);
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("FFT Size | Time (ms) | FFT/s\n");
    printf("---------|-----------|--------\n");

    for (int i = 0; i < num_sizes; i++) {
        benchmark_fft(sizes[i], DEFAULT_ITERATIONS);
    }

    return 0;
}
```

---

## 使用 benchkiss

KISS FFT 自带性能测试工具 `test/benchkiss.c`。

### 编译 benchkiss

```bash
cd kissfft/test
make benchkiss
```

### 运行测试

```bash
# 基础测试
./benchkiss

# 指定测试参数
./benchkiss -n 1024 -i 10000

# 不同数据类型
./benchkiss -t float
./benchkiss -t int16
```

### 理解输出

```
### KISS FFT 性能测试 ###
N=256:    0.045 ms,  22222 FFT/s
N=512:    0.098 ms,  10204 FFT/s
N=1024:   0.213 ms,   4695 FFT/s
N=2048:   0.467 ms,   2141 FFT/s
N=4096:   1.023 ms,    977 FFT/s
```

**解读：**
- `N`: FFT 大小
- `0.045 ms`: 单次 FFT 平均时间
- `22222 FFT/s`: 每秒可执行的 FFT 数量

---

## 自定义性能测试

### 测试 1：不同 FFT 大小

```c
// benchmark_sizes.c
void benchmark_all_sizes() {
    printf("%-10s | %-12s | %-12s | %-10s\n",
           "Size", "Time (us)", "Throughput", "Complexity");
    printf("-----------|--------------|--------------|------------\n");

    // 测试 2 的幂次
    for (int n = 16; n <= 16384; n *= 2) {
        double time_us = benchmark_fft(n, 10000);
        double throughput = 1.0 / (time_us * 1e-6);

        // 估算复杂度 (O(N log N))
        double complexity = (double)n * log2(n);

        printf("%-10d | %-12.2f | %-12.0f | %.0f\n",
               n, time_us, throughput, complexity);
    }
}
```

### 测试 2：复数 vs 实数 FFT

```c
// benchmark_complex_vs_real.c
#include "kiss_fftr.h"

void benchmark_comparison(int nfft, int iterations) {
    // 复数 FFT
    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_cpx *cpx_input = malloc(sizeof(kiss_fft_cpx) * nfft);
    kiss_fft_cpx *cpx_output = malloc(sizeof(kiss_fft_cpx) * nfft);

    // 实数 FFT
    kiss_fftr_cfg cfg_r = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    float *real_input = malloc(sizeof(float) * nfft);
    kiss_fft_cpx *real_output = malloc(sizeof(kiss_fft_cpx) * (nfft/2+1));

    // 初始化
    for (int i = 0; i < nfft; i++) {
        cpx_input[i].r = real_input[i] = (float)rand() / RAND_MAX;
        cpx_input[i].i = 0;
    }

    // 测试复数 FFT
    double start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        kiss_fft(cfg, cpx_input, cpx_output);
    }
    double cpx_time = get_time_seconds() - start;

    // 测试实数 FFT
    start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        kiss_fftr(cfg_r, real_input, real_output);
    }
    double real_time = get_time_seconds() - start;

    printf("N=%d: 复数=%.3f ms, 实数=%.3f ms, 加速=%.2fx\n",
           nfft, cpx_time * 1000 / iterations, real_time * 1000 / iterations,
           cpx_time / real_time);

    // 清理
    free(cpx_input); free(cpx_output);
    free(real_input); free(real_output);
    kiss_fft_free(cfg);
    kiss_fftr_free(cfg_r);
}
```

### 测试 3：内存使用分析

```c
// benchmark_memory.c
#include <stdlib.h>
#include <stdio.h>
#include "kiss_fft.h"

void analyze_memory_usage(int nfft) {
    // 配置内存
    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);

    // 数据内存
    size_t config_size = 0;
    kiss_fft_alloc(nfft, 0, NULL, &config_size);

    size_t input_size = sizeof(kiss_fft_cpx) * nfft;
    size_t output_size = sizeof(kiss_fft_cpx) * nfft;
    size_t total = config_size + input_size + output_size;

    printf("N=%d 内存使用:\n", nfft);
    printf("  配置: %zu bytes\n", config_size);
    printf("  输入: %zu bytes\n", input_size);
    printf("  输出: %zu bytes\n", output_size);
    printf("  总计: %zu bytes (%.2f KB)\n\n", total, total / 1024.0);

    kiss_fft_free(cfg);
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 5; i++) {
        analyze_memory_usage(sizes[i]);
    }
    return 0;
}
```

---

## 性能分析工具

### 1. 使用 perf (Linux)

```bash
# CPU 周期分析
perf stat -e cycles,instructions,cache-misses ./benchkiss

# 性能分析报告
perf record ./benchkiss
perf report

# 火焰图
perf record -F 99 -g ./benchkiss
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### 2. 使用 gprof

```bash
# 编译时启用性能分析
gcc -pg -O2 -o benchkiss benchkiss.c kiss_fft.c -lm

# 运行程序
./benchkiss

# 生成报告
gprof benchkiss gmon.out > analysis.txt
```

### 3. 使用 Valgrind (缓存分析)

```bash
# 缓存模拟
valgrind --tool=cachegrind ./benchkiss

# 查看结果
cg_annotate cachegrind.out.<pid>
```

### 4. 使用 time 命令

```bash
# 基础计时
/usr/bin/time -v ./benchkiss

# 输出包含：
# - 用户时间
# - 系统时间
# - 最大内存使用
# - 上下文切换次数
```

---

## 优化技巧

### 1. 编译器优化

```cmake
# CMakeLists.txt
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -march=native -funroll-loops")
```

或命令行：

```bash
gcc -O3 -ffast-math -march=native -o benchkiss benchkiss.c kiss_fft.c -lm
```

**选项说明：**
- `-O3`: 最高优化级别
- `-ffast-math`: 快速数学运算（可能牺牲精度）
- `-march=native`: 针对当前 CPU 优化
- `-funroll-loops`: 循环展开

### 2. SIMD 优化

```bash
# SSE2
gcc -msse2 -DUSE_SIMD -O3 -o fft_simd kiss_fft.c

# AVX
gcc -mavx -DUSE_SIMD -O3 -o fft_avx kiss_fft.c

# AVX2
gcc -mavx2 -DUSE_SIMD -O3 -o fft_avx2 kiss_fft.c

# NEON (ARM)
gcc -mfpu=neon -DUSE_SIMD -O3 -o fft_neon kiss_fft.c
```

### 3. 内存对齐

```c
// 使用对齐内存
#include <stdlib.h>

kiss_fft_cpx* allocate_aligned(int n) {
    void *ptr = NULL;
    posix_memalign(&ptr, 64, sizeof(kiss_fft_cpx) * n);  // 64 字节对齐
    return (kiss_fft_cpx*)ptr;
}
```

### 4. 预分配配置

```c
// 错误：每次都分配
void process_audio_frame(float *frame, int n) {
    kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, NULL, NULL);
    // ... 使用 FFT
    kiss_fft_free(cfg);
}

// 正确：预分配
typedef struct {
    kiss_fft_cfg cfg;
    kiss_fft_cpx *buffer;
} FFTProcessor;

FFTProcessor* processor_create(int n) {
    FFTProcessor *p = malloc(sizeof(FFTProcessor));
    p->cfg = kiss_fft_alloc(n, 0, NULL, NULL);
    p->buffer = malloc(sizeof(kiss_fft_cpx) * n);
    return p;
}

void processor_process(FFTProcessor *p, kiss_fft_cpx *input, kiss_fft_cpx *output) {
    kiss_fft(p->cfg, input, output);
}
```

### 5. 批量处理

```c
// 批量处理多个 FFT
void batch_process(kiss_fft_cfg cfg, kiss_fft_cpx **inputs,
                   kiss_fft_cpx **outputs, int nfft, int count) {
    for (int i = 0; i < count; i++) {
        kiss_fft(cfg, inputs[i], outputs[i]);
    }
}
```

---

## 对比测试

### KISS FFT vs FFTW

```c
// benchmark_vs_fftw.c
#include <fftw3.h>

void compare_with_fftw(int nfft, int iterations) {
    // KISS FFT
    kiss_fft_cfg k_cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_cpx *k_in = malloc(sizeof(kiss_fft_cpx) * nfft);
    kiss_fft_cpx *k_out = malloc(sizeof(kiss_fft_cpx) * nfft);

    // FFTW
    fftw_complex *f_in = fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_complex *f_out = fftw_malloc(sizeof(fftw_complex) * nfft);
    fftw_plan f_plan = fftw_plan_dft_1d(nfft, f_in, f_out, FFTW_FORWARD, FFTW_MEASURE);

    // 初始化相同数据
    srand(42);
    for (int i = 0; i < nfft; i++) {
        float val = (float)rand() / RAND_MAX;
        k_in[i].r = f_in[i][0] = val;
        k_in[i].i = f_in[i][1] = val;
    }

    // 测试 KISS FFT
    double start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        kiss_fft(k_cfg, k_in, k_out);
    }
    double kiss_time = get_time_seconds() - start;

    // 测试 FFTW
    start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        fftw_execute(f_plan);
    }
    double fftw_time = get_time_seconds() - start;

    printf("N=%d: KISS FFT=%.3f ms, FFTW=%.3f ms, FFTW 加速=%.2fx\n",
           nfft, kiss_time * 1000 / iterations, fftw_time * 1000 / iterations,
           kiss_time / fftw_time);

    // 清理
    free(k_in); free(k_out);
    kiss_fft_free(k_cfg);
    fftw_destroy_plan(f_plan);
    fftw_free(f_in); fftw_free(f_out);
}
```

### 不同平台性能对比

| 平台 | CPU | N=1024 (μs) | 说明 |
|------|-----|------------|------|
| Desktop | Intel i7-9700K | 15 | 3.6 GHz, 8核 |
| Laptop | Intel i5-8250U | 25 | 1.6 GHz, 4核 |
| Raspberry Pi 4 | ARM Cortex-A72 | 80 | 1.5 GHz, 4核 |
| STM32F4 | ARM Cortex-M4 | 5000 | 168 MHz, 1核 |

---

## 性能测试最佳实践

### 1. 预热

```c
// 避免冷启动影响
for (int i = 0; i < 100; i++) {
    kiss_fft(cfg, input, output);
}

// 然后开始计时
```

### 2. 多次运行取平均

```c
int trials = 10;
double total_time = 0;

for (int t = 0; t < trials; t++) {
    double start = get_time_seconds();
    // ... 运行测试
    double end = get_time_seconds();
    total_time += (end - start);
}

printf("平均时间: %.3f ms\n", total_time / trials * 1000);
```

### 3. 检查正确性

```c
void verify_correctness() {
    // 已知的简单测试用例
    kiss_fft_cpx input[4] = {{1,0}, {1,0}, {1,0}, {1,0}};
    kiss_fft_cpx output[4];

    kiss_fft_cfg cfg = kiss_fft_alloc(4, 0, NULL, NULL);
    kiss_fft(cfg, input, output);
    kiss_fft_free(cfg);

    // 验证：只有直流分量非零
    printf("DC: %.0f+%.0fi\n", output[0].r, output[0].i);
    printf("Others should be ~0\n");
}
```

### 4. 记录环境信息

```c
void print_system_info() {
    printf("=== 系统信息 ===\n");
    printf("编译器: %s\n", __VERSION__);
    printf("优化级别: %s\n",
#ifdef __OPTIMIZE__
    "已启用"
#else
    "未启用"
#endif
    );
    printf("SIMD: %s\n",
#ifdef USE_SIMD
    "已启用"
#else
    "未启用"
#endif
    );
    printf("================\n\n");
}
```

---

## 实践练习

### 练习 1：创建性能测试套件

```bash
# 创建性能测试脚本
cat > run_benchmarks.sh << 'EOF'
#!/bin/bash

echo "=== KISS FFT 性能测试 ==="
echo ""

# 测试不同大小
echo "测试不同 FFT 大小..."
./benchmark_sizes

# 测试复数 vs 实数
echo ""
echo "测试复数 vs 实数 FFT..."
./benchmark_complex_vs_real

# 测试不同优化级别
echo ""
echo "测试不同优化级别..."
for opt in O0 O1 O2 O3; do
    gcc -$opt -o bench_opt benchkiss.c kiss_fft.c -lm
    echo "优化级别: $opt"
    ./bench_opt
done
EOF

chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

### 练习 2：生成性能报告

```python
# generate_report.py
import matplotlib.pyplot as plt
import numpy as np

# 读取性能数据
sizes = np.array([64, 128, 256, 512, 1024, 2048, 4096])
times = np.array([5.2, 11.3, 24.5, 53.1, 115.2, 251.3, 548.7])

# 绘图
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(sizes, times, 'o-')
plt.xlabel('FFT Size')
plt.ylabel('Time (µs)')
plt.title('FFT Performance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.loglog(sizes, times, 'o-')
plt.xlabel('FFT Size')
plt.ylabel('Time (µs)')
plt.title('FFT Performance (log-log)')
plt.grid(True)

plt.tight_layout()
plt.savefig('fft_performance.png')
print("性能报告已保存: fft_performance.png")
```

---

## 总结

- 使用 `benchkiss` 进行快速性能测试
- 创建自定义测试满足特定需求
- 使用性能分析工具找出瓶颈
- 尝试不同的编译优化选项
- 考虑 SIMD、并行化等高级优化

**下一步：** 掌握性能测试后，进入[阶段 4：高级特性](../phase4-advanced/)。

---

**参考资源：**
- [benchkiss.c](../../test/benchkiss.c)
- [性能优化指南](../phase4-advanced/README.md)
- [FFTW 性能优化](http://www.fftw.org/optimization.html)
