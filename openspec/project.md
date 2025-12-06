# Project Context

## Purpose
KISS FFT 是一个基于"Keep It Simple, Stupid"原则的混合基数快速傅里叶变换库。它旨在提供一个：
- 相当高效的 FFT 实现
- 支持定点和浮点数据类型
- 可以在几分钟内集成到 C 程序中的轻量级库
- 使用宽松的 BSD 许可证

KISS FFT 并非试图成为比其他现有 FFT 库更好的实现，而是专注于简单性和易用性。

## Tech Stack
- **主要语言**: C 语言
- **构建系统**:
  - CMake (>= 3.10) - 现代构建系统
  - Make - 传统 Unix/Linux 构建系统
- **支持的数据类型**:
  - float (默认)
  - double
  - int16_t (16位定点)
  - int32_t (32位定点)
  - simd (需要 SSE 指令集支持)
- **编译器支持**:
  - GCC
  - Clang
  - MSVC (Microsoft Visual C++)
- **并行计算**: OpenMP (可选)
- **测试依赖**:
  - libpng (用于 psdpng 工具)
  - libfftw3 (用于验证结果)
  - Python 2/3 with NumPy (用于验证)

## Project Conventions

### Code Style
- 使用 C 语言标准风格
- 函数命名使用下划线分隔的小写字母 (kiss_fft_alloc, kiss_fft_cfg)
- 宏定义使用全大写字母 (KISS_FFT_SHARED, FIXED_POINT)
- 严格的编译警告设置:
  - -Wall -Wcast-align -Wcast-qual -Wshadow -Wwrite-strings
  - 对于 C: -Wstrict-prototypes -Wmissing-prototypes -Wnested-externs -Wbad-function-cast
- 优化标志: -ffast-math -fomit-frame-pointer

### Architecture Patterns
- **核心设计原则**: KISS (Keep It Simple, Stupid)
- **FFT 算法**: 时间抽取、混合基数、就地变换
- **模块化设计**:
  - 核心 FFT: kiss_fft.c/h
  - 多维 FFT: kiss_fftnd.c/h
  - 实数 FFT: kiss_fftr.c/h
  - 实数多维 FFT: kiss_fftndr.c/h
- **工具模块**: tools/ 目录包含实用工具
  - fftutil.c - 命令行 FFT 工具
  - kiss_fastfir.c - 快速卷积滤波
  - psdpng.c - 频谱图像生成
- **无静态数据**: 所有核心函数都是线程安全的
- **内存管理**: 支持 malloc/free 或 alloca

### Testing Strategy
- **单元测试**: test/ 目录包含各种测试用例
  - test_real.c - 实数 FFT 测试
  - test_simd.c - SIMD 优化测试
  - twotonetest.c - 双音测试
- **性能测试**:
  - benchkiss.c - KISS FFT 性能测试
  - benchfftw.c - FFTW 性能对比
- **验证测试**:
  - 与 FFTW 库结果对比验证
  - Python 脚本验证 (testkiss.py)
- **完整测试套件**: test/kissfft-testsuite.sh 测试所有配置

### Git Workflow
- **主分支**: master
- **版本控制**: 使用语义化版本控制
  - KFVER_MAJOR: ABI 版本 (当前: 131)
  - KFVER_MINOR: 次版本 (当前: 1)
  - KFVER_PATCH: 补丁版本 (当前: 0)
- **提交规范**: 遵循常规的 Git 提交信息格式

## Domain Context
KISS FFT 是一个数字信号处理领域的库，主要用于：
- **信号处理**: 音频、视频、通信信号的分析和处理
- **频谱分析**: 将时域信号转换为频域表示
- **滤波**: 在频域进行滤波操作
- **卷积**: 使用 FFT 进行快速卷积运算
- **多维信号处理**: 图像处理、2D/3D 信号分析

**关键概念**:
- **FFT (Fast Fourier Transform)**: 快速傅里叶变换算法
- **时域/频域**: 信号的两种表示方式
- **实数/复数 FFT**: 处理不同类型的输入信号
- **定点/浮点**: 不同的数值表示方式，适用于不同的精度和性能需求

## Important Constraints
- **性能约束**: 不是世界上最快的 FFT，优先考虑简单性
- **内存约束**: 核心代码约 500 行，避免代码膨胀
- **许可证约束**: 使用 BSD-3-Clause 许可证，允许商业和开源使用
- **兼容性约束**:
  - 支持 C99 标准
  - 跨平台兼容 (Linux, Windows, macOS)
  - 避免使用汇编代码以保持可移植性
- **API 稳定性**: ABI 版本控制确保向后兼容性

## External Dependencies
- **构建时依赖**:
  - CMake >= 3.10 或 GNU Make
  - C 编译器 (GCC/Clang/MSVC)
- **可选依赖**:
  - OpenMP 运行时库 (用于并行计算)
  - libpng (用于 psdpng 工具)
  - libfftw3 (用于性能对比和验证)
  - Python with NumPy (用于测试验证)
- **系统库**:
  - libm (数学库，在 Linux/Unix 系统上)
- **平台特定**:
  - Windows: MSVC 编译器支持
  - x86/x86_64: SSE 指令集支持 (用于 SIMD 优化)
