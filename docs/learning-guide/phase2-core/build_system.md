# KISS FFT 构建系统指南

本文档详细介绍 KISS FFT 的构建系统，包括不同的编译方法和配置选项。

## 目录

1. [构建系统概览](#构建系统概览)
2. [使用 Makefile](#使用-makefile)
3. [使用 CMake](#使用-cmake)
4. [编译选项](#编译选项)
5. [跨平台编译](#跨平台编译)
6. [集成到项目](#集成到项目)

---

## 构建系统概览

KISS FFT 支持多种构建系统：

- **Make**：传统的 Unix Makefile
- **CMake**：跨平台构建系统
- **手动编译**：直接编译单个文件

### 项目根目录文件

```
kissfft/
├── Makefile           # 主 Makefile（用于编译测试和工具）
├── CMakeLists.txt     # CMake 配置
├── kiss_fft.h         # 核心头文件
├── kiss_fft.c         # 核心源文件
└── test/              # 测试代码
```

---

## 使用 Makefile

### 基础编译

#### 编译静态库

```bash
# 在项目根目录
make

# 这会生成 libkissfft.a 静态库
```

#### 编译测试程序

```bash
# 编译所有测试
make test

# 编译单个测试
make -C test/kiss_fft_test
make -C test/kiss_fftr_test
```

#### 编译示例程序

```bash
# 编译示例代码
make -C test/benchkiss        # 性能测试
make -C test/fftutil          # FFT 命令行工具
```

### Makefile 目标

查看主 Makefile：

```makefile
# Makefile 示例内容
PREFIX = /usr/local

# 默认目标：编译库
all: kiss_fft.o kiss_fftr.o

# 编译核心 FFT
kiss_fft.o: kiss_fft.c kiss_fft.h kiss_fft_log.h _kiss_fft_guts.h
	$(CC) -c $(CFLAGS) -DKISS_FFT_BUILD $(DEFINES) kiss_fft.c -o kiss_fft.o

# 编译实数 FFT
kiss_fftr.o: kiss_fftr.c kiss_fftr.h kiss_fft.h _kiss_fft_guts.h
	$(CC) -c $(CFLAGS) -DKISS_FFT_BUILD $(DEFINES) kiss_fftr.c -o kiss_fftr.o

# 编译测试
test: all
	cd test && $(MAKE)

# 清理
clean:
	rm -f *.o *.a
	cd test && $(MAKE) clean

# 安装
install: all
	install -d $(PREFIX)/lib $(PREFIX)/include
	install libkissfft.a $(PREFIX)/lib
	install kiss_fft.h kiss_fftr.h $(PREFIX)/include
```

### 使用示例

```bash
# 1. 查看所有可用目标
make help

# 2. 编译库
make all

# 3. 编译并运行测试
make test
./test/kiss_fft_test

# 4. 安装到系统
sudo make install PREFIX=/usr/local

# 5. 清理构建文件
make clean
```

---

## 使用 CMake

### 基础使用

```bash
# 创建构建目录
mkdir build
cd build

# 配置
cmake ..

# 编译
cmake --build .

# 或者使用 make
make

# 安装
sudo make install
```

### CMake 选项

```bash
# 查看所有选项
cmake -L ..

# 常用选项
cmake .. -DBUILD_TESTS=ON           # 编译测试
cmake .. -DBUILD_TOOLS=ON           # 编译工具
cmake .. -DUSE_SIMD=ON              # 启用 SIMD 优化
cmake .. -DPACKAGE_INSTALL_PREFIX=/usr/local
```

### 交叉编译

```bash
# ARM Linux
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm-linux.cmake

# Windows (MinGW)
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-mingw.cmake

# Android
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=armeabi-v7a
```

---

## 编译选项

### 核心编译宏

#### 1. 定点数 vs 浮点数

```bash
# 浮点数（默认）
gcc -c kiss_fft.c -o kiss_fft.o

# 16位定点数
gcc -DFIXED_POINT=16 -c kiss_fft.c -o kiss_fft.o

# 32位定点数
gcc -DFIXED_POINT=32 -c kiss_fft.c -o kiss_fft.o
```

**代码示例：**

```c
// 在代码中检测定点数模式
#ifdef FIXED_POINT
    #if (FIXED_POINT == 32)
        typedef int32_t kiss_fft_scalar;
    #else
        typedef int16_t kiss_fft_scalar;
    #endif
#else
    typedef float kiss_fft_scalar;
#endif
```

#### 2. 使用 alloca

```bash
# 使用栈分配（小 FFT 时更快）
gcc -DKISS_FFT_USE_ALLOCA -c kiss_fft.c

# 使用堆分配（默认）
gcc -c kiss_fft.c
```

#### 3. SIMD 优化

```bash
# SSE2
gcc -msse2 -DUSE_SIMD -c kiss_fft.c

# AVX
gcc -mavx -DUSE_SIMD -c kiss_fft.c

# NEON (ARM)
gcc -mfpu=neon -DUSE_SIMD -c kiss_fft.c
```

#### 4. 调试选项

```bash
# 调试版本
gcc -g -O0 -DKISS_FFT_DEBUG -c kiss_fft.c

# 性能分析
gcc -g -pg -c kiss_fft.c

# 优化版本
gcc -O3 -ffast-math -c kiss_fft.c
```

### 完整编译示例

#### 示例 1：编译为共享库

```bash
# 编译为位置无关代码
gcc -fPIC -O2 -c kiss_fft.c -o kiss_fft.o

# 创建共享库
gcc -shared -o libkissfft.so kiss_fft.o
```

#### 示例 2：静态链接库

```bash
# 创建静态库
ar rcs libkissfft.a kiss_fft.o kiss_fftr.o

# 查看库内容
ar t libkissfft.a
```

#### 示例 3：在项目中使用

```makefile
# 项目 Makefile
KISSFFT_DIR = ../kissfft
CFLAGS = -I$(KISSFFT_DIR)
LDFLAGS = -L$(KISSFFT_DIR) -lkissfft -lm

myapp: main.o
	gcc -o myapp main.o $(LDFLAGS)

main.o: main.c
	gcc $(CFLAGS) -c main.c
```

---

## 跨平台编译

### Windows

#### 使用 MinGW

```bash
# 配置环境
export PATH=/path/to/mingw/bin:$PATH

# 编译
gcc -O2 -c kiss_fft.c -o kiss_fft.o
ar rcs libkissfft.a kiss_fft.o
```

#### 使用 Visual Studio

1. 创建静态库项目
2. 添加 kiss_fft.c 和 kiss_fft.h
3. 配置项目属性：
   - 配置类型：静态库 (.lib)
   - C++ 语言标准：C99
   - 预处理器定义：添加需要的宏

### macOS

```bash
# 使用 Xcode 的 clang
clang -O2 -c kiss_fft.c -o kiss_fft.o

# 创建通用二进制（Intel + ARM）
clang -arch x86_64 -arch arm64 -c kiss_fft.c -o kiss_fft.o

# 框架版本
xcodebuild -project KissFFT.xcodeproj -scheme KissFFT
```

### Linux

```bash
# 标准 GCC 编译
gcc -O2 -c kiss_fft.c -o kiss_fft.o

# 针对不同架构
gcc -march=native -O2 -c kiss_fft.c  # 优化为当前 CPU
```

### 嵌入式系统

#### ARM Cortex-M

```bash
# 使用 ARM GCC 工具链
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard \
                  -mfpu=fpv4-sp-d16 -O2 \
                  -DFIXED_POINT=16 \
                  -c kiss_fft.c -o kiss_fft.o
```

#### 交叉编译到嵌入式 Linux

```bash
# ARM Linux
arm-linux-gnueabihf-gcc -O2 -c kiss_fft.c -o kiss_fft.o

# MIPS
mips-linux-gnu-gcc -O2 -c kiss_fft.c -o kiss_fft.o
```

---

## 集成到项目

### 方法 1：直接编译源文件

```makefile
# 最简单的方法：直接编译源文件
SOURCES = kiss_fft.c kiss_fftr.c main.c
TARGET = myapp

myapp: $(SOURCES)
	gcc -O2 -o $(TARGET) $(SOURCES) -lm
```

### 方法 2：静态链接

```bash
# 1. 编译 KISS FFT 库
cd kissfft
make

# 2. 在项目中使用
cd myproject
gcc -I../kissfft -c main.c
gcc -o main.o -L../kissfft -lkissfft -lm
```

### 方法 3：动态链接

```bash
# 1. 编译共享库
gcc -fPIC -c kiss_fft.c
gcc -shared -o libkissfft.so kiss_fft.o

# 2. 链接
gcc -o myapp main.c -L. -lkissfft -lm

# 3. 运行时需要指定库路径
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./myapp
```

### 方法 4：CMake 集成

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyApp)

# 添加 KISS FFT 子目录
add_subdirectory(kissfft)

# 链接 KISS FFT
add_executable(myapp main.c)
target_link_libraries(myapp kissfft)
```

或者使用 FetchContent：

```cmake
include(FetchContent)

FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft
    GIT_TAG master
)

FetchContent_MakeAvailable(kissfft)

add_executable(myapp main.c)
target_link_libraries(myapp kissfft)
```

### 方法 5：头文件集成

```bash
# 将 KISS FFT 复制到项目
mkdir -p lib/kissfft
cp kiss_fft.* lib/kissfft/
cp kiss_fftr.* lib/kissfft/

# 在代码中包含
#include "kissfft/kiss_fft.h"
```

---

## 编译配置检查清单

### ✅ 验证编译成功

```bash
# 1. 编译测试程序
make test

# 2. 运行基本测试
./test/kiss_fft_test

# 3. 检查输出（应该全部通过）
# All tests passed
```

### ✅ 性能验证

```bash
# 运行性能测试
./test/benchkiss

# 输出示例：
# n=256:    0.123 ms
# n=512:    0.234 ms
# n=1024:   0.456 ms
# n=2048:   0.987 ms
```

### ✅ 正确性验证

```bash
# 使用 Python 测试脚本
cd test
python3 testkiss.py

# 应该输出：
# Testing complex FFT... OK
# Testing real FFT... OK
```

---

## 常见编译问题

### 问题 1：找不到 math.h

```bash
# 解决方案：确保链接数学库
gcc myapp.c -lm
```

### 问题 2：未定义的引用

```bash
# 错误：undefined reference to 'kiss_fft_alloc'
# 解决方案：确保编译了所有需要的源文件
gcc kiss_fft.c kiss_fftr.c myapp.c -o myapp
```

### 问题 3：SIMD 编译错误

```bash
# 如果 CPU 不支持 SIMD
# 解决方案：不启用 SIMD 选项
gcc -c kiss_fft.c  # 不要加 -msse2 或其他 SIMD 选项
```

### 问题 4：Windows 下 long double 问题

```bash
# KISS FFT 不使用 long double
# 如果遇到问题，检查编译器设置
gcc -Dkiss_fft_scalar=float ...
```

---

## 实践练习

### 练习 1：编译并测试

```bash
# 1. 克隆或进入项目
cd kissfft

# 2. 使用 Makefile 编译
make
make test

# 3. 使用 CMake 编译
mkdir build && cd build
cmake ..
make
ctest

# 4. 对比两种方法
```

### 练习 2：创建自定义编译脚本

```bash
#!/bin/bash
# build_kissfft.sh

KISSFFT_DIR="kissfft"
BUILD_TYPE="${1:-Release}"  # Release 或 Debug

echo "Building KISS FFT in $BUILD_TYPE mode..."

if [ "$BUILD_TYPE" = "Debug" ]; then
    CFLAGS="-g -O0 -DKISS_FFT_DEBUG"
else
    CFLAGS="-O3 -ffast-math"
fi

gcc $CFLAGS -c $KISSFFT_DIR/kiss_fft.c -o kiss_fft.o
ar rcs libkissfft.a kiss_fft.o

echo "Build complete: libkissfft.a"
```

使用：

```bash
chmod +x build_kissfft.sh
./build_kissfft.sh Release
./build_kissfft.sh Debug
```

### 练习 3：交叉编译练习

```bash
# 尝试为不同平台编译
./build_kissfft.sh native     # 本地平台
./build_kissfft.sh arm        # ARM
./build_kissfft.sh windows    # Windows
```

---

## 总结

- KISS FFT 支持多种构建系统
- Makefile 适合简单的 Unix 开发
- CMake 提供跨平台支持
- 可以根据需求选择不同的编译选项
- 集成到项目有多种方法

**下一步：** 在理解了构建系统后，让我们探索[数据类型切换的方法](./data_types.md)。

---

**参考资源：**
- [项目 README](../../README.md)
- [CMake 文档](https://cmake.org/documentation/)
- [GCC 编译选项](https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html)
