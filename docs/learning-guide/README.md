# KISS FFT 学习指南

欢迎来到 KISS FFT 学习指南！本指南将帮助您从零开始掌握 FFT 算法和 KISS FFT 库的使用。

## 🎯 学习完成状态

**所有学习资源已完成！** ✅

查看 [完成总结](./COMPLETION_SUMMARY.md) 了解完整的学习资源列表和快速开始指南。

## 学习路径概述

本学习指南分为四个阶段，每个阶段都有明确的学习目标和实践任务：

| 阶段 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [**阶段 1**](./phase1-theory/) | FFT 理论基础 | ✅ 完成 | 建立坚实的数字信号处理和 FFT 理论基础 |
| [**阶段 2**](./phase2-core/) | 核心代码探索 | ✅ 完成 | 深入理解 KISS FFT 的实现细节 |
| [**阶段 3**](./phase3-practical/) | 实践应用 | ✅ 完成 | 将理论转化为实际应用（含可编译示例） |
| [**阶段 4**](./phase4-advanced/) | 高级主题 | ✅ 完成 | 探索优化和扩展技术 |

## 开始之前

### 前置知识

- C 语言编程基础
- 基本的数学知识（复数、三角函数）
- 线性代数基础（矩阵、向量）

### 开发环境准备

1. **编译器**
   - GCC 或 Clang (Linux/macOS)
   - MSVC (Windows)

2. **构建工具**
   - CMake 3.10+
   - 或 GNU Make

3. **可选工具**
   - Git（用于版本控制）
   - GDB 或其他调试器
   - 性能分析工具（gprof、perf）

## 学习建议

1. **理论与实践结合**：每个理论概念都应通过代码实验来加深理解
2. **循序渐进**：不要跳过阶段，确保每个阶段都充分掌握
3. **主动探索**：尝试修改代码，观察结果变化
4. **记录笔记**：记录重要的概念和发现
5. **寻求帮助**：遇到困难时，查阅文档或寻求社区支持

## 时间安排

- **总时长**：6-9 周
- **每周投入**：10-15 小时
- **理论学习**：40%
- **编程实践**：60%

## 评估方式

每个阶段结束后，建议进行自我评估：
- [ ] 理论知识理解测验
- [ ] 代码分析任务
- [ ] 编程练习完成
- [ ] 项目实践作品

## 学习资源

### 必读文档
- KISS FFT README.md
- TIPS 性能优化指南
- 各个头文件的注释

### 推荐书籍
- 《数字信号处理》- Oppenheim
- 《快速傅里叶变换及其应用》- Brigham

### 在线资源
- [Understanding the FFT](https://jackschaedler.github.io/circles-sines-signals/)（交互式教程）
- [The Scientist and Engineer's Guide to DSP](http://www.dspguide.com/)（免费在线书籍）

## 开始学习

### 快速开始（阶段 3 实践示例）

如果您已经具备 FFT 理论基础，可以直接运行实践示例：

```bash
cd docs/learning-guide/phase3-practical
make           # 编译所有示例
make run       # 运行所有示例
```

### 完整学习路径

准备好开始您的 FFT 学习之旅了吗？

1. **初学者**：点击[阶段 1：理论基础](./phase1-theory/)开始！
2. **有经验者**：从[阶段 2：核心代码](./phase2-core/)开始深入
3. **实践者**：直接跳转到[阶段 3：实践应用](./phase3-practical/)
4. **高级用户**：探索[阶段 4：高级主题](./phase4-advanced/)

## 📚 新增资源

本次更新新增了以下学习资源：

- ✨ **构建系统指南** ([build_system.md](phase2-core/build_system.md)) - Makefile 和 CMake 使用详解
- ✨ **数据类型切换指南** ([data_types.md](phase2-core/data_types.md)) - 浮点/定点数类型切换
- ✨ **基础 FFT 示例** ([basic_fft.c](phase3-practical/basic_fft.c)) - 完整的可编译示例
- ✨ **实数 FFT 示例** ([real_fft_example.c](phase3-practical/real_fft_example.c)) - 实数 FFT 使用演示
- ✨ **性能测试指南** ([performance_testing.md](phase3-practical/performance_testing.md)) - 性能分析和优化
- ✨ **便捷 Makefile** ([Makefile](phase3-practical/Makefile)) - 一键编译和运行示例

---

**查看 [完成总结](./COMPLETION_SUMMARY.md) 了解完整的学习资源！**

Happy Learning! 🎵📊