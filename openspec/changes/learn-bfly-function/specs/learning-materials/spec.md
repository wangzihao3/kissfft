# 学习材料规格

## ADDED Requirements

### Requirement: FFT 数学公式推导文档

**描述**：项目 SHALL 创建一份完整的 FFT 核心公式推导文档，从傅里叶级数开始，逐步推导到蝶形运算公式。

**理由**：现有的学习文档缺少数学推导过程，学习者难以理解 FFT 算法的数学基础和蝶形运算公式的由来。

#### Scenario: 理论基础学习

**Given** 学习者具备基本的高等数学知识（包括复数、三角函数、级数）

**When** 阅读 `fft-mathematical-derivation.md` 文档

**Then** 应该能够：
- 从傅里叶级数推导出连续时间傅里叶变换
- 从连续时间傅里叶变换推导出离散时间傅里叶变换 (DTFT)
- 从 DTFT 推导出离散傅里叶变换 (DFT)
- 理解 DFT 的矩阵表示形式
- 分析直接计算 DFT 的计算复杂度 O(N²)

#### Scenario: Cooley-Tukey FFT 算法推导

**Given** 学习者已理解 DFT 定义和基本性质

**When** 跟随文档中的推导步骤

**Then** 应该能够：
- 理解时域抽取（DIT）算法的分解思想
- 推导出基-2 DIT FFT 的递归公式
- 绘制 N=8 的完整蝶形图
- 理解蝶形运算公式的数学由来：
  ```
  X[k]     = G[k] + W_N^k · H[k]
  X[k+N/2] = G[k] - W_N^k · H[k]
  ```
- 理解旋转因子 W_N^k = e^(-j2πk/N) 的作用
- 分析 FFT 的计算复杂度降低到 O(N log N)

#### Scenario: 混合基数 FFT 推导

**Given** 学习者已掌握基-2 FFT 算法

**When** 学习混合基数推广部分

**Then** 应该能够：
- 理解 N = p₁ × p₂ × ... × p_k 的分解策略
- 推导基-4 和基-8 的优化公式
- 理解不同基数选择对性能的影响
- 解释为什么优先分解因子 4

#### Scenario: 数值精度和误差分析

**Given** 学习者已理解 FFT 算法原理

**When** 阅读数值考虑章节

**Then** 应该能够：
- 理解浮点运算的舍入误差来源
- 计算定点数运算的动态范围
- 理解定标（scaling）策略防止溢出
- 分析不同 FFT 长度的误差累积

---

### Requirement: 蝶形运算详细说明文档

**描述**：项目 SHALL 创建一份图文并茂的蝶形运算专题文档，详细解释蝶形图的绘制、数据流动和旋转因子的作用。

**理由**：蝶形运算是 FFT 的核心计算单元，但现有文档缺少系统的讲解，学习者难以建立直观认识。

#### Scenario: 理解蝶形图

**Given** 学习者已了解 FFT 基本原理

**When** 阅读 `butterfly-operations.md` 文档

**Then** 应该能够：
- 解释为什么叫"蝶形"运算（形状像蝴蝶翅膀）
- 绘制基-2 蝶形运算的基本图示
- 理解箭头表示的数据依赖关系
- 理解菱形节点代表的计算操作

#### Scenario: 多阶段蝶形图理解

**Given** FFT 长度 N = 8

**When** 分析完整的蝶形图

**Then** 应该能够：
- 绘制 3 阶段（log₂8 = 3）的完整蝶形图
- 解释每一阶段的数据处理模式
- 理解原位计算（in-place）的原理
- 解释位反转排序的必要性
- 标注每个节点的旋转因子值

#### Scenario: 旋转因子理解

**Given** 复数和欧拉公式的基础知识

**When** 学习旋转因子章节

**Then** 应该能够：
- 定义旋转因子 W_N^k = e^(-j2πk/N)
- 在单位圆上表示旋转因子
- 利用旋转因子的对称性：
  - W_N^(k+N/2) = -W_N^k
  - W_N^(k+N) = W_N^k
- 计算任意 N 和 k 的旋转因子值
- 理解预计算旋转因子的优化意义

#### Scenario: 不同基数蝶形运算对比

**Given** 学习者已掌握基-2 蝶形运算

**When** 比较不同基数实现

**Then** 应该能够：
- 对比基-2、基-4、基-8 的复数乘法次数
- 理解基-4 的特殊优化（减少乘法）
- 解释为什么混合 radix 比单一 radix 更灵活
- 选择合适的基数分解策略

#### Scenario: 实践练习

**Given** 完整阅读蝶形运算文档

**When** 完成文档中的练习题

**Then** 应该能够：
- 手动计算 N=4 或 N=8 的简单 FFT
- 绘制给定 N 值的蝶形图
- 验证蝶形运算结果的正确性
- 分析具体实例中的旋转因子取值

---

### Requirement: kf_bfly 代码分析文档

**描述**：项目 SHALL 创建 kf_bfly 函数族的详细代码分析文档，逐行解释实现细节，并建立代码与数学公式的对应关系。

**理由**：现有的代码解剖文档（kiss_fft_anatomy.md）对 kf_bfly 的讲解较简略，需要更深入的分析帮助学习者理解实现细节。

#### Scenario: 理解 kf_bfly 函数族结构

**Given** 学习者已阅读 kiss_fft.h 和 _kiss_fft_guts.h

**When** 阅读 `kf_bfly-code-analysis.md` 文档

**Then** 应该能够：
- 列出所有 kf_bfly 函数：kf_bfly2, kf_bfly3, kf_bfly4, kf_bfly5, kf_bfly_generic
- 解释每个函数的参数含义：
  - `Fout`: 输出数据指针
  - `fstride`: 旋转因子步长
  - `st`: FFT 配置结构体
  - `m`: 当前蝶形组的大小
- 说明函数的调用时机和上下文

#### Scenario: kf_bfly2 代码解析

**Given** 基-2 蝶形运算公式和 kf_bfly2 函数代码

**When** 逐行分析代码

**Then** 应该能够：
- 建立代码与数学公式的对应关系：

  | 数学符号 | 代码变量 | 含义 |
  |---------|---------|------|
  | G[k] | *Fout | 偶数点 DFT |
  | H[k] | *Fout2 | 奇数点 DFT |
  | W_N^k | *tw1 | 旋转因子 |
  | W_N^k · H[k] | t | 临时变量 |

- 解释 C_MUL, C_SUB, C_ADDTO 宏的作用
- 理解循环结构：`do { ... } while (--m)`
- 解释 fstride 参数的作用和变化规律

#### Scenario: kf_bfly4 优化技巧分析

**Given** kf_bfly4 函数实现

**When** 分析优化技巧

**Then** 应该能够：
- 理解基-4 相比两次基-2 的优势
- 解释 scratch 数组的作用和复用
- 分析特殊的复数运算优化：
  ```c
  Fout[m].r = scratch[5].r - scratch[4].i;  // 利用 j 的性质
  ```
- 解释正变换和逆变换的处理差异

#### Scenario: 定点数定标分析

**Given** FIXED_POINT 宏定义和 C_FIXDIV 宏

**When** 分析定点数实现

**Then** 应该能够：
- 计算每级蝶形运算后的幅度增长
- 理解右移定标的原理
- 解释为什么定点数需要 C_FIXDIV 而浮点数不需要
- 分析不同定点数位宽（16/32 位）的限制

#### Scenario: kf_bfly3 和 kf_bfly5 特殊处理

**Given** kf_bfly3 和 kf_bfly5 函数实现

**When** 分析非 2 的幂次基数

**Then** 应该能够：
- 理解基-3 的特殊角度优化（W_3^1, W_3^2）
- 分析基-5 的复数乘法减少技巧
- 对比不同基数的代码复杂度和性能

#### Scenario: kf_bfly_generic 通用实现

**Given** kf_bfly_generic 函数

**When** 分析通用算法

**Then** 应该能够：
- 理解通用实现如何处理任意基数 p
- 解释双重循环结构
- 分析通用实现的性能权衡
- 理解何时使用 generic vs 专用函数

---

## 交叉引用

### 相关 Capabilities

- **learning-guide**: 本规格扩展了现有的学习指南体系
- **code-documentation**: 与 kiss_fft_anatomy.md 互补，提供更深入的分析

### 依赖关系

- 本规格不修改任何现有代码或文档
- 新文档将添加到 `docs/learning-guide/phase2-core/` 目录
- 与现有的 phase2-core 学习材料保持一致性

### 后续扩展

- 可基于本规格添加交互式可视化演示
- 可扩展为视频教程系列
- 可添加更多练习题和实验项目
