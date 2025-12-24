# Makefile for KISS FFT Learning Guide - Phase 3 Examples
# 用于编译阶段 3 的所有示例程序

# 编译器和选项
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm

# KISS FFT 路径
KISSFFT_ROOT = ../../..
CFLAGS += -I$(KISSFFT_ROOT)

# 源文件
KISSFFT_SRC = $(KISSFFT_ROOT)/kiss_fft.c
KISSFFTR_SRC = $(KISSFFT_ROOT)/kiss_fftr.c

# 目标程序
TARGETS = basic_fft real_fft_example

# 颜色输出
COLOR_RESET = \033[0m
COLOR_BOLD = \033[1m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

.PHONY: all clean help test run

# 默认目标
all: $(TARGETS)
	@echo "$(COLOR_GREEN)✓ 所有示例程序编译完成！$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)可用的程序：$(COLOR_RESET)"
	@echo "  $(COLOR_BLUE)basic_fft$(COLOR_RESET)         - 基础 FFT 使用示例"
	@echo "  $(COLOR_BLUE)real_fft_example$(COLOR_RESET)  - 实数 FFT 使用示例"
	@echo ""
	@echo "$(COLOR_BOLD)运行示例：$(COLOR_RESET)"
	@echo "  make run_basic"
	@echo "  make run_real"

# 基础 FFT 示例
basic_fft: basic_fft.c
	@echo "$(COLOR_YELLOW)编译 basic_fft...$(COLOR_RESET)"
	$(CC) $(CFLAGS) -o $@ $< $(KISSFFT_SRC) $(LDFLAGS)
	@echo "$(COLOR_GREEN)✓ basic_fft 编译成功$(COLOR_RESET)"

# 实数 FFT 示例
real_fft_example: real_fft_example.c
	@echo "$(COLOR_YELLOW)编译 real_fft_example...$(COLOR_RESET)"
	$(CC) $(CFLAGS) -o $@ $< $(KISSFFTR_SRC) $(KISSFFT_SRC) $(LDFLAGS)
	@echo "$(COLOR_GREEN)✓ real_fft_example 编译成功$(COLOR_RESET)"

# 运行基础 FFT 示例
run_basic: basic_fft
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)运行基础 FFT 示例$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@./basic_fft

# 运行实数 FFT 示例
run_real: real_fft_example
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)运行实数 FFT 示例$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@./real_fft_example

# 运行所有示例
run: run_basic run_real

# 测试所有程序
test: all
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@echo "$(COLOR_BOLD)测试所有示例程序$(COLOR_RESET)"
	@echo "$(COLOR_BLUE)═════════════════════════════════════════$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)测试 1/2: basic_fft$(COLOR_RESET)"
	@./basic_fft > /dev/null && echo "$(COLOR_GREEN)✓ basic_fft 测试通过$(COLOR_RESET)" || echo "$(COLOR_BOLD)✗ basic_fft 测试失败$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)测试 2/2: real_fft_example$(COLOR_RESET)"
	@./real_fft_example > /dev/null && echo "$(COLOR_GREEN)✓ real_fft_example 测试通过$(COLOR_RESET)" || echo "$(COLOR_BOLD)✗ real_fft_example 测试失败$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)所有测试完成！$(COLOR_RESET)"

# 清理
clean:
	@echo "$(COLOR_YELLOW)清理编译文件...$(COLOR_RESET)"
	rm -f $(TARGETS)
	@echo "$(COLOR_GREEN)✓ 清理完成$(COLOR_RESET)"

# 帮助信息
help:
	@echo "$(COLOR_BOLD)KISS FFT 学习指南 - 阶段 3 示例程序$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)可用目标：$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)make$(COLOR_RESET)           - 编译所有示例程序"
	@echo "  $(COLOR_GREEN)make basic_fft$(COLOR_RESET)  - 编译基础 FFT 示例"
	@echo "  $(COLOR_GREEN)make real_fft_example$(COLOR_RESET) - 编译实数 FFT 示例"
	@echo ""
	@echo "$(COLOR_BOLD)运行示例：$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)make run$(COLOR_RESET)        - 运行所有示例"
	@echo "  $(COLOR_GREEN)make run_basic$(COLOR_RESET)  - 运行基础 FFT 示例"
	@echo "  $(COLOR_GREEN)make run_real$(COLOR_RESET)   - 运行实数 FFT 示例"
	@echo ""
	@echo "$(COLOR_BOLD)其他：$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)make test$(COLOR_RESET)       - 测试所有程序（不输出结果）"
	@echo "  $(COLOR_GREEN)make clean$(COLOR_RESET)      - 清理编译文件"
	@echo "  $(COLOR_GREEN)make help$(COLOR_RESET)       - 显示此帮助信息"
	@echo ""
	@echo "$(COLOR_BOLD)示例：$(COLOR_RESET)"
	@echo "  make              # 编译所有程序"
	@echo "  make run_basic    # 编译并运行基础 FFT 示例"
	@echo "  make clean        # 清理"
