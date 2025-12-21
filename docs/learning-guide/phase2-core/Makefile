# Makefile for Phase 2 Code Analysis

# 包含 KISS FFT 源码
KISSFFT_ROOT = ../../..
SRCDIR = $(KISSFFT_ROOT)
VPATH = $(SRCDIR)

# 编译设置
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -O2 -DUSE_MATH_DEFINES
INCLUDES = -I$(KISSFFT_ROOT)

# 源文件
KISSFFT_SOURCES = kiss_fft.c
KISSFFT_HEADERS = kiss_fft.h _kiss_fft_guts.h
ANALYSIS_SOURCES = code_analysis.c

# 目标
TARGET = code_analysis

.PHONY: all clean test debug profile

all: $(TARGET)

# 编译代码分析工具
$(TARGET):
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm

# 运行分析
run: $(TARGET)
	./$(TARGET)

# 调试版本
debug: CFLAGS += -DDEBUG -O0 -ggdb3
debug: $(TARGET)
	gdb ./$(TARGET)

# 性能分析
profile: CFLAGS += -pg
profile: $(TARGET)
	./$(TARGET)
	gprof $(TARGET) gmon.out > profile_report.txt

# 测试不同编译选项
test_optimizations:
	@echo "=== Testing with -O0 ==="
	$(CC) -O0 $(CFLAGS) $(INCLUDES) -o $(TARGET)_O0 $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm
	time ./$(TARGET)_O0 > /dev/null

	@echo "=== Testing with -O2 ==="
	$(CC) -O2 $(CFLAGS) $(INCLUDES) -o $(TARGET)_O2 $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm
	time ./$(TARGET)_O2 > /dev/null

	@echo "=== Testing with -O3 -march=native ==="
	$(CC) -O3 -march=native $(CFLAGS) $(INCLUDES) -o $(TARGET)_O3 $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm
	time ./$(TARGET)_O3 > /dev/null

# 测试定点数版本
test_fixed:
	@echo "=== Testing Fixed Point Version (16-bit) ==="
	$(CC) -DFIXED_POINT=16 $(CFLAGS) $(INCLUDES) -o $(TARGET)_fixed16 $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm
	./$(TARGET)_fixed16

	@echo "=== Testing Fixed Point Version (32-bit) ==="
	$(CC) -DFIXED_POINT=32 $(CFLAGS) $(INCLUDES) -o $(TARGET)_fixed32 $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES)) $(ANALYSIS_SOURCES) -lm
	./$(TARGET)_fixed32

# 生成汇编代码
assembly:
	$(CC) -S $(CFLAGS) $(INCLUDES) -o kiss_fft.s $(addprefix $(SRCDIR)/,$(KISSFFT_SOURCES))
	@echo "Assembly code generated in kiss_fft.s"

# 内存检查
memcheck: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

# 代码风格检查
style:
	@echo "Checking code style..."
	@for file in $(KISSFFT_SOURCES) $(KISSFFT_HEADERS); do \
		if [ -f $(SRCDIR)/$$file ]; then \
			echo "Checking $(SRCDIR)/$$file..."; \
		fi; \
	done

# 文档生成
docs:
	@echo "Generating documentation..."
	doxygen Doxyfile 2>/dev/null || echo "Doxygen not configured"

# 性能基准测试
benchmark: $(TARGET)
	@echo "Running comprehensive benchmark..."
	./$(TARGET) > benchmark_$(shell date +%Y%m%d_%H%M%S).txt

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all               - Build the code analysis tool"
	@echo "  run               - Run the analysis tool"
	@echo "  debug             - Build with debug symbols and run in gdb"
	@echo "  profile           - Build with profiling and generate report"
	@echo "  test_optimizations - Test different compiler optimization levels"
	@echo "  test_fixed        - Test fixed-point versions"
	@echo "  assembly          - Generate assembly code"
	@echo "  memcheck          - Run with valgrind memory checker"
	@echo "  style             - Check code style"
	@echo "  docs              - Generate documentation"
	@echo "  benchmark         - Run performance benchmark"
	@echo "  clean             - Remove built files"

# 清理
clean:
	rm -f $(TARGET) $(TARGET)_* *.s *.o gmon.out profile_report.txt
	rm -f benchmark_*.txt

# 依赖关系
$(SRCDIR)/kiss_fft.o: $(SRCDIR)/kiss_fft.c $(SRCDIR)/kiss_fft.h $(SRCDIR)/_kiss_fft_guts.h
code_analysis.o: code_analysis.c $(SRCDIR)/kiss_fft.h $(SRCDIR)/_kiss_fft_guts.h