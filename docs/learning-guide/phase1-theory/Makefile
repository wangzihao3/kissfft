# Makefile for Phase 1 Learning Materials

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm
TARGETS = dft_demo

.PHONY: all clean run

all: $(TARGETS)

dft_demo: dft_demo.c
	$(CC) $(CFLAGS) -o $@ $^

run: dft_demo
	./dft_demo

clean:
	rm -f $(TARGETS)

# Python visualization (optional requirement)
check-python:
	@python3 -c "import numpy; import matplotlib" 2>/dev/null || \
		echo "Note: For visualization, install: pip install numpy matplotlib"

help:
	@echo "Available targets:"
	@echo "  all        - Build all demonstration programs"
	@echo "  dft_demo   - Build DFT demonstration"
	@echo "  run        - Run DFT demonstration"
	@echo "  clean      - Remove built files"
	@echo "  check-python - Check if Python requirements are installed"