# Makefile for Pure CUDA Transformer Engine
NVCC ?= nvcc
CXX ?= g++

# Target architectures: sm_60 (P100), sm_70 (V100), sm_75 (T4), sm_80 (A100), sm_86 (RTX 30-series), sm_89 (L4/RTX 40), sm_90 (H100)
ARCH_FLAGS ?= -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89

NVCC_FLAGS = -O3 -std=c++17 -Iinclude $(ARCH_FLAGS) --use_fast_math -Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra,-fopenmp
LDFLAGS = -lcublas -lcublasLt -lcudart -lm

BUILD_DIR = build
BIN_DIR = bin

CORE_SRCS = src/model.cu \
            src/optimizer.cu \
            src/dataloader.cpp \
            src/kernels/embedding.cu \
            src/kernels/layernorm.cu \
            src/kernels/matmul.cu \
            src/kernels/attention.cu \
            src/kernels/ffn.cu \
            src/kernels/residual.cu \
            src/kernels/loss.cu

all: dirs $(BIN_DIR)/train $(BIN_DIR)/benchmark $(BIN_DIR)/parity_audit $(BIN_DIR)/test_kernels

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

$(BIN_DIR)/train: src/main.cu $(CORE_SRCS) | dirs
	$(NVCC) $(NVCC_FLAGS) src/main.cu $(CORE_SRCS) -o $@ $(LDFLAGS)

$(BIN_DIR)/benchmark: src/benchmark.cu $(CORE_SRCS) | dirs
	$(NVCC) $(NVCC_FLAGS) src/benchmark.cu $(CORE_SRCS) -o $@ $(LDFLAGS)

$(BIN_DIR)/parity_audit: src/parity_audit.cu $(CORE_SRCS) | dirs
	$(NVCC) $(NVCC_FLAGS) src/parity_audit.cu $(CORE_SRCS) -o $@ $(LDFLAGS)

$(BIN_DIR)/test_kernels: tests/test_kernels.cu $(CORE_SRCS) | dirs
	$(NVCC) $(NVCC_FLAGS) tests/test_kernels.cu $(CORE_SRCS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) tests/parity_data checkpoint.bin

.PHONY: all dirs clean
