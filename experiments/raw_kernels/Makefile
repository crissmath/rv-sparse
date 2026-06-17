# ==============================================================================
# 				Raw Kernel Experiment Build System
# ==============================================================================

TARGET_ARCH ?= native
BUILD_TYPE  ?= release

ROOT_DIR := ../..
INC_DIR  := $(ROOT_DIR)/include

KERNEL_DIR := kernels
TEST_DIR   := tests
OBJ_DIR    := obj
BIN_DIR    := bin

ifeq ($(TARGET_ARCH), riscv)
    CC := riscv64-linux-gnu-gcc
    ARCH_FLAGS := -march=rv64gcv -mabi=lp64d -static
    EXEC_WRAPPER := qemu-riscv64-static -cpu rv64,v=true
    $(info ---> Target Architecture: RISC-V)
else
    CC := gcc
    ARCH_FLAGS := -march=native
    EXEC_WRAPPER :=
    $(info ---> Target Architecture: NATIVE Host)
endif

CFLAGS := -Wall -Wextra -Wpedantic -std=c11 -I$(INC_DIR) -I$(KERNEL_DIR) $(ARCH_FLAGS)

ifeq ($(BUILD_TYPE), debug)
    CFLAGS += -g -O0 -DDEBUG
    $(info ---> Build Type: DEBUG)
else
    CFLAGS += -O3
    $(info ---> Build Type: RELEASE)
endif

KERNEL_SRCS := $(shell find $(KERNEL_DIR) -name '*.c')
KERNEL_OBJS := $(patsubst $(KERNEL_DIR)/%.c,$(OBJ_DIR)/kernels/%.o,$(KERNEL_SRCS))

TEST_SRCS := $(shell find $(TEST_DIR) -name '*.c')
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SRCS))

.PHONY: all dirs test clean print-config

all: dirs $(TEST_BINS)

dirs:
	@mkdir -p $(OBJ_DIR)/kernels
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR)/kernels/%.o: $(KERNEL_DIR)/%.c | dirs
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/%: $(TEST_DIR)/%.c $(KERNEL_OBJS) | dirs
	@echo "  CCLD    $@"
	@$(CC) $(CFLAGS) $< $(KERNEL_OBJS) -o $@

test: all
	@echo ""
	@echo "--- Running Raw Kernel Tests ---"
	@for test_bin in $(TEST_BINS); do \
		echo "Executing $$test_bin..."; \
		$(EXEC_WRAPPER) ./$$test_bin || exit 1; \
	done
	@echo "--- All Raw Kernel Tests Passed ---"
	@echo ""

print-config:
	@echo "TARGET_ARCH = $(TARGET_ARCH)"
	@echo "BUILD_TYPE  = $(BUILD_TYPE)"
	@echo "CC          = $(CC)"
	@echo "CFLAGS      = $(CFLAGS)"
	@echo "KERNEL_SRCS = $(KERNEL_SRCS)"
	@echo "TEST_SRCS   = $(TEST_SRCS)"

clean:
	@echo "  CLEAN   $(OBJ_DIR) $(BIN_DIR)"
	@rm -rf $(OBJ_DIR) $(BIN_DIR)