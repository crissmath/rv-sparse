# ==============================================================================
# rv-sparse Build System
# ==============================================================================

#Note : If you want to test separate part of the back-end dont edit this makefile 
# this makefile is only for test the entire API

# ------------------------------------------------------------------------------
# Build Configuration
# ------------------------------------------------------------------------------
# TARGET_ARCH can be set to 'native' (default) or 'riscv' via command line:
# e.g., make TARGET_ARCH=riscv
TARGET_ARCH ?= native

# Build type (debug or release)
BUILD_TYPE ?= release

# ------------------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------------------
SRC_DIR     := src
INC_DIR     := include
OBJ_DIR     := obj
BIN_DIR     := bin
TEST_DIR    := tests
LIB_DIR     := lib

# ------------------------------------------------------------------------------
# Toolchain & Architecture Flags Selection
# ------------------------------------------------------------------------------
ifeq ($(TARGET_ARCH), riscv)
    # RISC-V Cross-Compilation Toolchain (Linux GNU version)
    CC        := riscv64-linux-gnu-gcc
    AR        := riscv64-linux-gnu-ar
    
    # Architecture: 64-bit, General (IMAFD), Compressed (C), Vector (V)
    # IMPORTANTE: Se añade -static para que QEMU pueda ejecutarlo sin librerías externas
    ARCH_FLAGS := -march=rv64gcv -mabi=lp64d -static
    
    # Execution Wrapper for Tests (QEMU with Vector support enabled)
    EXEC_WRAPPER := qemu-riscv64-static -cpu rv64,v=true
else
    # Native Compilation Toolchain (x86_64 or local host)
    CC        := gcc
    AR        := ar
    
    # Native architecture optimizations
    ARCH_FLAGS := -march=native
    
    # No wrapper needed for native execution
    EXEC_WRAPPER := 
    
    $(info ---> Target Architecture: NATIVE Host)
endif

# ------------------------------------------------------------------------------
# Compiler Flags
# ------------------------------------------------------------------------------
# Base flags
CFLAGS := -Wall -Wextra -Wpedantic -std=c99 -I$(INC_DIR) $(ARCH_FLAGS)

# Build type flags
ifeq ($(BUILD_TYPE), debug)
    CFLAGS += -g -O0 -DDEBUG
    $(info ---> Build Type: DEBUG)
else
    CFLAGS += -O3 -flto
    $(info ---> Build Type: RELEASE)
endif

# ------------------------------------------------------------------------------
# Source Files & Object Mapping
# ------------------------------------------------------------------------------
# Find all C files in the src directory and its subdirectories (core, formats, kernels)
SRCS := $(shell find $(SRC_DIR) -name '*.c')

# Map source files to object files in the build/obj directory
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Static library output
STATIC_LIB := $(LIB_DIR)/librvsparse.a

# Test sources and binaries
TEST_SRCS := $(shell find $(TEST_DIR) -name '*.c')
TEST_BINS := $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/%, $(TEST_SRCS))

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------
.PHONY: all clean test dirs

# Default target
all: dirs $(STATIC_LIB) $(TEST_BINS)

# Create necessary directories
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)
	@mkdir -p $(dir $(OBJS)) # Ensure nested obj directories exist

# Build the static library
$(STATIC_LIB): $(OBJS)
	@echo "  AR      $@"
	@$(AR) rcs $@ $^

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Build test executables (linking against the static library)
$(BIN_DIR)/%: $(TEST_DIR)/%.c $(STATIC_LIB)
	@echo "  CCLD    $@"
	@$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lrvsparse -o $@

# Run all tests
test: all
	@echo "\n--- Running Tests ---"
	@for test_bin in $(TEST_BINS); do \
		echo "Executing $$test_bin..."; \
		$(EXEC_WRAPPER) ./$$test_bin || exit 1; \
	done
	@echo "--- All Tests Passed ---\n"

# Clean build artifacts
clean:
	@echo "  CLEAN   $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)"
	@rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR)