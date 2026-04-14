PROJECT_ROOT := $(abspath .)
BUILD_DIR := $(PROJECT_ROOT)/build
TARGET := $(BUILD_DIR)/kws_demo.elf

RISCV_PREFIX ?= riscv64-unknown-elf-
CXX := $(RISCV_PREFIX)g++
CC := $(RISCV_PREFIX)gcc
AR := $(RISCV_PREFIX)ar
SPIKE ?= spike
PK ?= /opt/homebrew/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk

TFLM_DIR := $(PROJECT_ROOT)/third_party/tflm
HOMEBREW_PREFIX ?= /opt/homebrew

CXXFLAGS := -std=c++17 -O2 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -DTF_LITE_STATIC_MEMORY -DNDEBUG
CXXFLAGS += -march=rv64gc -mabi=lp64d

LDFLAGS := -Wl,--gc-sections

INCLUDES := \
	-I$(PROJECT_ROOT) \
	-I$(TFLM_DIR) \
	-I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I$(HOMEBREW_PREFIX)/include

APP_SRCS := \
	$(PROJECT_ROOT)/tflm_demo/main.cc \
	$(PROJECT_ROOT)/tflm_demo/kws_inference.cc \
	$(PROJECT_ROOT)/tflm_demo/test_input_data.cc \
	$(PROJECT_ROOT)/artifacts/ds_cnn/model_data.cc

TFLM_MICRO_SRCS := \
	$(TFLM_DIR)/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc \
	$(TFLM_DIR)/tensorflow/compiler/mlir/lite/schema/schema_utils.cc \
	$(TFLM_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
	$(TFLM_DIR)/tensorflow/lite/core/c/common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/debug_log.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/flatbuffer_utils.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/memory_helpers.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_allocation_info.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_allocator.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_context.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter_context.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter_graph.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_log.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_op_resolver.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_resource_variable.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/micro_utils.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/system_setup.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc

TFLM_KERNEL_SRCS := \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/activations_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/conv.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/conv_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/depthwise_conv.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/fully_connected.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/fully_connected_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/kernel_runner.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/kernel_util.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/reduce.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/reduce_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/reshape.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/reshape_common.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax_common.cc

TFLM_INTERNAL_SRCS := \
	$(TFLM_DIR)/tensorflow/lite/kernels/kernel_util.cc \
	$(TFLM_DIR)/tensorflow/lite/kernels/internal/common.cc \
	$(TFLM_DIR)/tensorflow/lite/kernels/internal/portable_tensor_utils.cc \
	$(TFLM_DIR)/tensorflow/lite/kernels/internal/quantization_util.cc \
	$(TFLM_DIR)/tensorflow/lite/kernels/internal/tensor_utils.cc

SRCS := $(APP_SRCS) $(TFLM_MICRO_SRCS) $(TFLM_KERNEL_SRCS) $(TFLM_INTERNAL_SRCS)
OBJS := $(patsubst $(PROJECT_ROOT)/%.cc,$(BUILD_DIR)/%.o,$(filter %.cc,$(SRCS)))

.PHONY: all clean run check-tools print-tools

all: $(TARGET)

print-tools:
	@echo "CXX=$(CXX)"
	@echo "SPIKE=$(SPIKE)"
	@echo "PK=$(PK)"

check-tools:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Missing compiler: $(CXX)"; exit 1; }
	@command -v $(SPIKE) >/dev/null 2>&1 || { echo "Missing simulator: $(SPIKE)"; exit 1; }
	@command -v $(PK) >/dev/null 2>&1 || { echo "Missing proxy kernel: $(PK)"; exit 1; }

run: $(TARGET)
	$(SPIKE) $(PK) $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(PROJECT_ROOT)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/third_party/tflm/%.o: $(PROJECT_ROOT)/third_party/tflm/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
