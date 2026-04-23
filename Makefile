# ----------------------------------------------------------------------------
# KWS TFLM demo — three-variant build for RVV research.
#
# KERNEL selects the int8 invoke path for Conv2D / DepthwiseConv2D /
# FullyConnected (Init/Prepare stay on TFLM default):
#   ref  → TFLM reference integer kernels (baseline)
#   opt  → hand-written scalar kernels, rv64gc only (software-only contribution)
#   rvv  → RVV 1.0 intrinsic kernels, rv64gcv (ISA-extension contribution)
#
# MODEL_DIR points at the directory containing model_data.{cc,h}. The default
# tracks the DS-CNN baseline; override to study the M-tier sweep winner:
#   make KERNEL=ref MODEL_DIR=artifacts/ds_cnn_M_retrained
#
# Outputs go to build/$(KERNEL)/kws_demo.elf so the three variants can coexist.
# ----------------------------------------------------------------------------

PROJECT_ROOT := $(abspath .)

KERNEL     ?= ref
MODEL_DIR  ?= artifacts/ds_cnn
# VLEN is only consumed when KERNEL=rvv (passed to spike --varch).
VLEN       ?= 128

BUILD_DIR := $(PROJECT_ROOT)/build/$(KERNEL)
TARGET    := $(BUILD_DIR)/kws_demo.elf

RISCV_PREFIX ?= riscv64-unknown-elf-
CXX := $(RISCV_PREFIX)g++
CC  := $(RISCV_PREFIX)gcc
AR  := $(RISCV_PREFIX)ar
SPIKE ?= spike
PK ?= /opt/homebrew/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk

TFLM_DIR := $(PROJECT_ROOT)/third_party/tflm
HOMEBREW_PREFIX ?= /opt/homebrew

# Per-variant arch / preprocessor flags.
ifeq ($(KERNEL),rvv)
  ARCH      := rv64gcv
  SPIKE_ISA := rv64gcv_zicntr_zihpm
  VARIANT_DEF := -DKWS_KERNEL_RVV
  # Some spike builds expose --varch=vlen:N,elen:64 to control VLEN; spike
  # 1.1.1-dev in this repo does not and uses its internal default VLEN. If you
  # upgrade to a spike that accepts --varch, set SPIKE_VARCH_OVERRIDE on the
  # make line, e.g. `make ... SPIKE_VARCH_OVERRIDE=--varch=vlen:256,elen:64`.
  SPIKE_VARCH := $(SPIKE_VARCH_OVERRIDE)
else ifeq ($(KERNEL),opt)
  ARCH      := rv64gc
  SPIKE_ISA := rv64gc_zicntr_zihpm
  VARIANT_DEF := -DKWS_KERNEL_OPT
  SPIKE_VARCH :=
else ifeq ($(KERNEL),ref)
  ARCH      := rv64gc
  SPIKE_ISA := rv64gc_zicntr_zihpm
  VARIANT_DEF := -DKWS_KERNEL_REF
  SPIKE_VARCH :=
else
  $(error Unknown KERNEL=$(KERNEL). Use one of: ref | opt | rvv)
endif

CXXFLAGS := -std=c++17 -O2 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -DTF_LITE_STATIC_MEMORY -DNDEBUG
CXXFLAGS += -march=$(ARCH) -mabi=lp64d
CXXFLAGS += $(VARIANT_DEF)
CXXFLAGS += -DKWS_MODEL_DATA_HEADER='"$(MODEL_DIR)/model_data.h"'

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
	$(PROJECT_ROOT)/tflm_demo/bench/micro_time_riscv.cc \
	$(PROJECT_ROOT)/tflm_demo/kernels/register_custom_ops.cc \
	$(PROJECT_ROOT)/tflm_demo/kernels/kws_conv_int8_$(KERNEL).cc \
	$(PROJECT_ROOT)/tflm_demo/kernels/kws_depthwise_int8_$(KERNEL).cc \
	$(PROJECT_ROOT)/tflm_demo/kernels/kws_fully_connected_int8_$(KERNEL).cc \
	$(PROJECT_ROOT)/$(MODEL_DIR)/model_data.cc

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
	$(TFLM_DIR)/tensorflow/lite/micro/micro_profiler.cc \
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

.PHONY: all clean run check-tools print-tools all-variants

all: $(TARGET)

# Build all three variants in sequence (note: one per make invocation, so
# KERNEL-specific objects go to separate build subdirs).
all-variants:
	$(MAKE) KERNEL=ref  MODEL_DIR=$(MODEL_DIR)
	$(MAKE) KERNEL=opt  MODEL_DIR=$(MODEL_DIR)
	$(MAKE) KERNEL=rvv  MODEL_DIR=$(MODEL_DIR)

print-tools:
	@echo "KERNEL=$(KERNEL)  ARCH=$(ARCH)  SPIKE_ISA=$(SPIKE_ISA)$(if $(SPIKE_VARCH), $(SPIKE_VARCH),)"
	@echo "MODEL_DIR=$(MODEL_DIR)"
	@echo "TARGET=$(TARGET)"
	@echo "CXX=$(CXX)"
	@echo "SPIKE=$(SPIKE)"
	@echo "PK=$(PK)"

check-tools:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Missing compiler: $(CXX)"; exit 1; }
	@command -v $(SPIKE) >/dev/null 2>&1 || { echo "Missing simulator: $(SPIKE)"; exit 1; }
	@command -v $(PK) >/dev/null 2>&1 || { echo "Missing proxy kernel: $(PK)"; exit 1; }

run: $(TARGET)
	$(SPIKE) --isa=$(SPIKE_ISA) $(SPIKE_VARCH) $(PK) $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@

# Force the opt and rvv kernel source files to compile without auto-
# vectorization so the observed ref → opt → rvv instret differences reflect
# *hand-written* contributions, not GCC auto-vec. `_ref.cc` is intentionally
# left on default flags so the ref baseline matches what a TFLM out-of-box
# deploy would get on the same -march.
NO_AUTOVEC_FLAGS := -fno-tree-vectorize -fno-tree-slp-vectorize
$(BUILD_DIR)/tflm_demo/kernels/kws_conv_int8_opt.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)
$(BUILD_DIR)/tflm_demo/kernels/kws_depthwise_int8_opt.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)
$(BUILD_DIR)/tflm_demo/kernels/kws_fully_connected_int8_opt.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)
$(BUILD_DIR)/tflm_demo/kernels/kws_conv_int8_rvv.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)
$(BUILD_DIR)/tflm_demo/kernels/kws_depthwise_int8_rvv.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)
$(BUILD_DIR)/tflm_demo/kernels/kws_fully_connected_int8_rvv.o: CXXFLAGS += $(NO_AUTOVEC_FLAGS)

$(BUILD_DIR)/%.o: $(PROJECT_ROOT)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/third_party/tflm/%.o: $(PROJECT_ROOT)/third_party/tflm/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(PROJECT_ROOT)/build
