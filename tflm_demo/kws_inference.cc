#include "tflm_demo/kws_inference.h"

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace kws {

namespace {

// The unrolled GRU has ~100 FULLY_CONNECTED ops, each per-channel quantized
// (192 multipliers/shifts -> ~1.5 KB persistent each); plus per-timestep state
// tensors. 1 MiB is generous but the spike+pk environment has GBs of DRAM, so
// we trade RAM for safety here.
constexpr int kTensorArenaSize = 1024 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

int ArgMax(const int8_t* values, int count) {
  int best_index = 0;
  int8_t best_value = values[0];
  for (int i = 1; i < count; ++i) {
    if (values[i] > best_value) {
      best_value = values[i];
      best_index = i;
    }
  }
  return best_index;
}

}  // namespace

struct KeywordSpottingRunner::Impl {
  const tflite::Model* model = nullptr;
  // Superset of operators required by both the DS-CNN INT8 model
  // (CONV_2D / DEPTHWISE_CONV_2D / MEAN / …) and the unrolled GRU INT8 model
  // (FULLY_CONNECTED / LOGISTIC / TANH / SPLIT / STRIDED_SLICE / SUB / …).
  tflite::MicroMutableOpResolver<16> resolver;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
};

KeywordSpottingRunner::KeywordSpottingRunner() : impl_(new Impl()) {}

TfLiteStatus KeywordSpottingRunner::Init() {
  impl_->model = tflite::GetModel(model_int8_tflite);
  if (impl_->model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d does not match runtime schema %d.",
                impl_->model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  if (impl_->resolver.AddConv2D() != kTfLiteOk ||
      impl_->resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      impl_->resolver.AddFullyConnected() != kTfLiteOk ||
      impl_->resolver.AddMean() != kTfLiteOk ||
      impl_->resolver.AddReshape() != kTfLiteOk ||
      impl_->resolver.AddSoftmax() != kTfLiteOk ||
      impl_->resolver.AddQuantize() != kTfLiteOk ||
      impl_->resolver.AddDequantize() != kTfLiteOk ||
      impl_->resolver.AddAdd() != kTfLiteOk ||
      impl_->resolver.AddMul() != kTfLiteOk ||
      impl_->resolver.AddSub() != kTfLiteOk ||
      impl_->resolver.AddLogistic() != kTfLiteOk ||
      impl_->resolver.AddTanh() != kTfLiteOk ||
      impl_->resolver.AddSplit() != kTfLiteOk ||
      impl_->resolver.AddStridedSlice() != kTfLiteOk) {
    MicroPrintf("Failed to register one or more operators.");
    return kTfLiteError;
  }

  static tflite::MicroInterpreter static_interpreter(
      impl_->model, impl_->resolver, g_tensor_arena, kTensorArenaSize);
  impl_->interpreter = &static_interpreter;

  if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed. Try increasing tensor arena size.");
    return kTfLiteError;
  }

  MicroPrintf("ARENA used=%u / reserved=%u bytes",
              static_cast<unsigned>(impl_->interpreter->arena_used_bytes()),
              static_cast<unsigned>(kTensorArenaSize));

  impl_->input = impl_->interpreter->input(0);
  impl_->output = impl_->interpreter->output(0);
  return kTfLiteOk;
}

TfLiteStatus KeywordSpottingRunner::SetInput(const int8_t* input_data,
                                             int input_element_count) {
  if (impl_->input == nullptr) {
    return kTfLiteError;
  }
  // The DS-CNN model takes int8 input; the GRU model keeps a float32 input
  // (the converter inserts an explicit QUANTIZE op). Handle both, but the
  // public ABI is always "int8 MFCC samples, kFeatureElementCount of them".
  if (impl_->input->type == kTfLiteInt8) {
    if (input_element_count != impl_->input->bytes) {
      MicroPrintf("Input element count mismatch. Expected %d, got %d.",
                  impl_->input->bytes, input_element_count);
      return kTfLiteError;
    }
    std::memcpy(impl_->input->data.int8, input_data, input_element_count);
  } else if (impl_->input->type == kTfLiteFloat32) {
    const int expected_elements =
        static_cast<int>(impl_->input->bytes / sizeof(float));
    if (input_element_count != expected_elements) {
      MicroPrintf("Input element count mismatch. Expected %d, got %d.",
                  expected_elements, input_element_count);
      return kTfLiteError;
    }
    float* dst = impl_->input->data.f;
    for (int i = 0; i < input_element_count; ++i) {
      dst[i] = static_cast<float>(input_data[i]) / 128.0f;
    }
  } else {
    MicroPrintf("Unsupported input tensor type: %d", impl_->input->type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus KeywordSpottingRunner::Invoke() {
  if (impl_->interpreter == nullptr) {
    return kTfLiteError;
  }
  return impl_->interpreter->Invoke();
}

const int8_t* KeywordSpottingRunner::GetOutput() const {
  return impl_->output == nullptr ? nullptr : impl_->output->data.int8;
}

int KeywordSpottingRunner::GetOutputBytes() const {
  return impl_->output == nullptr ? 0 : impl_->output->bytes;
}

int KeywordSpottingRunner::GetTopCategory() const {
  if (impl_->output == nullptr) return 0;
  if (impl_->output->type == kTfLiteFloat32) {
    const float* values = impl_->output->data.f;
    int best = 0;
    for (int i = 1; i < kCategoryCount; ++i) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }
  return ArgMax(GetOutput(), kCategoryCount);
}

const char* KeywordSpottingRunner::GetTopCategoryLabel() const {
  return kCategoryLabels[GetTopCategory()];
}

size_t KeywordSpottingRunner::GetArenaSize() const {
  return static_cast<size_t>(kTensorArenaSize);
}

size_t KeywordSpottingRunner::GetArenaUsedBytes() const {
  return impl_->interpreter == nullptr
             ? 0
             : impl_->interpreter->arena_used_bytes();
}

int KeywordSpottingRunner::GetInputBytes() const {
  return impl_->input == nullptr ? 0 : impl_->input->bytes;
}

}  // namespace kws
