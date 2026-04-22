#include "tflm_demo/kws_inference.h"

#include <cstdint>
#include <cstring>

#include "artifacts/ds_cnn/model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace kws {

const char* kCategoryLabels[kCategoryCount] = {
    "yes", "no", "stop", "go", "_unknown_", "_silence_",
};

namespace {

constexpr int kTensorArenaSize = 100 * 1024;
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
  tflite::MicroMutableOpResolver<9> resolver;
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

  // Start with the ops expected by this DS-CNN INT8 model.
  if (impl_->resolver.AddConv2D() != kTfLiteOk ||
      impl_->resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      impl_->resolver.AddFullyConnected() != kTfLiteOk ||
      impl_->resolver.AddMean() != kTfLiteOk ||
      impl_->resolver.AddReshape() != kTfLiteOk ||
      impl_->resolver.AddSoftmax() != kTfLiteOk ||
      impl_->resolver.AddQuantize() != kTfLiteOk ||
      impl_->resolver.AddDequantize() != kTfLiteOk) {
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

  if (impl_->input->type != kTfLiteInt8) {
    MicroPrintf("Expected INT8 input tensor.");
    return kTfLiteError;
  }
  if (impl_->output->type != kTfLiteInt8) {
    MicroPrintf("Expected INT8 output tensor.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus KeywordSpottingRunner::SetInput(const int8_t* input_data,
                                             int input_bytes) {
  if (impl_->input == nullptr) {
    return kTfLiteError;
  }
  if (input_bytes != impl_->input->bytes) {
    MicroPrintf("Input byte count mismatch. Expected %d, got %d.",
                impl_->input->bytes, input_bytes);
    return kTfLiteError;
  }
  std::memcpy(impl_->input->data.int8, input_data, input_bytes);
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
