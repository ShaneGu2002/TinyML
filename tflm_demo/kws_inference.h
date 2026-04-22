#ifndef FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_
#define FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"

namespace kws {

constexpr int kFeatureFrames = 49;
constexpr int kFeatureBins = 10;
constexpr int kFeatureChannels = 1;
constexpr int kFeatureElementCount =
    kFeatureFrames * kFeatureBins * kFeatureChannels;
constexpr int kCategoryCount = 6;

extern const char* kCategoryLabels[kCategoryCount];

class KeywordSpottingRunner {
 public:
  KeywordSpottingRunner();

  TfLiteStatus Init();
  TfLiteStatus SetInput(const int8_t* input_data, int input_bytes);
  TfLiteStatus Invoke();

  const int8_t* GetOutput() const;
  int GetOutputBytes() const;
  int GetTopCategory() const;
  const char* GetTopCategoryLabel() const;

  size_t GetArenaSize() const;
  size_t GetArenaUsedBytes() const;
  int GetInputBytes() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace kws

#endif  // FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_

