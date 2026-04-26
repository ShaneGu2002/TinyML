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
// Must match the output tensor size of the deployed model: the DS-CNN M
// retrained from the sweep uses the full 10-keyword set plus _unknown_ /
// _silence_.
constexpr int kCategoryCount = 12;

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

  // Clears per-op profiler events so each Invoke measurement starts fresh.
  void ResetProfiler();

  // Emits per-op profiler breakdown in CSV form over MicroPrintf. Paired with
  // the BENCH line that main.cc emits around Invoke, this gives the full
  // per-inference instruction profile for the ref / opt / rvv comparison.
  void DumpProfilerCsv() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace kws

#endif  // FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_

