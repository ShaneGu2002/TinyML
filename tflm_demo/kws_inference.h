#ifndef FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_
#define FINAL_PROJECT_TFLM_DEMO_KWS_INFERENCE_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"

// `kFeatureFrames`, `kFeatureBins`, `kFeatureChannels`,
// `kFeatureElementCount`, `kCategoryCount`, and `kCategoryLabels` are all
// supplied by the generated `model_data.h` (one per artifact directory).
#include "model_data.h"

namespace kws {

class KeywordSpottingRunner {
 public:
  KeywordSpottingRunner();

  TfLiteStatus Init();
  TfLiteStatus SetInput(const int8_t* input_data, int input_element_count);
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
