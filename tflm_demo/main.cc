#include <cstdint>
#include <cstdio>

#include "tensorflow/lite/micro/micro_log.h"
#include "tflm_demo/kws_inference.h"
#include "tflm_demo/test_input_data.h"

namespace {

bool LoadInputFromFile(const char* path, int8_t* buffer, int expected_bytes) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
    MicroPrintf("Failed to open input file: %s", path);
    return false;
  }

  const size_t bytes_read = std::fread(buffer, 1, expected_bytes, file);
  const int extra_byte = std::fgetc(file);
  std::fclose(file);

  if (bytes_read != static_cast<size_t>(expected_bytes) || extra_byte != EOF) {
    MicroPrintf("Input file must contain exactly %d bytes, got %d.", expected_bytes,
                static_cast<int>(bytes_read));
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  kws::KeywordSpottingRunner runner;
  if (runner.Init() != kTfLiteOk) {
    MicroPrintf("KWS runner initialization failed.");
    return 1;
  }

  int8_t input_buffer[kws::kFeatureElementCount];
  const int8_t* input_data = g_example_mfcc;

  if (argc >= 2) {
    if (!LoadInputFromFile(argv[1], input_buffer, kws::kFeatureElementCount)) {
      return 2;
    }
    input_data = input_buffer;
  }

  if (runner.SetInput(input_data, kws::kFeatureElementCount) != kTfLiteOk) {
    MicroPrintf("Failed to set input tensor.");
    return 1;
  }

  if (runner.Invoke() != kTfLiteOk) {
    MicroPrintf("Model invocation failed.");
    return 1;
  }

  const int output_index = runner.GetTopCategory();
  const int8_t* output = runner.GetOutput();
  MicroPrintf("RESULT label=%s index=%d score=%d", runner.GetTopCategoryLabel(),
              output_index, output[output_index]);
  MicroPrintf("Predicted label: %s (index=%d, score=%d)",
              runner.GetTopCategoryLabel(), output_index, output[output_index]);
  MicroPrintf("FOOTPRINT arena_used=%u arena_reserved=%u input_bytes=%d output_bytes=%d",
              static_cast<unsigned>(runner.GetArenaUsedBytes()),
              static_cast<unsigned>(runner.GetArenaSize()),
              runner.GetInputBytes(), runner.GetOutputBytes());
  return 0;
}
