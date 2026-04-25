#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/micro_log.h"
#include "tflm_demo/kws_inference.h"

namespace {

// Read RISC-V cycle counter (CSR 0xC00). On bare-metal Spike+pk this is
// always available in user mode; on hardware this needs the M-mode `mcounteren`
// to allow user-level access. If the build target stubs it out, fall back to
// `rdinstret` (CSR 0xC02) which counts retired instructions.
inline uint64_t ReadCycleCounter() {
  uint64_t cycles;
#if defined(__riscv)
  asm volatile("rdcycle %0" : "=r"(cycles));
#else
  cycles = 0;  // Native (host) builds: cycles unsupported.
#endif
  return cycles;
}

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

  // Zero-filled MFCC: a real keyword sample isn't required for cycle/latency
  // measurement (the convolutional and recurrent kernels execute the same
  // number of MACs regardless of input values). Pass `--input <file.bin>` to
  // benchmark with a real `kFeatureElementCount`-byte int8 MFCC dump.
  int8_t input_buffer[kws::kFeatureElementCount];
  std::memset(input_buffer, 0, sizeof(input_buffer));
  const int8_t* input_data = input_buffer;

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

  // Warm-up run to populate any lazy structures so the second invocation
  // measures the steady-state inference latency.
  if (runner.Invoke() != kTfLiteOk) {
    MicroPrintf("Warm-up invocation failed.");
    return 1;
  }

  const uint64_t start_cycles = ReadCycleCounter();
  const TfLiteStatus invoke_status = runner.Invoke();
  const uint64_t end_cycles = ReadCycleCounter();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Model invocation failed.");
    return 1;
  }

  const uint64_t elapsed_cycles = end_cycles - start_cycles;
  // `printf` (rather than MicroPrintf) is used so the value is emitted on
  // stdout in a stable format that benchmark scripts can grep for.
  std::printf("Inference Cycles: %" PRIu64 "\n", elapsed_cycles);

  const int output_index = runner.GetTopCategory();
  MicroPrintf("RESULT label=%s index=%d",
              runner.GetTopCategoryLabel(), output_index);
  MicroPrintf("Predicted label: %s (index=%d)",
              runner.GetTopCategoryLabel(), output_index);
  MicroPrintf("FOOTPRINT arena_used=%u arena_reserved=%u input_bytes=%d output_bytes=%d",
              static_cast<unsigned>(runner.GetArenaUsedBytes()),
              static_cast<unsigned>(runner.GetArenaSize()),
              runner.GetInputBytes(), runner.GetOutputBytes());
  MicroPrintf("LATENCY cycles=%" PRIu64, elapsed_cycles);
  return 0;
}
