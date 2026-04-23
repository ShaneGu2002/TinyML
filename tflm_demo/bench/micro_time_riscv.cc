// RISC-V backend for TFLM's MicroProfiler time source.
//
// TFLM's default micro_time.cc provides weak implementations of
// ticks_per_second() and GetCurrentTimeTicks(). We replace them here so the
// per-op MicroProfiler reports dynamic instruction count via `rdcycle` (on
// Spike, one cycle per retired instruction, so this is equivalent to
// `rdinstret`). This file is only compiled into the kws demo builds; the
// default micro_time.cc is intentionally left out of the Makefile to avoid a
// duplicate-symbol link error.

#include "tensorflow/lite/micro/micro_time.h"
#include "tflm_demo/bench/cycle_counter.h"

namespace tflite {

uint32_t ticks_per_second() {
  // We can't map ticks to wall-clock seconds on Spike, so report 0
  // (TFLM treats 0 as "unknown") and let downstream code consume ticks as
  // a raw instruction count.
  return 0;
}

uint32_t GetCurrentTimeTicks() {
  return static_cast<uint32_t>(kws::ReadCycle());
}

}  // namespace tflite
