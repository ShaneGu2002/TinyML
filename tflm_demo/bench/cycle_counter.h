#ifndef TFLM_DEMO_BENCH_CYCLE_COUNTER_H_
#define TFLM_DEMO_BENCH_CYCLE_COUNTER_H_

#include <cstdint>

// RISC-V Zicntr read-only counters. These are exposed by pk/spike.
//
// On Spike (a functional ISS) the `cycle` counter advances one per retired
// instruction, so `cycle` and `instret` end up close to equal. Report both:
// `instret` is the reliable number for scalar-vs-vector dynamic instruction
// count comparisons; `cycle` is reported for consistency with real hardware.
//
// Usage:
//   kws::Cycles c0 = kws::ReadCycles();
//   // ... region of interest ...
//   kws::Cycles c1 = kws::ReadCycles();
//   uint64_t dcy = c1.cycle - c0.cycle;
//   uint64_t din = c1.instret - c0.instret;

namespace kws {

struct Cycles {
  uint64_t cycle;
  uint64_t instret;
};

static inline uint64_t ReadCycle() {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline uint64_t ReadInstret() {
  uint64_t x;
  asm volatile("rdinstret %0" : "=r"(x));
  return x;
}

static inline Cycles ReadCycles() {
  // Read instret first, then cycle; on Spike both advance per retired
  // instruction so ordering matters only marginally.
  Cycles c;
  c.instret = ReadInstret();
  c.cycle = ReadCycle();
  return c;
}

}  // namespace kws

#endif  // TFLM_DEMO_BENCH_CYCLE_COUNTER_H_
