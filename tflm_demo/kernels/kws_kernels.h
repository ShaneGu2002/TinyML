#ifndef TFLM_DEMO_KERNELS_KWS_KERNELS_H_
#define TFLM_DEMO_KERNELS_KWS_KERNELS_H_

#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace kws_kernels {

// Variant-specific int8 invoke kernels. Each build configuration
// (KERNEL=ref|opt|rvv) links exactly one definition per function.
//
// Signatures match reference_integer_ops::{Conv,DepthwiseConv,FullyConnected}
// so ref_variant.cc can forward directly and opt/rvv variants remain drop-in
// compatible for bitwise-equivalence testing.

void KwsConvPerChannelInt8(
    const tflite::ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data);

void KwsDepthwiseConvPerChannelInt8(
    const tflite::DepthwiseParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data);

void KwsFullyConnectedInt8(
    const tflite::FullyConnectedParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data);

// Registration wrappers. These inherit init/prepare/free/reset from TFLM's
// default kernels (so per-channel multiplier/shift setup still happens) and
// override only invoke to point at the variant kernels above.
TFLMRegistration GetConvInt8Registration();
TFLMRegistration GetDepthwiseConvInt8Registration();
TFLMRegistration GetFullyConnectedInt8Registration();

// Human-readable build variant string ("ref" | "opt" | "rvv"), for logging.
inline const char* KernelVariantName() {
#if defined(KWS_KERNEL_RVV)
  return "rvv";
#elif defined(KWS_KERNEL_OPT)
  return "opt";
#else
  return "ref";
#endif
}

}  // namespace kws_kernels

#endif  // TFLM_DEMO_KERNELS_KWS_KERNELS_H_
