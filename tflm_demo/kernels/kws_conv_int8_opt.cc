// KERNEL=opt : hand-written scalar int8 per-channel convolution.
//
// For 1x1 pointwise convs (stride 1, no padding, no dilation) we use a
// specialized micro-kernel that unrolls the output-channel loop by 4 so each
// input[ic] load is shared across four weight multiplies and four int32
// accumulators. This is the single biggest hotspot in DS-CNN M — five
// pointwise convs account for ~52% of the reference invoke instret.
//
// Other geometries (first-layer 5x3 stride (1,2) conv, etc.) fall back to the
// TFLM reference implementation. Those fallbacks cover correctness; the
// optimized scalar depthwise + first-layer kernels live in their own files.

#include "tflm_demo/kernels/kws_kernels.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

namespace kws_kernels {
namespace {

// True when the conv is a plain pointwise 1x1 with unit stride and no padding.
// In that case the operation reduces to a matrix multiply over the flattened
// N*H*W spatial plane, which the micro-kernel below handles.
bool IsPointwise1x1(const tflite::ConvParams& params,
                    const tflite::RuntimeShape& filter_shape) {
  const int fh = filter_shape.Dims(1);
  const int fw = filter_shape.Dims(2);
  return fh == 1 && fw == 1 &&
         params.stride_width == 1 && params.stride_height == 1 &&
         params.dilation_width_factor == 1 &&
         params.dilation_height_factor == 1 &&
         params.padding_values.width == 0 && params.padding_values.height == 0;
}

inline int8_t RequantizeAndClamp(int32_t acc, int32_t mult, int shift,
                                 int32_t output_offset,
                                 int32_t out_min, int32_t out_max) {
  acc = tflite::MultiplyByQuantizedMultiplier(acc, mult, shift);
  acc += output_offset;
  acc = std::max(acc, out_min);
  acc = std::min(acc, out_max);
  return static_cast<int8_t>(acc);
}

// Specialized int8 pointwise 1x1 per-channel conv.
// Treat N*H*W as a flat spatial axis S, then for each s:
//   for oc in [0, OC): acc[oc] = bias[oc] + sum_{ic} (input[s,ic]+off) * W[oc,ic]
// With OC unrolled by 4 the inner loop does 4 MACs per input load.
void ConvInt8Pointwise1x1(
    const tflite::ConvParams& params,
    const int32_t* per_channel_multiplier, const int32_t* per_channel_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& /*bias_shape*/, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t out_min = params.quantized_activation_min;
  const int32_t out_max = params.quantized_activation_max;

  const int batches = input_shape.Dims(0);
  const int in_height = input_shape.Dims(1);
  const int in_width = input_shape.Dims(2);
  const int in_channels = input_shape.Dims(3);
  const int out_channels = filter_shape.Dims(0);
  const int spatial = batches * in_height * in_width;

  for (int s = 0; s < spatial; ++s) {
    const int8_t* in_row = input_data + s * in_channels;
    int8_t* out_row = output_data + s * out_channels;

    int oc = 0;
    for (; oc + 4 <= out_channels; oc += 4) {
      const int8_t* w0 = filter_data + (oc + 0) * in_channels;
      const int8_t* w1 = filter_data + (oc + 1) * in_channels;
      const int8_t* w2 = filter_data + (oc + 2) * in_channels;
      const int8_t* w3 = filter_data + (oc + 3) * in_channels;
      int32_t acc0 = bias_data ? bias_data[oc + 0] : 0;
      int32_t acc1 = bias_data ? bias_data[oc + 1] : 0;
      int32_t acc2 = bias_data ? bias_data[oc + 2] : 0;
      int32_t acc3 = bias_data ? bias_data[oc + 3] : 0;

      for (int ic = 0; ic < in_channels; ++ic) {
        const int32_t x = static_cast<int32_t>(in_row[ic]) + input_offset;
        acc0 += x * static_cast<int32_t>(w0[ic]);
        acc1 += x * static_cast<int32_t>(w1[ic]);
        acc2 += x * static_cast<int32_t>(w2[ic]);
        acc3 += x * static_cast<int32_t>(w3[ic]);
      }

      out_row[oc + 0] = RequantizeAndClamp(
          acc0, per_channel_multiplier[oc + 0], per_channel_shift[oc + 0],
          output_offset, out_min, out_max);
      out_row[oc + 1] = RequantizeAndClamp(
          acc1, per_channel_multiplier[oc + 1], per_channel_shift[oc + 1],
          output_offset, out_min, out_max);
      out_row[oc + 2] = RequantizeAndClamp(
          acc2, per_channel_multiplier[oc + 2], per_channel_shift[oc + 2],
          output_offset, out_min, out_max);
      out_row[oc + 3] = RequantizeAndClamp(
          acc3, per_channel_multiplier[oc + 3], per_channel_shift[oc + 3],
          output_offset, out_min, out_max);
    }

    for (; oc < out_channels; ++oc) {
      const int8_t* w = filter_data + oc * in_channels;
      int32_t acc = bias_data ? bias_data[oc] : 0;
      for (int ic = 0; ic < in_channels; ++ic) {
        const int32_t x = static_cast<int32_t>(in_row[ic]) + input_offset;
        acc += x * static_cast<int32_t>(w[ic]);
      }
      out_row[oc] = RequantizeAndClamp(acc, per_channel_multiplier[oc],
                                       per_channel_shift[oc], output_offset,
                                       out_min, out_max);
    }
  }
}

}  // namespace

void KwsConvPerChannelInt8(
    const tflite::ConvParams& params,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  if (IsPointwise1x1(params, filter_shape)) {
    ConvInt8Pointwise1x1(params, output_multiplier, output_shift,
                         input_shape, input_data,
                         filter_shape, filter_data,
                         bias_shape, bias_data,
                         output_shape, output_data);
    return;
  }
  tflite::reference_integer_ops::ConvPerChannel(
      params, output_multiplier, output_shift,
      input_shape, input_data,
      filter_shape, filter_data,
      bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace kws_kernels
