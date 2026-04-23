// KERNEL=opt : hand-written scalar int8 per-channel depthwise conv.
//
// DS-CNN M uses a 5x3 depthwise with same-padding, stride 1, depth
// multiplier 1. In the reference kernel every one of the 15 filter taps does
// four bounds-check comparisons (in_x >= 0, in_x < W, in_y >= 0, in_y < H)
// even though whether a tap is in-bounds depends only on (oh, ow), not on the
// tap. We hoist those bounds once per output position and inline the
// filter loop with the computed [fh_start, fh_end) / [fw_start, fw_end)
// range, killing 15x bounds checks and unlocking better register allocation.
//
// Depth multiplier != 1, dilation != 1, or other oddities fall back to the
// TFLM reference.

#include "tflm_demo/kernels/kws_kernels.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

namespace kws_kernels {
namespace {

inline int8_t RequantizeAndClamp(int32_t acc, int32_t mult, int shift,
                                 int32_t output_offset,
                                 int32_t out_min, int32_t out_max) {
  acc = tflite::MultiplyByQuantizedMultiplier(acc, mult, shift);
  acc += output_offset;
  acc = std::max(acc, out_min);
  acc = std::min(acc, out_max);
  return static_cast<int8_t>(acc);
}

bool IsSupportedSpecialization(const tflite::DepthwiseParams& params) {
  return params.depth_multiplier == 1 &&
         params.dilation_width_factor == 1 &&
         params.dilation_height_factor == 1;
}

void DepthwiseConvInt8Hoisted(
    const tflite::DepthwiseParams& params,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& /*bias_shape*/, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int pad_h = params.padding_values.height;
  const int pad_w = params.padding_values.width;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t out_min = params.quantized_activation_min;
  const int32_t out_max = params.quantized_activation_max;

  const int batches = input_shape.Dims(0);
  const int in_h = input_shape.Dims(1);
  const int in_w = input_shape.Dims(2);
  const int channels = input_shape.Dims(3);  // == output_depth (multiplier==1)
  const int filter_h = filter_shape.Dims(1);
  const int filter_w = filter_shape.Dims(2);
  const int out_h = output_shape.Dims(1);
  const int out_w = output_shape.Dims(2);

  // Strides (in elements) for NHWC tensors.
  const int in_row_stride = in_w * channels;
  const int filter_row_stride = filter_w * channels;

  for (int b = 0; b < batches; ++b) {
    const int8_t* in_batch = input_data + b * in_h * in_row_stride;
    int8_t* out_batch = output_data + b * out_h * out_w * channels;
    for (int oh = 0; oh < out_h; ++oh) {
      const int ih_origin = oh * stride_h - pad_h;
      const int fh_start = std::max(0, -ih_origin);
      const int fh_end = std::min(filter_h, in_h - ih_origin);

      for (int ow = 0; ow < out_w; ++ow) {
        const int iw_origin = ow * stride_w - pad_w;
        const int fw_start = std::max(0, -iw_origin);
        const int fw_end = std::min(filter_w, in_w - iw_origin);

        int8_t* out_row = out_batch + (oh * out_w + ow) * channels;

        for (int c = 0; c < channels; ++c) {
          int32_t acc = bias_data ? bias_data[c] : 0;

          for (int fh = fh_start; fh < fh_end; ++fh) {
            const int ih = ih_origin + fh;
            const int8_t* in_row = in_batch + ih * in_row_stride;
            const int8_t* f_row = filter_data + fh * filter_row_stride;

            for (int fw = fw_start; fw < fw_end; ++fw) {
              const int iw = iw_origin + fw;
              const int32_t x =
                  static_cast<int32_t>(in_row[iw * channels + c]) + input_offset;
              acc += x * static_cast<int32_t>(f_row[fw * channels + c]);
            }
          }

          out_row[c] = RequantizeAndClamp(acc, output_multiplier[c],
                                          output_shift[c], output_offset,
                                          out_min, out_max);
        }
      }
    }
  }
}

}  // namespace

void KwsDepthwiseConvPerChannelInt8(
    const tflite::DepthwiseParams& params,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  if (IsSupportedSpecialization(params)) {
    DepthwiseConvInt8Hoisted(params, output_multiplier, output_shift,
                             input_shape, input_data,
                             filter_shape, filter_data,
                             bias_shape, bias_data,
                             output_shape, output_data);
    return;
  }
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier, output_shift,
      input_shape, input_data,
      filter_shape, filter_data,
      bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace kws_kernels
