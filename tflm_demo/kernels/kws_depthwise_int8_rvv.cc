// KERNEL=rvv : RVV 1.0 intrinsic int8 per-channel depthwise conv.
//
// For depth_multiplier==1, dilation==1 cases (DS-CNN M uses 5x3 with
// stride 1, depth_multiplier 1) we vectorize across the channel axis. Each
// output position (oh, ow) processes its 76 channels in chunks of vl lanes;
// within each chunk the 5x3 filter taps widen int8→int16→int32 and accumulate
// via vwmacc. Bounds are hoisted to (oh, ow) so the inner filter loop has no
// per-tap padding branch.
//
// Requantization (per-channel multiplier/shift) stays scalar because the
// shift is a separate value per lane; RVV 1.0 has no single op that does
// per-lane rounding-divide-by-POT with arbitrary shifts.
//
// Everything else falls back to the TFLM reference.

#include "tflm_demo/kernels/kws_kernels.h"

#include <algorithm>
#include <cstdint>
#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

namespace kws_kernels {
namespace {

bool IsSupportedSpecialization(const tflite::DepthwiseParams& params) {
  return params.depth_multiplier == 1 &&
         params.dilation_width_factor == 1 &&
         params.dilation_height_factor == 1;
}

void DepthwiseConvInt8RVV(
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
  const int channels = input_shape.Dims(3);
  const int filter_h = filter_shape.Dims(1);
  const int filter_w = filter_shape.Dims(2);
  const int out_h = output_shape.Dims(1);
  const int out_w = output_shape.Dims(2);

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

        for (int c = 0; c < channels;) {
          // Process up to vl lanes of channels at once. SEW/LMUL ratio is
          // kept consistent across the e8m1 loads and e32m4 accumulator so
          // a single vsetvl_e32m4 governs both.
          const size_t vl =
              __riscv_vsetvl_e32m4(static_cast<size_t>(channels - c));

          // Load bias as initial int32 accumulator for this channel chunk.
          vint32m4_t vacc = bias_data
                                ? __riscv_vle32_v_i32m4(bias_data + c, vl)
                                : __riscv_vmv_v_x_i32m4(0, vl);

          for (int fh = fh_start; fh < fh_end; ++fh) {
            const int ih = ih_origin + fh;
            const int8_t* in_row = in_batch + ih * in_row_stride;
            const int8_t* f_row = filter_data + fh * filter_row_stride;
            for (int fw = fw_start; fw < fw_end; ++fw) {
              const int iw = iw_origin + fw;
              const int8_t* in_ptr = in_row + iw * channels + c;
              const int8_t* f_ptr = f_row + fw * channels + c;

              vint8m1_t vin8 = __riscv_vle8_v_i8m1(in_ptr, vl);
              vint8m1_t vf8 = __riscv_vle8_v_i8m1(f_ptr, vl);
              vint16m2_t vin16 = __riscv_vsext_vf2_i16m2(vin8, vl);
              vin16 = __riscv_vadd_vx_i16m2(vin16, input_offset, vl);
              vint16m2_t vf16 = __riscv_vsext_vf2_i16m2(vf8, vl);
              vacc = __riscv_vwmacc_vv_i32m4(vacc, vin16, vf16, vl);
            }
          }

          // Scalar requantize per lane. Spill vacc to a scratch buffer of
          // size >= vlmax_e32m4; 64 is safe for VLEN up to 512.
          alignas(16) int32_t acc_buf[64];
          alignas(16) int8_t out_buf[64];
          __riscv_vse32_v_i32m4(acc_buf, vacc, vl);
          for (size_t k = 0; k < vl; ++k) {
            int32_t a = tflite::MultiplyByQuantizedMultiplier(
                acc_buf[k], output_multiplier[c + k], output_shift[c + k]);
            a += output_offset;
            a = std::max(a, out_min);
            a = std::min(a, out_max);
            out_buf[k] = static_cast<int8_t>(a);
          }
          for (size_t k = 0; k < vl; ++k) out_row[c + k] = out_buf[k];

          c += static_cast<int>(vl);
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
    DepthwiseConvInt8RVV(params, output_multiplier, output_shift,
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
