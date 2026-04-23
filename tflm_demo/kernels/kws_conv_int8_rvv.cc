// KERNEL=rvv : RVV 1.0 intrinsic int8 per-channel convolution.
//
// Specializes 1x1 pointwise (stride 1, no padding, no dilation) by
// vectorizing over the input-channel axis with a widening int8→int16→int32
// MAC followed by a reduction per output channel. Other geometries fall back
// to the TFLM reference.
//
// Compiled with -fno-tree-vectorize (see Makefile); any instret reduction
// seen here is strictly attributable to hand-written intrinsics, not to GCC
// auto-vectorization.

#include "tflm_demo/kernels/kws_kernels.h"

#include <algorithm>
#include <cstdint>
#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

namespace kws_kernels {
namespace {

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

// DS-CNN's first conv reads a single input channel (MFCC has 1 channel per
// time-frequency bin), so its inner product collapses to a single scalar
// times a filter vector. We vectorize over the output-channel axis and use a
// strided load to gather filter[oc..oc+vl, fh, fw, 0] for the 76 adjacent
// output channels at once.
bool IsFirstLayerIc1(const tflite::RuntimeShape& input_shape,
                     const tflite::RuntimeShape& filter_shape,
                     const tflite::ConvParams& params) {
  const int in_channels = input_shape.Dims(3);
  const int filter_ic = filter_shape.Dims(3);
  return in_channels == 1 && filter_ic == 1 &&
         params.dilation_width_factor == 1 &&
         params.dilation_height_factor == 1;
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

// Vectorized int8 dot product with input-zero-point offset:
//   returns sum_{i=0..n-1} (a[i] + a_offset) * b[i]
// using RVV intrinsics. a and b are int8; offset is a small int32 known to
// fit in int16 (input_offset = -input_zero_point, zp is int8).
int32_t DotInt8WithOffset(const int8_t* a, const int8_t* b, int n,
                          int32_t a_offset) {
  int32_t acc = 0;
  int i = 0;
  while (i < n) {
    const size_t vl = __riscv_vsetvl_e8mf2(n - i);
    // Load int8 lanes and widen to int16 so we can add the offset.
    vint8mf2_t va_i8 = __riscv_vle8_v_i8mf2(a + i, vl);
    vint8mf2_t vb_i8 = __riscv_vle8_v_i8mf2(b + i, vl);
    vint16m1_t va_i16 = __riscv_vwadd_vx_i16m1(va_i8, a_offset, vl);
    vint16m1_t vb_i16 = __riscv_vsext_vf2_i16m1(vb_i8, vl);
    // Widening MAC into int32: acc_i32 += va_i16 * vb_i16.
    // vlmax is fine here — unused lanes are ignored by the final reduction.
    vint32m2_t vprod = __riscv_vwmul_vv_i32m2(va_i16, vb_i16, vl);
    // Reduce this chunk's products into the scalar accumulator.
    vint32m1_t vseed = __riscv_vmv_s_x_i32m1(acc, 1);
    vint32m1_t vsum =
        __riscv_vredsum_vs_i32m2_i32m1(vprod, vseed, vl);
    acc = __riscv_vmv_x_s_i32m1_i32(vsum);
    i += static_cast<int>(vl);
  }
  return acc;
}

void ConvInt8FirstLayerRVV(
    const tflite::ConvParams& params,
    const int32_t* per_channel_multiplier, const int32_t* per_channel_shift,
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
  const int out_channels = filter_shape.Dims(0);
  const int filter_h = filter_shape.Dims(1);
  const int filter_w = filter_shape.Dims(2);
  const int out_h = output_shape.Dims(1);
  const int out_w = output_shape.Dims(2);
  const ptrdiff_t filter_stride_oc = filter_h * filter_w;  // IC=1, int8

  for (int b = 0; b < batches; ++b) {
    const int8_t* in_batch = input_data + b * in_h * in_w;  // IC == 1
    int8_t* out_batch = output_data + b * out_h * out_w * out_channels;

    for (int oh = 0; oh < out_h; ++oh) {
      const int ih_origin = oh * stride_h - pad_h;
      const int fh_start = std::max(0, -ih_origin);
      const int fh_end = std::min(filter_h, in_h - ih_origin);

      for (int ow = 0; ow < out_w; ++ow) {
        const int iw_origin = ow * stride_w - pad_w;
        const int fw_start = std::max(0, -iw_origin);
        const int fw_end = std::min(filter_w, in_w - iw_origin);
        int8_t* out_row = out_batch + (oh * out_w + ow) * out_channels;

        for (int oc = 0; oc < out_channels;) {
          const size_t vl =
              __riscv_vsetvl_e32m4(static_cast<size_t>(out_channels - oc));
          vint32m4_t vacc = bias_data
                                ? __riscv_vle32_v_i32m4(bias_data + oc, vl)
                                : __riscv_vmv_v_x_i32m4(0, vl);

          for (int fh = fh_start; fh < fh_end; ++fh) {
            const int ih = ih_origin + fh;
            for (int fw = fw_start; fw < fw_end; ++fw) {
              const int iw = iw_origin + fw;
              const int16_t x =
                  static_cast<int16_t>(in_batch[ih * in_w + iw]) +
                  static_cast<int16_t>(input_offset);
              const int8_t* f_ptr =
                  filter_data + oc * filter_stride_oc + fh * filter_w + fw;
              vint8m1_t vf8 =
                  __riscv_vlse8_v_i8m1(f_ptr, filter_stride_oc, vl);
              vint16m2_t vf16 = __riscv_vsext_vf2_i16m2(vf8, vl);
              vacc = __riscv_vwmacc_vx_i32m4(vacc, x, vf16, vl);
            }
          }

          alignas(16) int32_t acc_buf[64];
          alignas(16) int8_t out_buf[64];
          __riscv_vse32_v_i32m4(acc_buf, vacc, vl);
          for (size_t k = 0; k < vl; ++k) {
            int32_t a = tflite::MultiplyByQuantizedMultiplier(
                acc_buf[k], per_channel_multiplier[oc + k],
                per_channel_shift[oc + k]);
            a += output_offset;
            a = std::max(a, out_min);
            a = std::min(a, out_max);
            out_buf[k] = static_cast<int8_t>(a);
          }
          for (size_t k = 0; k < vl; ++k) out_row[oc + k] = out_buf[k];
          oc += static_cast<int>(vl);
        }
      }
    }
  }
}

void ConvInt8Pointwise1x1RVV(
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

    for (int oc = 0; oc < out_channels; ++oc) {
      const int8_t* w = filter_data + oc * in_channels;
      int32_t acc = bias_data ? bias_data[oc] : 0;
      acc += DotInt8WithOffset(in_row, w, in_channels, input_offset);
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
    ConvInt8Pointwise1x1RVV(params, output_multiplier, output_shift,
                            input_shape, input_data,
                            filter_shape, filter_data,
                            bias_shape, bias_data,
                            output_shape, output_data);
    return;
  }
  if (IsFirstLayerIc1(input_shape, filter_shape, params)) {
    ConvInt8FirstLayerRVV(params, output_multiplier, output_shift,
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
