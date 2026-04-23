// KERNEL=rvv: RVV 1.0 intrinsic int8 fully connected.
// TODO: replace with a real RVV implementation. Placeholder delegates to the
// TFLM reference so the rvv variant still builds on rv64gcv.

#include "tflm_demo/kernels/kws_kernels.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"

namespace kws_kernels {

void KwsFullyConnectedInt8(
    const tflite::FullyConnectedParams& params,
    const tflite::RuntimeShape& input_shape, const int8_t* input_data,
    const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
    const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
    const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  tflite::reference_integer_ops::FullyConnected(
      params,
      input_shape, input_data,
      filter_shape, filter_data,
      bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace kws_kernels
