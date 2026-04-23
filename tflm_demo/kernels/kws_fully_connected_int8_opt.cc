// KERNEL=opt: hand-written scalar optimized int8 fully connected.
// TODO: replace this placeholder with a real scalar optimization. For now
// delegates to the TFLM reference so KERNEL=opt still builds and produces
// bit-identical output.

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
