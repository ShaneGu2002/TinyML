#include "tflm_demo/kernels/kws_kernels.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace kws_kernels {
namespace {

TfLiteStatus ConvInt8Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, tflite::kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *static_cast<const tflite::OpDataConv*>(node->user_data);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8) {
    MicroPrintf("kws: Conv2D only supports int8/int8.");
    return kTfLiteError;
  }

  KwsConvPerChannelInt8(
      tflite::ConvParamsQuantized(params, data),
      data.per_channel_output_multiplier, data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      bias != nullptr ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr,
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus DepthwiseConvInt8Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(
      context, node, tflite::kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(
      context, node, tflite::kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (tflite::NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node,
                                        tflite::kDepthwiseConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, tflite::kDepthwiseConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *static_cast<const tflite::OpDataConv*>(node->user_data);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8) {
    MicroPrintf("kws: DepthwiseConv2D only supports int8/int8.");
    return kTfLiteError;
  }

  KwsDepthwiseConvPerChannelInt8(
      tflite::DepthwiseConvParamsQuantized(params, data),
      data.per_channel_output_multiplier, data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      bias != nullptr ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr,
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedInt8Invoke(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias = tflite::micro::GetEvalInput(
      context, node, tflite::kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(
      context, node, tflite::kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *static_cast<const tflite::OpDataFullyConnected*>(node->user_data);

  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8) {
    MicroPrintf("kws: FullyConnected only supports int8/int8.");
    return kTfLiteError;
  }

  KwsFullyConnectedInt8(
      tflite::FullyConnectedParamsQuantized(data),
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      bias != nullptr ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr,
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration GetConvInt8Registration() {
  TFLMRegistration r = tflite::Register_CONV_2D();
  r.invoke = ConvInt8Invoke;
  return r;
}

TFLMRegistration GetDepthwiseConvInt8Registration() {
  TFLMRegistration r = tflite::Register_DEPTHWISE_CONV_2D();
  r.invoke = DepthwiseConvInt8Invoke;
  return r;
}

TFLMRegistration GetFullyConnectedInt8Registration() {
  TFLMRegistration r = tflite::Register_FULLY_CONNECTED();
  r.invoke = FullyConnectedInt8Invoke;
  return r;
}

}  // namespace kws_kernels
