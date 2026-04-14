# TFLM Demo Skeleton

This folder contains a minimal TensorFlow Lite Micro inference skeleton for the
INT8 DS-CNN model exported in `artifacts/ds_cnn/`.

## Files

- `main.cc`: minimal entry point that runs one inference.
- `kws_inference.h`: public inference wrapper.
- `kws_inference.cc`: TFLM interpreter setup and invoke logic.
- `../artifacts/ds_cnn/model_data.h`: declaration for the generated model blob.

## What to replace

1. `main.cc` is already wired to `test_input_data.cc`, which contains a real
   quantized MFCC exported from
   `speech-commands-v2/yes/0a7c2a8d_nohash_0.wav`.
2. Replace `test_input_data.cc` with your own exported sample if you want to
   test another utterance.
3. Copy these files into your RISC-V / Spike project and point the include
   paths to your local TFLM checkout.
4. If `AllocateTensors()` fails, increase `kTensorArenaSize` in
   `kws_inference.cc`.
5. If operator registration fails at runtime, inspect the TFLite model and add
   any missing ops to the resolver.

## Expected label order

The model outputs six categories in this order:

1. `yes`
2. `no`
3. `stop`
4. `go`
5. `_unknown_`
6. `_silence_`
