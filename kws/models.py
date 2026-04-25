from typing import Iterable, Optional, Tuple

import tensorflow as tf


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    name: Optional[str] = None,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=None if name is None else f"{name}_relu")(x)
    return x


def ds_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    strides: Tuple[int, int] = (1, 1),
    dropout: float = 0.1,
    name: Optional[str] = None,
) -> tf.Tensor:
    x = tf.keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_sepconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=None if name is None else f"{name}_relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=None if name is None else f"{name}_dropout")(x)
    return x


def build_cnn_trad_fpool3(
    input_shape: Tuple[int, int, int], num_classes: int
) -> tf.keras.Model:
    """Sainath & Parada 2015 `cnn-trad-fpool3` (Table 2 CNN-1)."""
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = tf.keras.layers.Conv2D(64, (20, 8), padding="valid", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), name="fpool3")(x)
    x = tf.keras.layers.Conv2D(64, (10, 4), padding="valid", activation="relu", name="conv2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(32, name="linear")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dnn")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_trad_fpool3")


def build_cnn_one_fstride4(
    input_shape: Tuple[int, int, int], num_classes: int, feature_maps: int = 186
) -> tf.keras.Model:
    """Sainath & Parada 2015 `cnn-one-fstride4` (Table 2 CNN-2)."""
    time_frames = input_shape[0]
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = tf.keras.layers.Conv2D(
        feature_maps,
        kernel_size=(time_frames, 8),
        strides=(1, 4),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(32, name="linear")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dnn1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dnn2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_one_fstride4")


def build_ds_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    ds_filters: Iterable[int] = (64, 64, 64, 64),
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = conv_block(inputs, 64, (3, 3), strides=(2, 2), name="stem")
    for index, filters in enumerate(ds_filters, start=1):
        x = ds_conv_block(x, filters, dropout=0.15, name=f"ds_block{index}")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ds_cnn")


class SharedGRUUnroll(tf.keras.layers.Layer):
    """A statically-unrolled GRU that reuses a single `GRUCell` across timesteps.

    We build the recurrence with a Python `for` loop and a *single* shared
    `GRUCell` instance. This sidesteps two TFLite conversion landmines:
      * `tf.keras.layers.GRU` on Apple Metal / CUDA gets fused into the opaque
        `CudnnRNNV3` op, which the TFLite converter cannot ingest.
      * `tf.keras.layers.GRU(unroll=True)` re-instantiates the kernel/bias
        tensors per timestep in the frozen graph, blowing the int8 .tflite
        file up by ~T×.
    Reusing the cell keeps variables shared and produces a compact, fully
    quantizable Dense + element-wise graph.
    """

    def __init__(self, units: int, time_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.time_steps = time_steps
        self.cell = tf.keras.layers.GRUCell(units, reset_after=True, name="gru_cell")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        state = tf.zeros((batch_size, self.units), dtype=inputs.dtype)
        for t in range(self.time_steps):
            state, _ = self.cell(inputs[:, t, :], [state])
        return state

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "time_steps": self.time_steps})
        return config


def build_gru(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    gru_units: int = 64,
) -> tf.keras.Model:
    """Single-layer GRU classifier over MFCC time-frequency frames.

    Input is the MFCC tensor `(time, freq, 1)`; we squeeze the trailing
    channel dim and feed `(time, freq)` to the GRU as a sequence.
    """
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    time_steps, feature_bins = input_shape[0], input_shape[1]
    x = tf.keras.layers.Reshape((time_steps, feature_bins), name="squeeze")(inputs)
    # See `SharedGRUUnroll` for why we hand-unroll instead of using `keras.layers.GRU`.
    state = SharedGRUUnroll(gru_units, time_steps, name="gru")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(state)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"gru_{gru_units}")


def get_gru_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    gru_units: int = 64,
) -> tf.keras.Model:
    """Public alias kept for backwards-compat with the project spec."""
    return build_gru(input_shape=input_shape, num_classes=num_classes, gru_units=gru_units)


def build_model(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    **kwargs,
) -> tf.keras.Model:
    if model_name == "cnn_trad_fpool3":
        return build_cnn_trad_fpool3(input_shape=input_shape, num_classes=num_classes)
    if model_name == "cnn_one_fstride4":
        return build_cnn_one_fstride4(input_shape=input_shape, num_classes=num_classes)
    if model_name == "ds_cnn":
        return build_ds_cnn(input_shape=input_shape, num_classes=num_classes)
    if model_name == "gru":
        return build_gru(
            input_shape=input_shape,
            num_classes=num_classes,
            gru_units=int(kwargs.get("gru_units", 64)),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def estimate_operations(model: tf.keras.Model) -> int:
    """Estimate multiply-accumulate ops per inference (≈ MACs)."""
    total_macs = 0
    for layer in model.layers:
        out_shape = getattr(layer, "output", None)
        if out_shape is None:
            continue
        out_shape = out_shape.shape
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kh, kw = layer.kernel_size
            _, oh, ow, oc = out_shape
            total_macs += kh * kw * oc * oh * ow
        elif isinstance(layer, tf.keras.layers.SeparableConv2D):
            kh, kw = layer.kernel_size
            _, oh, ow, oc = out_shape
            ic = layer.input.shape[-1]
            total_macs += kh * kw * ic * oh * ow        # depthwise
            total_macs += ic * oc * oh * ow             # pointwise
        elif isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            _, oh, ow, oc = out_shape
            ic = layer.input.shape[-1]
            total_macs += kh * kw * ic * oc * oh * ow
        elif isinstance(layer, tf.keras.layers.Dense):
            ic = layer.input.shape[-1]
            oc = out_shape[-1]
            total_macs += ic * oc
        elif isinstance(layer, tf.keras.layers.GRU):
            input_shape = layer.input.shape
            time_steps = input_shape[1]
            input_dim = input_shape[-1]
            units = layer.units
            if time_steps is not None and input_dim is not None:
                # 3 gates: input->hidden (input_dim*units) and hidden->hidden (units*units), per timestep.
                total_macs += time_steps * 3 * (input_dim * units + units * units)
    return int(total_macs)


def estimate_peak_activation_bytes(model: tf.keras.Model, bytes_per_element: int = 4) -> int:
    peak_bytes = 0
    for layer in model.layers:
        layer_output = getattr(layer, "output", None)
        if layer_output is None:
            continue
        if isinstance(layer_output, list):
            tensors = layer_output
        else:
            tensors = [layer_output]
        for tensor in tensors:
            shape = tensor.shape
            if not shape or any(dim is None for dim in shape[1:]):
                continue
            num_elements = 1
            for dim in shape[1:]:
                num_elements *= dim
            peak_bytes = max(peak_bytes, num_elements * bytes_per_element)
    return peak_bytes
