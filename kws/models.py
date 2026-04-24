from typing import Optional, Tuple

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


def build_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    layers: int = 3,
    filters: int = 32,
    kernel: Tuple[int, int] = (3, 3),
    pool: Tuple[int, int] = (2, 2),
    dropout: float = 0.1,
) -> tf.keras.Model:
    """VGG-style CNN with BatchNorm and Global Average Pooling head.

    Stacks `layers` blocks of Conv + BN + ReLU. A MaxPool with stride `pool`
    is applied between successive conv blocks (skipped when the spatial
    dimension would collapse below the pool size). A Global Average Pooling
    head replaces Flatten+Dense, which avoids a large fully-connected
    bottleneck and keeps training numerically stable. `same` padding is used
    so pool bookkeeping is straightforward.

    References: Simonyan & Zisserman 2014 (VGG 3x3 stacking), Ioffe & Szegedy
    2015 (BatchNorm), Lin et al. 2013 (Network-in-Network / GAP head).
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = conv_block(inputs, filters, kernel, name="conv1")
    height = int(inputs.shape[1])
    width = int(inputs.shape[2])
    for i in range(1, layers):
        if pool != (1, 1) and height >= pool[0] and width >= pool[1]:
            x = tf.keras.layers.MaxPooling2D(
                pool_size=pool, strides=pool, name=f"pool{i}"
            )(x)
            height //= pool[0]
            width //= pool[1]
        x = conv_block(x, filters, kernel, name=f"conv{i + 1}")
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"conv{i + 1}_dropout")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn")


def build_dnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    layers: int = 3,
    units: int = 144,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Hello Edge style fully-connected DNN (Appendix A, Table 4).

    Flatten -> (Dense + ReLU [+ Dropout]) x `layers` -> Dense softmax.
    No BatchNorm, matching the paper's baseline. `dropout=0` disables it.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = tf.keras.layers.Flatten(name="flatten")(inputs)
    for i in range(1, layers + 1):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"fc{i}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"fc{i}_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dnn")


def build_ds_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    layers: int = 5,
    filters: int = 64,
    kernel: Tuple[int, int] = (3, 3),
    first_stride: Tuple[int, int] = (2, 2),
    dropout: float = 0.15,
) -> tf.keras.Model:
    """Hello Edge style DS-CNN.

    `layers` is the total number of conv layers: 1 regular conv stem (with
    `first_stride`) plus `layers - 1` depthwise-separable conv blocks. All
    conv layers share the same `kernel` and `filters`.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = conv_block(inputs, filters, kernel, strides=first_stride, name="stem")
    for index in range(1, layers):
        x = ds_conv_block(x, filters, kernel_size=kernel, dropout=dropout, name=f"ds_block{index}")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ds_cnn")


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
    if model_name == "cnn":
        return build_cnn(input_shape=input_shape, num_classes=num_classes, **kwargs)
    if model_name == "ds_cnn":
        return build_ds_cnn(input_shape=input_shape, num_classes=num_classes, **kwargs)
    if model_name == "dnn":
        return build_dnn(input_shape=input_shape, num_classes=num_classes, **kwargs)
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
