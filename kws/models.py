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


def build_baseline_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="mfcc")
    x = conv_block(inputs, 32, (3, 3), name="block1")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = conv_block(x, 64, (3, 3), name="block2")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = conv_block(x, 96, (3, 3), name="block3")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


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


def build_model(model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    if model_name == "baseline_cnn":
        return build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)
    if model_name == "ds_cnn":
        return build_ds_cnn(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name}")


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
