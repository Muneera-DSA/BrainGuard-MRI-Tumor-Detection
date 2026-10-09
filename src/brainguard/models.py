"""Model definitions.

- baseline_cnn : 4-block CNN, the architecture of the original BrainGuard notebook.
- efficientnet : EfficientNetB0 (ImageNet weights) + small head. Trained in two phases:
                 head only, then the top of the backbone unfrozen at a lower learning rate
                 (BatchNorm layers stay frozen, the standard recipe for small datasets).

Layer names are fixed because explain.py looks them up for Grad-CAM.
"""

from __future__ import annotations

import keras
from keras import layers


def augmentation(seed: int = 0) -> keras.Sequential:
    """Mild, anatomy-preserving augmentation; active only during training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.04, fill_mode="constant", seed=seed),
        layers.RandomZoom(0.10, fill_mode="constant", seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
        layers.RandomContrast(0.15, seed=seed),
    ], name="augment")


def baseline_cnn(n_classes: int, img_size: int, seed: int = 0) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation(seed)(inputs)
    x = layers.Rescaling(1 / 255.0)(x)
    for i, f in enumerate([32, 64, 128, 256]):
        x = layers.Conv2D(f, 3, padding="same", activation="relu",
                          name="last_conv" if i == 3 else f"conv{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32", name="head_dense")(x)
    return keras.Model(inputs, out, name="baseline_cnn")


def efficientnet(n_classes: int, img_size: int, weights: str | None = "imagenet", seed: int = 0) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation(seed)(inputs)
    base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                             input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base(x, training=False)      # BatchNorm statistics frozen even after unfreezing
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32", name="head_dense")(x)
    return keras.Model(inputs, out, name="efficientnet_b0")


def unfreeze_top(model: keras.Model, fraction: float = 0.3) -> None:
    """Unfreeze the top `fraction` of EfficientNet layers, keeping BatchNorm frozen."""
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    n = len(base.layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= int(n * (1 - fraction)) and not isinstance(layer, layers.BatchNormalization)


def build(name: str, n_classes: int, img_size: int, weights: str | None = "imagenet", seed: int = 0) -> keras.Model:
    if name == "cnn":
        return baseline_cnn(n_classes, img_size, seed)
    if name == "effnet":
        return efficientnet(n_classes, img_size, weights, seed)
    raise ValueError(name)
