"""tf.data input pipeline. Images are read as greyscale, resized, cached as uint8 and
replicated to 3 channels (ImageNet backbones expect RGB). Pixel values stay 0-255:
EfficientNet normalises internally and the baseline CNN has its own Rescaling layer."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import tensorflow as tf


def _load(path, img_size: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def to_model_input(img_uint8):
    return tf.image.grayscale_to_rgb(tf.cast(img_uint8, tf.float32))


def make_dataset(paths: Sequence[str], labels: Sequence[int] | None, img_size: int, batch_size: int,
                 shuffle: bool = False, seed: int = 0,
                 perturb: Callable[[tf.Tensor], tf.Tensor] | None = None) -> tf.data.Dataset:
    """`perturb` (robustness tests) receives a float32 HxWx1 image in 0-255 and returns the same."""
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(paths, dtype=str))
    ds = ds.map(lambda p: _load(p, img_size), num_parallel_calls=tf.data.AUTOTUNE).cache()
    if labels is not None:
        ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(np.asarray(labels, dtype=np.int32))))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def finish(img):
        if perturb is not None:
            img = tf.cast(tf.clip_by_value(perturb(tf.cast(img, tf.float32)), 0, 255), tf.uint8)
        return to_model_input(img)

    if labels is not None:
        ds = ds.map(lambda x, y: (finish(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(finish, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_array(paths: Sequence[str], img_size: int) -> np.ndarray:
    """Eager load into an (n, H, W, 3) float32 array (explainability on small subsets)."""
    return np.stack([to_model_input(_load(tf.constant(p), img_size)).numpy() for p in paths])
