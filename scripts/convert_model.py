#!/usr/bin/env python3
"""Convert the official RVM TensorFlow SavedModel to a fixed-shape LiteRT model.

The fixed recurrent-state shapes avoid resizing tensors between frames and give the
WebGPU compiler a completely static graph. The default 512x288 profile runs RVM at
downsample_ratio=1; the browser scales camera frames to that resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


REC_CHANNELS = (16, 20, 40, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Store constant weights as float16 (inputs and outputs stay float32).",
    )
    return parser.parse_args()


def recurrent_shapes(height: int, width: int) -> list[list[int]]:
    if height % 16 or width % 16:
        raise ValueError("width and height must both be divisible by 16")
    return [
        [1, height // scale, width // scale, channels]
        for scale, channels in zip((2, 4, 8, 16), REC_CHANNELS, strict=True)
    ]


def main() -> None:
    args = parse_args()
    loaded_model = tf.saved_model.load(str(args.saved_model))
    rec_shapes = recurrent_shapes(args.height, args.width)

    class FixedRvm(tf.Module):
        def __init__(self):
            super().__init__()
            # Keep the revived variables alive and tracked during conversion.
            self.loaded_model = loaded_model
            self.signature = loaded_model.signatures["serving_default"]

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, args.height, args.width, 3], tf.float32, "src"),
                *[
                    tf.TensorSpec(shape, tf.float32, f"r{i}i")
                    for i, shape in enumerate(rec_shapes, start=1)
                ],
            ]
        )
        def serving_default(self, src, r1i, r2i, r3i, r4i):
            return self.signature(
                src=src,
                r1i=r1i,
                r2i=r2i,
                r3i=r3i,
                r4i=r4i,
                downsample_ratio=tf.constant(1.0),
            )

    module = FixedRvm()
    concrete = module.serving_default.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if args.float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    model = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model)
    print(f"Wrote {args.output} ({len(model) / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
