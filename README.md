# RVM on LiteRT.js / WebGPU

An experimental browser port of [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) using a fixed-shape LiteRT (`.tflite`) model and the LiteRT.js WebGPU runtime.

Everything runs locally in the browser: camera frames and mattes are not uploaded to a server.

## Highlights

- Fully accelerated LiteRT.js WebGPU inference with a WASM/XNNPack fallback.
- TypeGPU preprocessing and compositing keep the WebGPU camera path on the GPU.
- Recurrent RVM state is preserved between video frames.
- Fixed 512 × 288 and 256 × 256 model profiles.
- Built-in, permission-free 30-run benchmark.

The UI can switch between fixed **512 × 288** and **256 × 256** profiles. Camera frames are center-cropped to the selected aspect ratio, and both models run the MobileNetV3 RVM graph at `downsample_ratio=1`. RVM's four recurrent states remain on the selected accelerator between frames. The UI reports `CompiledModel.isFullyAccelerated`, so a mixed WebGPU/WASM graph is visible instead of being mistaken for full GPU execution.

On WebGPU, a TypeGPU pipeline mirrors, crops, resizes, and converts each camera frame to the model's RGB float tensor directly on the GPU. It also composites the alpha output straight to the display canvas, avoiding per-frame pixel loops and GPU-to-CPU matte readback. The WASM backend retains the CPU canvas path as a fallback.

## Run

Requirements: Node.js 20 or newer, a current Chromium browser, and Python 3.11 only if you need to build the models.

The model binaries are intentionally excluded from Git. Rebuild them first (see below), then:

```bash
npm install
npm run dev
```

Open the local URL in a current Chromium browser. Camera access works on `localhost`. Choose WebGPU or WASM before loading the model.

Use **Benchmark 30×** for a permission-free synthetic test: it discards five warm-up runs, synchronizes each run, then reports median latency, p95 latency in the runtime log, and derived FPS. The WebGPU benchmark includes TypeGPU preprocessing and GPU compositing; the WASM benchmark includes CPU preprocessing and alpha readback. The development and preview servers send cross-origin isolation headers so LiteRT's threaded WASM build can be measured fairly.

In one local Chromium run at 512 × 288, the complete TypeGPU path measured 8.3–10.1 ms per frame (98.8–120.6 FPS), compared with 13.4 ms (74.6 FPS) for the benchmark's CPU-input/alpha-readback path. Results depend heavily on the GPU, browser, and thermal state.

## Data flow

```text
camera frame
  → TypeGPU crop / mirror / resize / RGB conversion
  → LiteRT.js RVM inference on WebGPU
  → TypeGPU alpha compositing
  → WebGPU canvas
```

If WebGPU is unavailable or WASM is selected, canvas pixel conversion and alpha rendering run on the CPU instead.

## Rebuild the model

Python 3.11 and TensorFlow 2.15 are used because the official SavedModel dates from TensorFlow 2.x / Keras 2.

```bash
uv venv .venv --python 3.11
UV_CACHE_DIR=/tmp/rvm-uv-cache uv pip install --python .venv/bin/python tensorflow==2.15.1 numpy==1.26.4
npm run model:download
npm run model:convert
```

The converter produces both profiles with fixed input and recurrent-state shapes, supplies a constant `downsample_ratio=1`, and applies float16 weight quantization. Inputs, recurrent states, and outputs remain float32. Each model is about 7.4 MiB.

## Build

```bash
npm run build
npm run preview
```

## What to measure next

- Compare WebGPU and WASM after at least 20 warm-up frames.
- Trust the “fully accelerated” flag: on JSPI-capable browsers, unsupported operations can run on WASM while the rest remains on WebGPU.
- Try a smaller 384 × 216 profile for latency, or a fixed 1280 × 720 profile with a 0.375 internal scale and the guided refiner for higher-quality edges.
- Profile real camera frames with browser GPU tooling; the synthetic benchmark isolates the pipeline without camera scheduling and decoding costs.

## Licensing

This project does not redistribute model weights in Git. The upstream RVM source and weights are licensed under GPL-3.0; review that license before distributing a converted model or derivative application. LiteRT.js is Apache-2.0.
