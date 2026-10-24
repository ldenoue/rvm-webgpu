import {
  CompiledModel,
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  supportsFeature,
} from '@litertjs/core';
import { GpuFramePipeline } from './gpu-frame-pipeline';
import './style.css';

type Profile = {
  width: number;
  height: number;
  modelUrl: string;
};

const PROFILES: Record<'wide' | 'square', Profile> = {
  wide: {
    width: 512,
    height: 288,
    modelUrl: '/models/rvm_mobilenetv3_512x288_fp16.tflite',
  },
  square: {
    width: 256,
    height: 256,
    modelUrl: '/models/rvm_mobilenetv3_256x256_fp16.tflite',
  },
};

const element = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const video = element<HTMLVideoElement>('video');
const stageElement = element<HTMLElement>('stage');
const inputCanvas = element<HTMLCanvasElement>('input');
const outputCanvas = element<HTMLCanvasElement>('output');
const gpuOutputCanvas = element<HTMLCanvasElement>('gpu-output');
const inputContext = inputCanvas.getContext('2d', { willReadFrequently: true })!;
const outputContext = outputCanvas.getContext('2d')!;
const loadButton = element<HTMLButtonElement>('load');
const benchmarkButton = element<HTMLButtonElement>('benchmark');
const cameraButton = element<HTMLButtonElement>('camera');
const resetButton = element<HTMLButtonElement>('reset');
const acceleratorSelect = element<HTMLSelectElement>('accelerator');
const profileSelect = element<HTMLSelectElement>('profile');
const statusElement = element<HTMLDivElement>('status');
const emptyElement = element<HTMLDivElement>('empty');
const logElement = element<HTMLPreElement>('log');
const fpsElement = element<HTMLElement>('fps');
const latencyElement = element<HTMLElement>('latency');
const backendElement = element<HTMLElement>('backend');
const resolutionElement = element<HTMLElement>('resolution');

let model: CompiledModel | undefined;
let stream: MediaStream | undefined;
let running = false;
let resetRequested = false;
let recurrent: Record<string, Tensor> = {};
let smoothedLatency = 0;
let lastFrameFinished = 0;
let runtimeLoaded = false;
let gpuFramePipeline: GpuFramePipeline | undefined;

function selectedProfile(): Profile {
  return PROFILES[profileSelect.value as keyof typeof PROFILES];
}

function recurrentShapes(profile: Profile): Record<string, number[]> {
  const { width, height } = profile;
  return {
    r1i: [1, height / 2, width / 2, 16],
    r2i: [1, height / 4, width / 4, 20],
    r3i: [1, height / 8, width / 8, 40],
    r4i: [1, height / 16, width / 16, 64],
  };
}

function applyProfileLayout(profile: Profile) {
  inputCanvas.width = profile.width;
  inputCanvas.height = profile.height;
  outputCanvas.width = profile.width;
  outputCanvas.height = profile.height;
  gpuOutputCanvas.width = profile.width;
  gpuOutputCanvas.height = profile.height;
  stageElement.style.aspectRatio = `${profile.width} / ${profile.height}`;
  resolutionElement.textContent = `${profile.width} × ${profile.height}`;
}

function log(message: string) {
  logElement.textContent += `${new Date().toLocaleTimeString()}  ${message}\n`;
  logElement.scrollTop = logElement.scrollHeight;
}

function setStatus(message: string, kind: 'idle' | 'busy' | 'ok' | 'error') {
  statusElement.textContent = message;
  statusElement.dataset.kind = kind;
}

function showReadyStatus(accelerator: 'webgpu' | 'wasm' | 'webnn') {
  if (accelerator === 'wasm') {
    setStatus('wasm · XNNPack CPU', 'ok');
  } else if (accelerator === 'webgpu') {
    setStatus(
      model?.isFullyAccelerated ? 'webgpu · fully accelerated' : 'webgpu · mixed/fallback',
      model?.isFullyAccelerated ? 'ok' : 'idle',
    );
  } else {
    setStatus('webnn · experimental', 'idle');
  }
}

function deleteTensors(tensors: Record<string, Tensor>) {
  for (const tensor of Object.values(tensors)) tensor.delete();
}

function createZeroState() {
  deleteTensors(recurrent);
  recurrent = Object.fromEntries(
    Object.entries(recurrentShapes(selectedProfile())).map(([name, shape]) => [
      name,
      new Tensor(new Float32Array(shape.reduce((a, b) => a * b)), shape),
    ]),
  );
  resetRequested = false;
}

async function loadModel() {
  if (running) stopCamera();
  loadButton.disabled = true;
  acceleratorSelect.disabled = true;
  profileSelect.disabled = true;
  benchmarkButton.disabled = true;
  cameraButton.disabled = true;
  setStatus('Initializing LiteRT…', 'busy');
  try {
    const accelerator = acceleratorSelect.value as 'webgpu' | 'wasm';
    const profile = selectedProfile();
    applyProfileLayout(profile);
    if (accelerator === 'webgpu' && !isWebGPUSupported()) {
      throw new Error('This browser does not expose WebGPU. Try current Chrome or Edge.');
    }

    if (!runtimeLoaded) {
      const jspi = await supportsFeature('jspi');
      log(`WebGPU available: ${isWebGPUSupported()}; JSPI available: ${jspi}`);
      await loadLiteRt('/wasm/', {
        jspi: accelerator === 'webgpu',
        threads: accelerator === 'wasm',
      });
      runtimeLoaded = true;
    }
    deleteTensors(recurrent);
    recurrent = {};
    model?.delete();
    model = undefined;
    setStatus(`Compiling ${profile.width} × ${profile.height} for ${accelerator}…`, 'busy');
    model = await loadAndCompile(profile.modelUrl, {
      accelerator,
      gpuOptions: { precision: 'fp16' },
    });

    createZeroState();
    backendElement.textContent = accelerator;
    const inputs = model.getInputDetails()
      .map((detail) => `${detail.name} [${[...detail.shape].join(', ')}]`)
      .join('; ');
    log(`${profile.width} × ${profile.height} inputs: ${inputs}`);
    log(`Signatures: ${Object.keys(model.signatures).join(', ')}`);
    log(`Fully accelerated: ${model.isFullyAccelerated}`);
    showReadyStatus(accelerator);
    cameraButton.disabled = false;
    benchmarkButton.disabled = false;
    resetButton.disabled = false;
    profileSelect.disabled = false;
    loadButton.disabled = false;
    loadButton.textContent = 'Reload model';
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${message}`);
    setStatus('Model load failed', 'error');
    loadButton.disabled = false;
    acceleratorSelect.disabled = runtimeLoaded;
    profileSelect.disabled = false;
  }
}

async function runBenchmark() {
  if (!model || running) return;
  benchmarkButton.disabled = true;
  cameraButton.disabled = true;
  profileSelect.disabled = true;
  loadButton.disabled = true;
  createZeroState();
  setStatus('Warming up…', 'busy');
  const samples: number[] = [];
  const { width, height } = selectedProfile();
  let benchmarkGpuPipeline: GpuFramePipeline | undefined;
  const syntheticFrame = document.createElement('canvas');
  syntheticFrame.width = width;
  syntheticFrame.height = height;
  const syntheticContext = syntheticFrame.getContext('2d')!;
  syntheticContext.fillStyle = 'rgb(128, 128, 128)';
  syntheticContext.fillRect(0, 0, width, height);

  try {
    if (model.options.accelerator === 'webgpu') {
      benchmarkGpuPipeline = GpuFramePipeline.create({
        canvas: gpuOutputCanvas,
        sourceWidth: width,
        sourceHeight: height,
        width,
        height,
      });
    }
    for (let iteration = 0; iteration < 35; iteration++) {
      const previous = recurrent;
      const started = performance.now();
      const source = benchmarkGpuPipeline
        ? benchmarkGpuPipeline.createInputTensor(syntheticFrame)
        : new Tensor(
          new Float32Array(width * height * 3).fill(0.5),
          [1, height, width, 3],
        );
      const outputs = await model.run({ src: source, ...previous }) as Record<string, Tensor>;
      if (benchmarkGpuPipeline) {
        benchmarkGpuPipeline.composite(outputs.pha);
        outputs.pha.delete();
        await benchmarkGpuPipeline.waitForGpu();
      } else {
        const alphaOnCpu = await outputs.pha.moveTo('wasm');
        alphaOnCpu.toTypedArray();
        alphaOnCpu.delete();
      }
      const elapsed = performance.now() - started;
      recurrent = {
        r1i: outputs.r1o,
        r2i: outputs.r2o,
        r3i: outputs.r3o,
        r4i: outputs.r4o,
      };
      source.delete();
      deleteTensors(previous);
      outputs.fgr.delete();
      if (iteration >= 5) samples.push(elapsed);
      setStatus(iteration < 5 ? `Warming up ${iteration + 1}/5…` : `Benchmarking ${iteration - 4}/30…`, 'busy');
    }

    samples.sort((a, b) => a - b);
    const median = samples[Math.floor(samples.length / 2)];
    const p95 = samples[Math.floor(samples.length * 0.95)];
    latencyElement.textContent = median.toFixed(1);
    fpsElement.textContent = (1000 / median).toFixed(1);
    const pipelineLabel = benchmarkGpuPipeline
      ? 'TypeGPU preprocess + GPU composite'
      : 'CPU input + alpha readback';
    log(`${width} × ${height} benchmark (${pipelineLabel}): median ${median.toFixed(2)} ms; p95 ${p95.toFixed(2)} ms; ${(1000 / median).toFixed(1)} FPS`);
    showReadyStatus(model.options.accelerator);
  } catch (error) {
    log(`Benchmark error: ${error instanceof Error ? error.message : String(error)}`);
    setStatus('Benchmark failed', 'error');
  } finally {
    benchmarkGpuPipeline?.destroy();
    benchmarkButton.disabled = false;
    cameraButton.disabled = false;
    profileSelect.disabled = false;
    loadButton.disabled = false;
  }
}

function frameToTensor(): { tensor: Tensor; rgba: Uint8ClampedArray } {
  const { width, height } = selectedProfile();
  const videoAspect = video.videoWidth / video.videoHeight;
  const targetAspect = width / height;
  let sourceX = 0;
  let sourceY = 0;
  let sourceWidth = video.videoWidth;
  let sourceHeight = video.videoHeight;
  if (videoAspect > targetAspect) {
    sourceWidth = video.videoHeight * targetAspect;
    sourceX = (video.videoWidth - sourceWidth) / 2;
  } else {
    sourceHeight = video.videoWidth / targetAspect;
    sourceY = (video.videoHeight - sourceHeight) / 2;
  }
  inputContext.save();
  inputContext.translate(width, 0);
  inputContext.scale(-1, 1);
  inputContext.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    width,
    height,
  );
  inputContext.restore();
  const rgba = inputContext.getImageData(0, 0, width, height).data;
  const rgb = new Float32Array(width * height * 3);
  for (let source = 0, target = 0; source < rgba.length; source += 4) {
    rgb[target++] = rgba[source] / 255;
    rgb[target++] = rgba[source + 1] / 255;
    rgb[target++] = rgba[source + 2] / 255;
  }
  return { tensor: new Tensor(rgb, [1, height, width, 3]), rgba };
}

function drawAlpha(rgba: Uint8ClampedArray, alpha: Float32Array) {
  const { width, height } = selectedProfile();
  const matte = new ImageData(width, height);
  for (let pixel = 0, channel = 0; pixel < alpha.length; pixel++, channel += 4) {
    matte.data[channel] = rgba[channel];
    matte.data[channel + 1] = rgba[channel + 1];
    matte.data[channel + 2] = rgba[channel + 2];
    matte.data[channel + 3] = Math.max(0, Math.min(255, alpha[pixel] * 255));
  }
  outputContext.putImageData(matte, 0, 0);
}

function destroyGpuFramePipeline() {
  gpuFramePipeline?.destroy();
  gpuFramePipeline = undefined;
  gpuOutputCanvas.hidden = true;
  outputCanvas.hidden = false;
}

async function renderFrame() {
  if (!running || !model) return;
  if (resetRequested) createZeroState();

  const gpuPath = gpuFramePipeline;
  const cpuFrame = gpuPath ? undefined : frameToTensor();
  const source = gpuPath ? gpuPath.createInputTensor(video) : cpuFrame!.tensor;
  const previous = recurrent;
  const started = performance.now();
  try {
    const outputs = await model.run({ src: source, ...previous }) as Record<string, Tensor>;
    recurrent = {
      r1i: outputs.r1o,
      r2i: outputs.r2o,
      r3i: outputs.r3o,
      r4i: outputs.r4o,
    };
    source.delete();
    deleteTensors(previous);
    outputs.fgr.delete();

    const elapsed = performance.now() - started;
    smoothedLatency = smoothedLatency ? smoothedLatency * 0.85 + elapsed * 0.15 : elapsed;
    if (gpuPath) {
      gpuPath.composite(outputs.pha);
      outputs.pha.delete();
    } else {
      const alphaOnCpu = await outputs.pha.moveTo('wasm');
      drawAlpha(cpuFrame!.rgba, alphaOnCpu.toTypedArray() as Float32Array);
      alphaOnCpu.delete();
    }

    const now = performance.now();
    const frameRate = lastFrameFinished ? 1000 / (now - lastFrameFinished) : 0;
    lastFrameFinished = now;
    latencyElement.textContent = smoothedLatency.toFixed(1);
    fpsElement.textContent = frameRate ? frameRate.toFixed(1) : '—';
    requestAnimationFrame(() => void renderFrame());
  } catch (error) {
    source.delete();
    const message = error instanceof Error ? error.message : String(error);
    log(`Inference error: ${message}`);
    setStatus('Inference failed', 'error');
    stopCamera();
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  destroyGpuFramePipeline();
  if (model?.options.accelerator === 'webgpu') {
    try {
      const { width, height } = selectedProfile();
      gpuFramePipeline = GpuFramePipeline.create({
        canvas: gpuOutputCanvas,
        sourceWidth: video.videoWidth,
        sourceHeight: video.videoHeight,
        width,
        height,
      });
      outputCanvas.hidden = true;
      gpuOutputCanvas.hidden = false;
      log('TypeGPU GPU-only camera preprocessing and compositing enabled.');
    } catch (error) {
      log(`TypeGPU pipeline unavailable; using CPU pixels: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  running = true;
  emptyElement.hidden = true;
  cameraButton.textContent = 'Stop camera';
  createZeroState();
  lastFrameFinished = 0;
  void renderFrame();
}

function stopCamera() {
  running = false;
  destroyGpuFramePipeline();
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  cameraButton.textContent = 'Start camera';
}

loadButton.addEventListener('click', () => void loadModel());
benchmarkButton.addEventListener('click', () => void runBenchmark());
profileSelect.addEventListener('change', () => {
  applyProfileLayout(selectedProfile());
  fpsElement.textContent = '—';
  latencyElement.textContent = '—';
  if (model) void loadModel();
});
cameraButton.addEventListener('click', () => {
  if (running) stopCamera();
  else void startCamera().catch((error) => {
    log(`Camera error: ${error instanceof Error ? error.message : String(error)}`);
    setStatus('Camera unavailable', 'error');
  });
});
resetButton.addEventListener('click', () => { resetRequested = true; });

applyProfileLayout(selectedProfile());

if (!isWebGPUSupported()) {
  log('WebGPU is unavailable in this browser; the WASM backend can still be tested.');
}
