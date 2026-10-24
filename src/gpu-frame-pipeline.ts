import { Tensor, getWebGpuDevice } from '@litertjs/core';
import tgpu, { common, d, std, type TgpuRoot } from 'typegpu';

type GpuFramePipelineOptions = {
  canvas: HTMLCanvasElement;
  sourceWidth: number;
  sourceHeight: number;
  width: number;
  height: number;
};

/**
 * GPU-only camera preprocessing and alpha compositing around LiteRT inference.
 * TypeGPU and LiteRT share one GPUDevice and exchange raw GPUBuffer objects.
 */
export class GpuFramePipeline {
  readonly #root: TgpuRoot;
  readonly #sourceTexture;
  readonly #inputBuffer: GPUBuffer;
  readonly #preprocess;
  readonly #composite;
  readonly #alphaLayout;
  readonly #context: GPUCanvasContext;
  readonly #width: number;
  readonly #height: number;

  private constructor(options: GpuFramePipelineOptions, device: GPUDevice) {
    const { canvas, sourceWidth, sourceHeight, width, height } = options;
    this.#width = width;
    this.#height = height;
    this.#root = tgpu.initFromDevice({ device });

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not create a WebGPU canvas context.');
    this.#context = context;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });

    this.#sourceTexture = this.#root
      .createTexture({
        size: [sourceWidth, sourceHeight],
        format: 'rgba8unorm',
      })
      .$usage('sampled', 'render');
    const sourceView = this.#sourceTexture.createView();
    const sampler = this.#root.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    const targetAspect = width / height;
    const videoAspect = sourceWidth / sourceHeight;
    const cropScaleX = videoAspect > targetAspect ? targetAspect / videoAspect : 1;
    const cropScaleY = videoAspect > targetAspect ? 1 : videoAspect / targetAspect;
    const cropOffsetX = (1 - cropScaleX) / 2;
    const cropOffsetY = (1 - cropScaleY) / 2;

    const sampleVideo = tgpu.fn([d.vec2f], d.vec4f)((uv) => {
      'use gpu';
      const sourceUv = d.vec2f(
        cropOffsetX + (1 - uv.x) * cropScaleX,
        cropOffsetY + uv.y * cropScaleY,
      );
      return std.textureSampleLevel(sourceView.$, sampler.$, sourceUv, 0);
    });

    this.#inputBuffer = device.createBuffer({
      label: 'RVM normalized RGB input',
      size: width * height * 3 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    const inputMutable = this.#root.createMutable(
      d.arrayOf(d.f32, width * height * 3),
      this.#inputBuffer,
    );
    this.#preprocess = this.#root.createGuardedComputePipeline((x, y) => {
      'use gpu';
      const uv = d.vec2f(
        (d.f32(x) + 0.5) / width,
        (d.f32(y) + 0.5) / height,
      );
      const color = sampleVideo(uv);
      const offset = (y * width + x) * 3;
      inputMutable.$[offset] = color.r;
      inputMutable.$[offset + 1] = color.g;
      inputMutable.$[offset + 2] = color.b;
    });

    this.#alphaLayout = tgpu.bindGroupLayout({
      alpha: { storage: d.arrayOf(d.f32, width * height) },
    });
    const compositeFragment = tgpu.fragmentFn({
      in: { uv: d.vec2f },
      out: d.vec4f,
    })((input) => {
      'use gpu';
      const x = std.min(d.u32(input.uv.x * width), d.u32(width - 1));
      const y = std.min(d.u32(input.uv.y * height), d.u32(height - 1));
      const alpha = std.clamp(this.#alphaLayout.$.alpha[y * width + x], 0, 1);
      const color = sampleVideo(input.uv);
      return d.vec4f(std.mul(color.rgb, alpha), alpha);
    });
    this.#composite = this.#root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: compositeFragment,
      targets: { format: presentationFormat },
    });
  }

  static create(options: GpuFramePipelineOptions): GpuFramePipeline {
    const device = getWebGpuDevice();
    if (!device) throw new Error('LiteRT did not expose its WebGPU device.');
    if (!options.sourceWidth || !options.sourceHeight) {
      throw new Error('Source dimensions are not available.');
    }
    return new GpuFramePipeline(options, device);
  }

  createInputTensor(source: GPUCopyExternalImageSource): Tensor {
    this.#sourceTexture.write(source);
    this.#preprocess.dispatchThreads(this.#width, this.#height);
    return new Tensor(
      this.#inputBuffer,
      [1, this.#height, this.#width, 3],
      'float32',
    );
  }

  composite(alpha: Tensor): void {
    const bindGroup = this.#root.createBindGroup(this.#alphaLayout, {
      alpha: alpha.toGpuBuffer(),
    });
    this.#composite
      .with(bindGroup)
      .withColorAttachment({
        view: this.#context,
        loadOp: 'clear',
        storeOp: 'store',
      })
      .draw(3);
  }

  async waitForGpu(): Promise<void> {
    await this.#root.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    this.#sourceTexture.destroy();
    this.#inputBuffer.destroy();
    this.#root.destroy();
  }
}
