# Based GAN

Experimental Thermal2RGB-compatible ONNX translation for the TOPDON viewer's
Custom node. It synthesizes RGB display pixels from thermal grayscale images.
Generated textures/colors are predictions; temperature measurements remain in
the viewer's separate native radiometric arrays.

## Load the finished node

This repository belongs at `examples/custom_nodes/based-gan/` in the viewer.
In the pipeline editor add **Custom**, choose **Open module folder…**, and select
**`examples/custom_nodes/based-gan/nodes/Based GAN`**. Leave **ONNX model** empty
to download the checksum-pinned native-aspect model from our
[v0.1.0 release](https://github.com/manbehindthemadness/based-gan/releases/tag/v0.1.0).
The worker returns the input image while the background download is pending.
Alternatively, select a local converted model with **ONNX model → Browse…**:

```text
examples/custom_nodes/based-gan/artifacts/thermal2rgb-instance-portable-256x192.onnx
```

The local prototype also provides an importable pipeline with embedded package
code and its current absolute model path:
`artifacts/worker/based-gan-demo.json`. It uses the raw source and a 15–40°C display
range; adjust that display range for your scene. Import it through the viewer's
pipeline import control. Neither loading method changes existing saved presets.

Use a grayscale thermal branch before color palettes, annotations, combines and
AI upscaling. The model sees the image before viewer rotation. The native-aspect
256×192 export avoids stretching that landscape source to a square. Output returns
to the incoming image's dimensions, preserving subsequent geometric mapping.

Controls:

| Control | Behavior |
| --- | --- |
| ONNX model | Empty downloads the default model; Browse overrides it with local converter output |
| Inference device | CPU, Apple Core ML or NVIDIA CUDA, with viewer capability fallback |
| Apple compute devices | Default CPU + GPU; ALL also tested locally |
| Input | Grayscale replicated to RGB by default; RGB supports explicit input experiments |
| Invert model input | Switch white-hot/black-hot polarity only for model input |
| Generated image strength | Blend generated RGB with incoming display pixels; zero skips inference |

Sessions initialize lazily in the worker and persist between frames. A changed
model path or effective device rebuilds the session. Missing models, unsupported
schemas and inference errors use the viewer's pipeline-error behavior. Select
the folder again after code updates, restore your desired controls, then save the
pipeline: saved pipelines embed code and do not watch this checkout.

The runtime uses dependencies already installed in the viewer. Conversion needs
optional Torch/ONNX dependencies in this repository's separate environment; it
does not install them in the viewer or download models during folder selection.
The default model uses the viewer's shared background downloader and model cache.
Its URL pins release `v0.1.0`, its SHA-256 pins the exported bytes, and its descriptor
allows 200 MB because the FP32 model exceeds the downloader's 70 MB default.

## Results on Apple M4

ONNX Runtime 1.30.0, macOS, FP32 external tensors, local saved thermal still:

| Measurement | CPU | Core ML CPU + GPU |
| --- | ---: | ---: |
| Full CustomProcessor, 256×192, median | 50.3 ms | 20.9 ms |
| Full CustomProcessor, 256×192, p95 | 57.9 ms | 24.6 ms |

Worker results use 20 timed runs after three warmup calls. A separate 30-run
preprocess/inference/postprocess benchmark measured GPU median 17.4 ms/p95
19.2 ms. ALL measured 17.3 ms/p95 19.8 ms. These are short still-image probes,
not live camera or whole-viewer frame-rate guarantees. First inference and model
compilation are slower; the 256×192 GPU benchmark's initialization took 1.08 s
and first frame took 0.67 s. The 25 fps camera gives the whole pipeline 40 ms.

The reconstructed Torch generator and portable boundary rewrites exactly matched
the upstream reference on a seeded input. ONNX CPU parity passed combined
absolute/relative tolerance 2e-4; maximum normalized error was 0.0002031 for
256×192. Real worker CPU/GPU maximum difference was 0.0114 of one 0–255 display
level. A compute-plan probe of the 256×256 portable graph placed all 300 supported
nodes in one Core ML partition and reported Apple M4 GPU execution. The native
256×192 profile also contained a single Core ML execution partition.

The model has 45,851,907 parameters; FP32 ONNX files are about 175 MiB. CUDA has
not been tested. Live video temporal stability and generalization to different
TOPDON scenes remain untested. The first sample invents scene textures/details.
Running-statistics inference produces visibly different, darker output on that
sample; the default retains the upstream test script's per-frame statistics.

## Reproduce conversion

Run these commands from this repository. Keep the upstream source and original
checkpoint as local external resources. `export_onnx.py` checks the checkpoint's
SHA-256 and uses `torch.load(..., weights_only=True)`.

```sh
uv sync --dev --group export --group viewer
uv run --group export tools/export_onnx.py \
  --checkpoint artifacts/latest_net_G.pth \
  --upstream /path/to/Thermal2RGB-af47b637376777d2bb8b8d8d48290f04e7bb206d \
  --output artifacts/thermal2rgb-instance-portable-256x192.onnx \
  --height 192 --width 256
```

Omit dimensions for the original 256×256 size. `--mode running` evaluates saved
normalization statistics; `--native-ops` retains original padding/convolution
operators for comparison. Default exports preserve per-frame instance statistics.
For Core ML, reflection is expressed as static positive slices/concatenation;
transposed convolution uses an unpadded convolution followed by a leading crop.
Both transformations preserve border values and trained parameters.

```sh
uv run tools/benchmark.py --model artifacts/thermal2rgb-instance-portable-256x192.onnx \
  --image /path/to/grayscale-thermal.png --output artifacts/benchmark \
  --backend coreml --runs 30

# Viewer package must be available for contract tests and the worker experiment.
uv run --group export --group viewer pytest
uv run --group viewer tools/verify_worker.py \
  --model artifacts/thermal2rgb-instance-portable-256x192.onnx \
  --image /path/to/grayscale-thermal.png --output artifacts/worker
uv run ruff check .
```

`tools/prepare_capture.py` converts a local NPZ's raw-count plane into an
unannotated, min/max-scaled white-hot PNG without changing the capture. All actual
weights/network/GPU experiments are explicit tools outside normal pytest tests.
Tests use fake sessions; optional Torch tests verify boundary rewrites separately.
The optional `viewer` group links the parent TOPDON checkout into this repository's
environment. Conversion alone needs only the `export` group. PyCharm may choose a
different supported Python version for this standalone module; viewer runtime
verification above used its existing Python 3.12 interpreter. The separate
conversion environment's tests also pass on Python 3.14.

## Artifacts and hosting

Source, conversion recipes, provenance and small reports belong in Git.
`artifacts/` retains local checkpoint, ONNX variants, profiles, images and the
machine-specific demo pipeline; large/local artifacts are ignored by Git.
Keep downloaded original weights as well as derived variants.

The [v0.1.0 release](https://github.com/manbehindthemadness/based-gan/releases/tag/v0.1.0)
hosts the original generator checkpoint and three converted ONNX variants:
256×192 instance statistics (default), 256×256 instance statistics, and 256×256
running statistics. `SHA256SUMS.txt`, `provenance.json`, and the conversion reports
record checksums, shapes, normalization and tool versions. Download alternate
ONNX variants and choose them with Browse. The default URL uses the versioned
release rather than a mutable `latest` alias.

## Model source

The generator and pretrained weights come from
[shivom9713/Thermal2RGB](https://github.com/shivom9713/Thermal2RGB/tree/af47b637376777d2bb8b8d8d48290f04e7bb206d).
The original `latest_net_G.pth` is supplied through the upstream README's
[Google Drive pretrained-model link](https://drive.google.com/file/d/1N_vjU2db2HWWsKiQXqWTujR5_XtOEUjQ/view).
Upstream reports training on KAIST Multispectral. These ONNX files preserve that
checkpoint's trained parameters; this project reconstructs its inference graph
and converts its operators for GPU execution. See `provenance.json` for the exact
revision, checkpoint checksum and architecture. Attribution and authorship of the
original model remain with its upstream creators.
