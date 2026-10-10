"""
Benchmark external ONNX weights on a still image without opening the camera.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


def main() -> None:
    """
    Measure preprocessing/inference/postprocessing and save provider parity evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "coreml", "cuda"), default="cpu")
    parser.add_argument("--compute", default="CPUAndGPU")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--profile-compute", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Cannot read input image")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 1 if args.profile_compute else 3
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output / args.backend)
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                "coreml": "CoreMLExecutionProvider"}[args.backend]
    if provider not in ort.get_available_providers():
        raise ValueError(f"Unavailable provider: {provider}")
    selected: list[Any] = (
        [provider, "CPUExecutionProvider"] if args.backend != "cpu" else [provider]
    )
    if args.backend == "coreml":
        selected[0] = (provider, {
            "ModelFormat": "MLProgram", "MLComputeUnits": args.compute,
            "RequireStaticInputShapes": "1", "ProfileComputePlan": str(int(args.profile_compute)),
        })
    started = time.perf_counter()
    session = ort.InferenceSession(str(args.model), options, providers=selected)
    session.disable_fallback()
    if provider not in session.get_providers():
        raise ValueError("Requested GPU provider failed to initialize")
    initialization = time.perf_counter() - started
    model_input = session.get_inputs()[0]
    _, channels, height, width = model_input.shape
    if channels != 3 or not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("Expected a static RGB NCHW model")
    durations, inference_durations = [], []
    output = None
    bgr = None
    first = 0.0
    for index in range(args.runs + 3):
        began = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32)
        tensor = np.ascontiguousarray((rgb.transpose(2, 0, 1)[None] / 127.5) - 1)
        infer_began = time.perf_counter()
        outputs = session.run(None, {model_input.name: tensor})
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise ValueError("Expected one dense RGB output")
        output = np.asarray(outputs[0])
        inference = time.perf_counter() - infer_began
        bgr = np.clip((output[0].transpose(1, 2, 0)[:, :, ::-1] + 1) * 127.5, 0, 255)
        bgr = cv2.resize(bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        elapsed = time.perf_counter() - began
        if index == 0:
            first = elapsed
        if index >= 3:
            durations.append(elapsed * 1000)
            inference_durations.append(inference * 1000)
    if output is None or bgr is None or not np.isfinite(output).all():
        raise ValueError("Invalid inference output")
    np.save(args.output / f"{args.backend}-{args.compute}.npy", output)
    cv2.imwrite(str(args.output / f"{args.backend}-{args.compute}.png"), bgr.astype(np.uint8))
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text())
    providers = Counter(event.get("args", {}).get("provider") for event in events
                        if event.get("cat") == "Node")
    report = {
        "backend": args.backend, "compute": args.compute, "runtime": ort.__version__,
        "input_shape": model_input.shape, "providers": dict(providers),
        "initialization_seconds": initialization, "first_frame_seconds": first,
        "runs": args.runs, "total_median_ms": float(np.median(durations)),
        "total_p95_ms": float(np.percentile(durations, 95)),
        "inference_median_ms": float(np.median(inference_durations)),
        "profile": str(profile_path), "model": str(args.model), "image": str(args.image),
    }
    (args.output / f"{args.backend}-{args.compute}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
