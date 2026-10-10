"""
Export a verified local Thermal2RGB checkpoint; never install or download at runtime.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from generator import Generator, prepare_export

CHECKPOINT_SHA256 = "6de0c19446cc5edb2e6b38507dcb94c50267c88f1a3a39acb71fe48efd5930ef"


def digest(path: Path) -> str:
    """
    Calculate an artifact checksum without reading the entire file into memory.
    """
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    return value


def upstream_reference(source: Path, state: dict, mode: str) -> torch.nn.Module:
    """
    Instantiate explicitly provided upstream code to verify reconstruction parity.
    """
    spec = importlib.util.spec_from_file_location("thermal2rgb_reference", source / "models/networks.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied upstream networks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.LocalEnhancer(
        3, 3, 32, n_downsample_global=3, n_blocks_global=9,
        n_local_enhancers=1, n_blocks_local=3,
        norm_layer=module.get_norm_layer("instance"),
    )
    reference.load_state_dict(state, strict=True)
    reference.train(mode == "instance")
    return reference


def main() -> None:
    """
    Verify weights, reconstruction, graph rewrites and ONNX numerical parity.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("instance", "running"), default="instance")
    parser.add_argument("--native-ops", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if min(args.height, args.width) < 32 or args.height % 16 or args.width % 16:
        parser.error("Dimensions must be multiples of 16 and at least 32")
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum does not match the investigated upstream weights")
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Generator()
    model.load_state_dict(state, strict=True)
    prepare_export(model, args.mode, not args.native_ops)
    model.eval()
    reference = upstream_reference(args.upstream, state, args.mode)
    random = np.random.default_rng(9713)
    input_array = random.uniform(-1, 1, (1, 3, args.height, args.width)).astype(np.float32)
    tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        expected = reference(tensor).numpy()
        actual = model(tensor).numpy()
    reconstruction_error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=2e-5, rtol=2e-5):
        raise ValueError(f"Reconstructed/rewritten model differs: {reconstruction_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        model, (tensor,), str(args.output), input_names=["thermal"], output_names=["rgb"],
        opset_version=17, dynamo=False, external_data=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    metadata = {
        "based_gan_normalization": args.mode,
        "based_gan_checkpoint_sha256": CHECKPOINT_SHA256,
        "based_gan_input": "RGB NCHW float32 [-1,1]",
        "based_gan_output": "RGB NCHW float32 [-1,1]",
    }
    onnx.helper.set_model_props(graph, metadata)
    onnx.save(graph, args.output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"thermal": input_array})
    if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
        raise ValueError("ONNX CPU reference returned an invalid tensor")
    predicted = np.asarray(outputs[0])
    maximum_error = float(np.max(np.abs(predicted - expected)))
    if not np.allclose(predicted, expected, atol=2e-4, rtol=2e-4):
        raise ValueError(f"ONNX CPU parity failed: {maximum_error}")
    report = {
        "checkpoint_sha256": CHECKPOINT_SHA256, "onnx_sha256": digest(args.output),
        "onnx_bytes": args.output.stat().st_size,
        "mode": args.mode, "portable": not args.native_ops,
        "input_shape": list(input_array.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reconstruction_max_error": reconstruction_error, "onnx_max_error": maximum_error,
        "export_and_verify_seconds": time.perf_counter() - started,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__},
        "operators": sorted({item.op_type for item in graph.graph.node}),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
