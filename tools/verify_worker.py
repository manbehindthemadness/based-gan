"""
Exercise the real Custom worker using local weights and write a loadable demo preset.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from topdon_duo.custom_nodes.bundle import load_folder
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline


def main() -> None:
    """
    Validate embedded-package portability, geometry, session reuse and actual GPU output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).parents[1] / "nodes" / "Based GAN"
    params = load_folder(folder)
    config = json.loads(params["config"])
    config["model_path"] = str(args.model.resolve())
    params["config"] = json.dumps(config)
    item = node("software", "custom", **params)
    document = default_pipeline()
    document["hardware"] = []
    document["software"][0]["params"]["source"] = "raw"
    document["software"] = [
        document["software"][0], node("software", "range", low=15.0, high=40.0),
        item, node("software", "output"),
    ]
    document = validate_pipeline(json.loads(json.dumps(document)))
    (args.output / "based-gan-demo.json").write_text(json.dumps(document, indent=2) + "\n")
    item = document["software"][2]
    loaded = cv2.imread(str(args.image))
    if loaded is None:
        raise ValueError("Cannot read test display image")
    image = loaded.astype(np.float32)
    report, results = {}, {}
    for backend in ("cpu", "coreml"):
        config["inference"]["backend"] = backend
        item["params"]["config"] = json.dumps(config)
        processor = CustomProcessor(apple_available=True, nvidia_available=False)
        try:
            durations = []
            output = None
            for index in range(23):
                started = time.perf_counter()
                output = processor.apply(image, item)
                elapsed = (time.perf_counter() - started) * 1000
                if index >= 3:
                    durations.append(elapsed)
            if output is None or output.shape != image.shape or output.dtype != np.float32:
                raise ValueError("Worker output lost display geometry/dtype")
            results[backend] = output
            report[backend] = {"median_ms": float(np.median(durations)),
                               "p95_ms": float(np.percentile(durations, 95)),
                               "shape": list(output.shape)}
            cv2.imwrite(str(args.output / f"worker-{backend}.png"), output.astype(np.uint8))
        finally:
            processor.close()
    difference = np.abs(results["cpu"] - results["coreml"])
    report["cpu_gpu_difference_display_levels"] = {
        "maximum": float(difference.max()), "mean": float(difference.mean()),
        "p99": float(np.percentile(difference, 99)),
    }
    (args.output / "worker-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
