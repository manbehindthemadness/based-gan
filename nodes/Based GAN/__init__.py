# Folder names become display labels; runtime assigns the Python module name.
# ruff: noqa: N999
"""
Synthetic thermal-to-RGB display translation using external ONNX weights.
"""

from pathlib import Path
from typing import Any

import numpy as np
from topdon_duo.custom_nodes.models import ModelResolver

from .inference import Translator

CONFIG_JSON = {
    "defaults": {
        "model_path": "",
        "inference": {"backend": "coreml", "apple_compute": "CPUAndGPU"},
        "input_mode": "grayscale",
        "invert_input": False,
        "blend": 1.0,
    },
    "controls": [
        {"key": "model_path", "label": "ONNX model", "type": "model"},
        {"key": "inference", "label": "Inference device", "type": "device"},
        {"key": "input_mode", "label": "Input", "type": "choice",
         "options": ["grayscale", "rgb"]},
        {"key": "invert_input", "label": "Invert model input", "type": "boolean"},
        {"key": "blend", "label": "Generated image strength", "type": "number",
         "min": 0, "max": 1, "step": 0.05},
    ],
}

_models = ModelResolver()
_translator: Translator | None = None

MODEL_SOURCE = {
    "name": "Based GAN Thermal2RGB 256x192 (instance)",
    "url": "https://github.com/manbehindthemadness/based-gan/releases/download/v0.1.0/thermal2rgb-instance-portable-256x192.onnx",
    "sha256": "797620eb59a06d9a234c827fc15554f4e3a7a4a4bf2f9086172b3971c4c0bf99",
    "max_bytes": 200_000_000,
}


def process(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Translate display pixels, preserving geometry and the viewer's radiometry.

    Use the pinned hosted model by default, or override it through the model control.
    """
    global _translator
    blend = config.get("blend", 1.0)
    mode = config.get("input_mode", "grayscale")
    inverted = config.get("invert_input", False)
    if (isinstance(blend, bool) or not isinstance(blend, (int, float))
            or not np.isfinite(blend) or not 0 <= blend <= 1):
        raise ValueError("Generated image strength must be between 0 and 1")
    if mode not in ("grayscale", "rgb") or type(inverted) is not bool:
        raise ValueError("Invalid model input settings")
    if blend == 0:
        return image
    local = config.get("model_path", "")
    if not isinstance(local, str):
        raise TypeError("Local model path must be a string")
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Selected local ONNX model does not exist")
    else:
        source = config.get("model")
        if source is None:
            raise ValueError("Choose a Based GAN ONNX export with the ONNX model Browse control")
        if not isinstance(source, (str, dict)):
            raise TypeError("Model source must be a catalog name or checksum-pinned descriptor")
        path = _models.resolve(source)
        if path is None:
            return image
    device = config["inference"]
    key = (str(path), device["backend"], device["apple_compute"])
    translator = _translator
    if translator is None or translator.key != key:
        translator = Translator(path, device)
        _translator = translator
    generated = translator.translate(image, mode, inverted)
    output = generated * blend + image * (1 - blend)
    return output
