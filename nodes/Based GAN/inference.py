"""
Validate exported generator tensors and retain a worker-local ONNX session.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from topdon_duo.custom_nodes.devices import onnx_providers, validate_device


class Translator:
    """
    Own one session for a model path and effective execution preference.
    """

    def __init__(self, path: Path, device: dict[str, str]) -> None:
        """
        Reject unsupported schemas and failed GPU initialization explicitly.
        """
        settings = validate_device(device)
        self.key = (str(path), settings["backend"], settings["apple_compute"])
        providers = onnx_providers(settings)
        names = {provider if isinstance(provider, str) else provider[0]
                 for provider in providers}
        if names - set(ort.get_available_providers()):
            raise ValueError("Requested ONNX execution provider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.session.disable_fallback()
        if names - set(self.session.get_providers()):
            raise ValueError("GPU initialization failed; select CPU to diagnose this model")
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        metadata = self.session.get_modelmeta().custom_metadata_map
        if (len(inputs) != 1 or len(outputs) != 1
                or inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4 or inputs[0].shape[:2] != [1, 3]
                or outputs[0].shape != inputs[0].shape
                or any(type(size) is not int or size < 32 or size % 16
                       for size in inputs[0].shape[2:])
                or inputs[0].shape[2] * inputs[0].shape[3] > 4_000_000
                or metadata.get("based_gan_normalization") not in ("instance", "running")
                or metadata.get("based_gan_input") != "RGB NCHW float32 [-1,1]"
                or metadata.get("based_gan_output") != "RGB NCHW float32 [-1,1]"):
            raise ValueError("Use a static float32 Based GAN export produced by tools/export_onnx.py")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.shape = tuple(inputs[0].shape)

    def translate(self, image: np.ndarray, mode: str, inverted: bool) -> np.ndarray:
        """
        Normalize RGB/grayscale input and map synthetic RGB back to BGR display pixels.
        """
        resized = cv2.resize(
            np.clip(image, 0, 255).astype(np.float32),
            (self.shape[3], self.shape[2]), interpolation=cv2.INTER_CUBIC,
        )
        if mode == "grayscale":
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = np.clip(rgb, 0, 255)
        if inverted:
            rgb = 255 - rgb
        tensor = np.ascontiguousarray((rgb.transpose(2, 0, 1)[None] / 127.5) - 1)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
            raise ValueError("Generator returned an invalid RGB tensor")
        result = np.asarray(outputs[0])
        if result.shape != self.shape or not np.isfinite(result).all():
            raise ValueError("Generator returned an invalid RGB tensor")
        bgr = (result[0].transpose(1, 2, 0)[:, :, ::-1] + 1) * 127.5
        restored = cv2.resize(
            bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC,
        )
        output = np.clip(restored, 0, 255).astype(np.float32)
        return output
