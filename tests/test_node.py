"""
Deterministic Custom contract tests; no weights, network, camera or GPU required.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from topdon_duo.custom_nodes.bundle import load_folder, validate_custom
from topdon_duo.custom_nodes.runtime import CustomProcessor
from topdon_duo.pipeline import default_pipeline, node, validate_pipeline

FOLDER = Path(__file__).parents[1] / "nodes" / "Based GAN"


class FakeSession:
    """
    Return synthetic RGB primaries while recording input tensors and session reuse.
    """

    instances: ClassVar[list[Any]] = []
    bad_output = False
    bad_metadata = False

    def __init__(self, path: str, options: Any, providers: list[Any]) -> None:
        """
        Retain provider choices without loading any file.
        """
        self.path = path
        assert options.intra_op_num_threads == 4
        self.providers = [value if isinstance(value, str) else value[0] for value in providers]
        self.tensor = None
        self.instances.append(self)

    def disable_fallback(self) -> None:
        """
        Match the checked runtime session API.
        """

    def get_providers(self) -> list[str]:
        """
        Expose the selected fake execution providers.
        """
        return self.providers

    @staticmethod
    def get_inputs() -> list[Any]:
        """
        Describe a supported small static input.
        """
        return [SimpleNamespace(name="thermal", shape=[1, 3, 32, 48], type="tensor(float)")]

    @staticmethod
    def get_outputs() -> list[Any]:
        """
        Describe the matching RGB output.
        """
        return [SimpleNamespace(name="rgb", shape=[1, 3, 32, 48], type="tensor(float)")]

    def get_modelmeta(self) -> Any:
        """
        Provide verified converter metadata or a malformed variant.
        """
        metadata = {} if self.bad_metadata else {
            "based_gan_normalization": "instance",
            "based_gan_input": "RGB NCHW float32 [-1,1]",
            "based_gan_output": "RGB NCHW float32 [-1,1]",
        }
        return SimpleNamespace(custom_metadata_map=metadata)

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Emit red RGB pixels, including an optional invalid value.
        """
        self.tensor = feed["thermal"].copy()
        assert names == ["rgb"]
        result = np.full((1, 3, 32, 48), -1, dtype=np.float32)
        result[:, 0] = 1
        if self.bad_output:
            result[0, 0, 0, 0] = np.nan
        return [result]


@pytest.fixture
def setup_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Any, ...]]:
    """
    Load the real folder and fake only the external model/session boundary.
    """
    FakeSession.instances = []
    FakeSession.bad_output = False
    FakeSession.bad_metadata = False
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    monkeypatch.setattr("onnxruntime.get_available_providers",
                        lambda: ["CPUExecutionProvider", "CoreMLExecutionProvider"])
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model; session boundary is mocked")
    params = load_folder(FOLDER)
    config = json.loads(params["config"])
    config["model_path"] = str(model)
    config["inference"]["backend"] = "cpu"
    params["config"] = json.dumps(config)
    validate_custom(json.loads(json.dumps(params)))
    item = node("software", "custom", **params)
    document = default_pipeline()
    document["software"] = [document["software"][0], item, node("software", "output")]
    document = validate_pipeline(json.loads(json.dumps(document)))
    item = document["software"][1]
    processor = CustomProcessor(apple_available=True, nvidia_available=False)
    yield processor, item, config
    processor.close()


def update(item: dict[str, Any], config: dict[str, Any]) -> None:
    """
    Simulate configuration edits without reloading the package.
    """
    item["params"]["config"] = json.dumps(config)


def test_color_geometry_normalization_and_blend(setup_node: tuple[Any, ...]) -> None:
    """
    Preserve BGR order, float32 geometry, input ownership and config updates.
    """
    processor, item, config = setup_node
    image = np.full((17, 29, 3), 127.5, dtype=np.float32)
    result = processor.apply(image, item)
    assert result.shape == image.shape and result.dtype == np.float32
    assert np.allclose(result, (0, 0, 255))
    assert np.all(image == 127.5)
    session = FakeSession.instances[0]
    assert session.tensor.shape == (1, 3, 32, 48)
    assert session.tensor.dtype == np.float32 and session.tensor.flags.c_contiguous
    assert np.allclose(session.tensor, 0, atol=2e-7)
    config["blend"] = 0.5
    update(item, config)
    assert np.allclose(processor.apply(image, item), (63.75, 63.75, 191.25))
    assert len(FakeSession.instances) == 1


def test_rgb_inversion_and_grayscale(setup_node: tuple[Any, ...]) -> None:
    """
    Honor model input channel order and intensity polarity.
    """
    processor, item, config = setup_node
    image = np.empty((12, 18, 3), dtype=np.float32)
    image[:] = (0, 127.5, 255)
    config.update(input_mode="rgb", invert_input=True)
    update(item, config)
    processor.apply(image, item)
    tensor = FakeSession.instances[0].tensor
    assert np.allclose(tensor[0, :, 0, 0], (-1, 0, 1))
    config.update(input_mode="grayscale", invert_input=False)
    update(item, config)
    processor.apply(image, item)
    tensor = FakeSession.instances[0].tensor
    assert np.array_equal(tensor[:, 0], tensor[:, 1])
    assert np.array_equal(tensor[:, 1], tensor[:, 2])


def test_device_changes_and_independent_nodes(setup_node: tuple[Any, ...]) -> None:
    """
    Rebuild for device changes and isolate identical node sessions.
    """
    processor, item, config = setup_node
    image = np.zeros((8, 8, 3), dtype=np.float32)
    processor.apply(image, item)
    config["inference"]["backend"] = "coreml"
    update(item, config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 2
    assert "CoreMLExecutionProvider" in FakeSession.instances[-1].providers
    second = node("software", "custom", **item["params"])
    processor.apply(image, second)
    assert len(FakeSession.instances) == 3
    config["model_path"] = FakeSession.instances[0].path
    config["inference"]["apple_compute"] = "ALL"
    update(item, config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 4
    other_model = Path(config["model_path"]).with_name("second.onnx")
    other_model.write_bytes(b"second mocked model")
    config["model_path"] = str(other_model)
    update(item, config)
    processor.apply(image, item)
    assert len(FakeSession.instances) == 5


def test_pending_model_passes_through(
    setup_node: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Poll the shared downloader without creating a session or changing display pixels.
    """
    processor, item, config = setup_node
    config["model_path"] = ""
    assert config["model"]["url"].endswith(
        "/v0.1.0/thermal2rgb-instance-portable-256x192.onnx"
    )
    assert config["model"]["max_bytes"] == 200_000_000
    update(item, config)

    def pending(_self: Any, _source: Any) -> None:
        """
        Represent a shared download still in progress.
        """
        assert _source == config["model"]

    monkeypatch.setattr("topdon_duo.custom_nodes.models.ModelResolver.resolve", pending)
    image = np.full((8, 8, 3), 42, dtype=np.float32)
    assert np.array_equal(processor.apply(image, item), image)
    assert not FakeSession.instances


@pytest.mark.parametrize("kind, message", [
    ("metadata", "Use a static float32"), ("output", "invalid RGB tensor"),
    ("missing-model", "Browse control"), ("blend", "finite bounded numbers"),
])
def test_informative_errors(setup_node: tuple[Any, ...], kind: str, message: str) -> None:
    """
    Report malformed schemas, outputs and settings through the Custom error path.
    """
    processor, item, config = setup_node
    FakeSession.bad_metadata = kind == "metadata"
    FakeSession.bad_output = kind == "output"
    if kind == "missing-model":
        config["model_path"] = ""
        config.pop("model")
    if kind == "blend":
        config["blend"] = True
    update(item, config)
    with pytest.raises(ValueError, match=message):
        processor.apply(np.zeros((8, 8, 3), dtype=np.float32), item)
