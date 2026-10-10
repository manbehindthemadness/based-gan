"""
Verify Core ML boundary rewrites against independent native Torch operators.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch", reason="Optional isolated conversion dependency")
spec = importlib.util.spec_from_file_location(
    "based_gan_generator", Path(__file__).parents[1] / "tools" / "generator.py",
)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.mark.parametrize("padding", [1, 3])
@pytest.mark.parametrize("shape", [(1, 3, 32, 48), (1, 5, 48, 32)])
def test_reflected_boundary_rewrite(padding: int, shape: tuple[int, ...]) -> None:
    """
    Match reflection at every pixel, including corner behavior.
    """
    image = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    reference = torch.nn.ReflectionPad2d(padding)(image)
    rewritten = generator.StaticReflection(padding)(image)
    assert torch.equal(reference, rewritten)


@pytest.mark.parametrize("shape", [(1, 3, 16, 24), (1, 3, 24, 16)])
def test_asymmetric_transpose_rewrite(shape: tuple[int, ...]) -> None:
    """
    Preserve right/bottom convolution contributions when removing output_padding.
    """
    torch.manual_seed(9713)
    original = torch.nn.ConvTranspose2d(3, 5, 3, 2, 1, output_padding=1)
    image = torch.randn(shape)
    reference = original(image)
    rewritten = generator.CroppedTranspose(original)(image)
    assert torch.equal(reference, rewritten)
