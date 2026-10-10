"""
Reconstruct the checkpoint's global/local generator for isolated ONNX conversion.

Tensor names/layout follow Thermal2RGB af47b637; see provenance.json.
"""

import torch
from torch import nn


def normalized_conv(source: int, target: int, *, stride: int = 1) -> list[nn.Module]:
    """
    Build a three-pixel convolution with tracked instance normalization.
    """
    return [nn.Conv2d(source, target, 3, stride, 1),
            nn.InstanceNorm2d(target, track_running_stats=True), nn.ReLU()]


class Residual(nn.Module):
    """
    Preserve the residual checkpoint's nested convolution names.
    """

    def __init__(self, channels: int) -> None:
        """
        Build two reflected convolutions and one residual addition.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, track_running_stats=True), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, track_running_stats=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add the learned residual to its input.
        """
        result = image + self.conv_block(image)
        return result


def opening(channels: int) -> list[nn.Module]:
    """
    Build the RGB input projection, keeping checkpoint indices intact.
    """
    return [nn.ReflectionPad2d(3), nn.Conv2d(3, channels, 7),
            nn.InstanceNorm2d(channels, track_running_stats=True), nn.ReLU()]


def upsample(source: int, target: int) -> list[nn.Module]:
    """
    Double spatial dimensions with the checkpoint's transposed convolution.
    """
    return [nn.ConvTranspose2d(source, target, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(target, track_running_stats=True), nn.ReLU()]


class Generator(nn.Module):
    """
    Reconstruct the verified 32-filter, one-local-stage gll generator.
    """

    def __init__(self) -> None:
        """
        Keep state dictionary names compatible with latest_net_G.pth.
        """
        super().__init__()
        global_layers = opening(64)
        for source, target in ((64, 128), (128, 256), (256, 512)):
            global_layers.extend(normalized_conv(source, target, stride=2))
        global_layers.extend(Residual(512) for _ in range(9))
        for source, target in ((512, 256), (256, 128), (128, 64)):
            global_layers.extend(upsample(source, target))
        self.model = nn.Sequential(*global_layers)
        self.model1_1 = nn.Sequential(*opening(32), *normalized_conv(32, 64, stride=2))
        self.model1_2 = nn.Sequential(
            *(Residual(64) for _ in range(3)), *upsample(64, 32),
            nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7), nn.Tanh(),
        )
        self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Fuse the half-resolution global features with the local encoder.
        """
        global_features = self.model(self.downsample(image))
        output = self.model1_2(self.model1_1(image) + global_features)
        return output


class StaticReflection(nn.Module):
    """
    Express reflection with positive static slices and concatenation for Core ML.
    """

    def __init__(self, padding: int) -> None:
        """
        Store a symmetric spatial reflection width.
        """
        super().__init__()
        self.padding = padding

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Reflect individual boundary rows/columns without Pad or reverse Slice.
        """
        p = self.padding
        width = image.shape[3]
        columns = [image[:, :, :, i:i + 1] for i in range(p, 0, -1)]
        columns += [image]
        columns += [image[:, :, :, width - i - 1:width - i] for i in range(1, p + 1)]
        reflected = torch.cat(columns, dim=3)
        height = reflected.shape[2]
        rows = [reflected[:, :, i:i + 1, :] for i in range(p, 0, -1)]
        rows += [reflected]
        rows += [reflected[:, :, height - i - 1:height - i, :] for i in range(1, p + 1)]
        result = torch.cat(rows, dim=2)
        return result


class CroppedTranspose(nn.Module):
    """
    Preserve asymmetric transpose-convolution output using an unpadded op and crop.
    """

    def __init__(self, original: nn.ConvTranspose2d) -> None:
        """
        Retain trained parameters and replace padding/output-padding with a crop.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            original.in_channels, original.out_channels, 3, 2, 0, output_padding=0,
        )
        self.conv.weight = original.weight
        self.conv.bias = original.bias

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Crop one leading row/column from the full convolution result.
        """
        result = self.conv(image)[:, :, 1:, 1:]
        return result


def prepare_export(module: nn.Module, mode: str, portable: bool) -> None:
    """
    Choose normalization semantics and optionally rewrite Core ML boundary ops.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.InstanceNorm2d) and mode == "instance":
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None
        if portable and isinstance(child, nn.ReflectionPad2d):
            module.add_module(name, StaticReflection(child.padding[0]))
        elif portable and isinstance(child, nn.ConvTranspose2d):
            module.add_module(name, CroppedTranspose(child))
        else:
            prepare_export(child, mode, portable)
