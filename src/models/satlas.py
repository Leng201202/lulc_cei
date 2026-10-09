"""Satlas-pretrained Swin-V2-B backbone behind the UPerNet decoder.

Every other model in this project starts from ImageNet weights, which were
learned on ground-level photographs. Satlas-Pretrain's ``aerial`` checkpoint is
a Swin-V2-B trained on high-resolution nadir aerial RGB (NAIP, sub-metre GSD) --
the same viewpoint, band set and rough scale as the CEI tiles -- so it is the
one readily available backbone whose pretraining domain matches the target.

Redistributed by ``torchgeo/satlas`` on the Hugging Face Hub under Apache-2.0;
``tools/fetch_satlas.py`` downloads the file. The weights are a plain
``torchvision.models.swin_v2_b`` state_dict (see that repo's ``convert.py``),
not a timm checkpoint, so they cannot go through smp's ``tu-`` encoder path.

Swin-V2-B's stage widths (128/256/512/1024) are identical to the Swin-B used by
``upernet_swinb``, so the decoder, its channel counts and the segmentation head
are unchanged -- swapping the two configs isolates the effect of the
pretraining corpus and nothing else.
"""

import torch
import torch.nn as nn
import torchvision
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder


class SatlasSwinV2Encoder(nn.Module):
    """torchvision Swin-V2-B exposing smp's encoder feature-pyramid contract.

    smp expects ``depth + 1`` feature maps at strides 1, 2, 4, 8, 16, 32 and an
    ``out_channels`` list to match. Swin produces nothing at stride 2, so that
    slot is a zero-channel placeholder -- exactly what smp's own timm Swin
    encoder reports, which is why the UPerNet decoder already tolerates it.
    """

    # Stage widths of swin_v2_b, with the stride-1 input and the empty stride-2
    # level prepended.
    OUT_CHANNELS = (3, 0, 128, 256, 512, 1024)

    def __init__(self, weights_path=None, depth=5):
        super().__init__()
        if depth != 5:
            raise ValueError(f"SatlasSwinV2Encoder supports depth=5 only, got {depth}")

        backbone = torchvision.models.swin_v2_b()
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            # The published file carries the full classifier; the pyramid we use
            # stops at `features`, so `norm`/`head` are expected leftovers.
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            unexpected = [k for k in unexpected if not k.startswith(("norm.", "head."))]
            if missing or unexpected:
                raise RuntimeError(
                    "Satlas weights do not match torchvision swin_v2_b. "
                    f"missing={missing[:5]} unexpected={unexpected[:5]}"
                )
        # Only the hierarchical stages are needed; the pooled classifier is not.
        self.features = backbone.features
        self.out_channels = list(self.OUT_CHANNELS)
        self.output_stride = 32

    def forward(self, x):
        # torchvision's Swin carries activations as NHWC and permutes only at
        # the very end, so every level is transposed back for the decoder.
        empty = x.new_zeros((x.shape[0], 0, x.shape[2] // 2, x.shape[3] // 2))
        features = [x, empty]
        out = x
        for index, stage in enumerate(self.features):
            out = stage(out)
            # features = [embed, stage1, merge, stage2, merge, stage3, merge, stage4]
            if index % 2 == 1:
                features.append(out.permute(0, 3, 1, 2).contiguous())
        return features


class SatlasUPerNet(SegmentationModel):
    """UPerNet on the Satlas aerial Swin-V2-B encoder.

    Composed directly rather than via ``smp.UPerNet`` because that constructor
    resolves its encoder through ``get_encoder``, which only knows timm and
    smp's own registry. The decoder and head are smp's, unmodified.
    """

    def __init__(self, num_classes, weights_path=None, decoder_channels=256, in_channels=3):
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                f"Satlas aerial weights are RGB-only, got in_channels={in_channels}"
            )
        self.encoder = SatlasSwinV2Encoder(weights_path=weights_path)
        self.decoder = UPerNetDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=5,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4,
        )
        self.classification_head = None
        self.name = "upernet-satlas_swinv2_b"
        # Initializes decoder and head only; the encoder keeps its Satlas weights.
        self.initialize()
