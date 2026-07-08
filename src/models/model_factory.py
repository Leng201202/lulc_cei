import segmentation_models_pytorch as smp
import torch.nn as nn


class LeadingChannelDrop(nn.Module):
    """Adapt a model whose head has extra leading output channels.

    The OpenEarthMap-SAR baseline U-Net predicts 9 classes where index 0 is the
    ``background``/``unknown`` class and indices 1-8 are the eight land-cover
    classes. This project trains on those eight classes as indices 0-7, so we
    drop the leading channel(s); the remaining logits line up 1:1 with the
    training class order and every downstream component (loss, argmax, metrics,
    palette) can treat this as a plain N-class model.

    The wrapped model is exposed as ``self.inner`` so the original,
    unprefixed checkpoint can be loaded straight into it.
    """

    def __init__(self, inner, drop):
        super().__init__()
        if drop < 0:
            raise ValueError(f"drop must be non-negative, got {drop}")
        self.inner = inner
        self.drop = drop

    def forward(self, x):
        logits = self.inner(x)
        return logits[:, self.drop:, :, :]


def build_model(config):
    model_config = config["model"]

    model_name = model_config["name"].lower()
    encoder_name = model_config["encoder_name"]
    encoder_weights = model_config["encoder_weights"]
    in_channels = model_config["in_channels"]
    num_classes = model_config["num_classes"]

    # Optional: some pretrained checkpoints (e.g. OpenEarthMap-SAR) were trained
    # with a larger head that includes a leading background class and with SCSE
    # decoder attention. These must be declared so the architecture matches the
    # weights exactly.
    decoder_attention_type = model_config.get("decoder_attention_type")
    pretrained_classes = model_config.get("pretrained_classes")

    if model_name in {"unet", "u-net"}:
        head_classes = pretrained_classes or num_classes
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=head_classes,
            decoder_attention_type=decoder_attention_type,
        )
        if pretrained_classes is not None:
            model = LeadingChannelDrop(model, drop=pretrained_classes - num_classes)
        return model

    if model_name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        return model
    raise ValueError(f"Unknown model name: {model_name}")
