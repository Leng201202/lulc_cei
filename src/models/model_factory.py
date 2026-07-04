import segmentation_models_pytorch as smp


def build_model(config):
    model_config = config["model"]

    model_name = model_config["name"]
    encoder_name = model_config["encoder_name"]
    encoder_weights = model_config["encoder_weights"]
    in_channels = model_config["in_channels"]
    num_classes = model_config["num_classes"]

    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
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