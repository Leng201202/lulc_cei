"""End-to-end smoke test using synthetic data (no dataset or weights needed).

Exercises the pieces that have runtime behavior but no dataset dependency:
the metrics, the visualization palette helpers, the loss forward pass, and --
when ``segmentation_models_pytorch`` is installed -- model construction plus
both the full-image and tiled inference paths in ``predict.py``.

Run:
    python tests/smoke_test.py

Model-dependent checks are skipped (not failed) when smp is unavailable, so
the test still gives useful signal on a partial install. Exit code is non-zero
only on an actual failure.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils import visualization as viz

# The loss and model factories import segmentation_models_pytorch at module
# load, so those tests are gated on it being installed.
try:
    import segmentation_models_pytorch  # noqa: F401
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

NUM_CLASSES = 8
IGNORE_INDEX = 255

passed = 0
failed = 0
skipped = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


def skip(name, reason):
    global skipped
    skipped += 1
    print(f"SKIP: {name} ({reason})")


def base_config():
    return {
        "experiment": {"output_dir": "./experiments/_smoke"},
        "dataset": {"num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX},
        "model": {
            "name": "unet",
            "encoder_name": "efficientnet-b4",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": NUM_CLASSES,
        },
        "training": {"loss": "ce_dice"},
    }


def test_metrics():
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    targets = np.zeros((1, 16, 16), dtype=np.int64)
    targets[0, :8] = 1
    preds = targets.copy()
    targets[0, 0, 0] = IGNORE_INDEX  # ignored pixel must not affect the score
    metrics.update(preds, targets)
    result = metrics.compute()
    check("metrics: perfect prediction -> OA == 1.0", result["OA"] == 1.0)
    check("metrics: perfect prediction -> mIoU == 1.0", result["mIoU"] == 1.0)
    check(
        "metrics: unseen classes reported as None",
        result["per_class_iou"][7] is None,
    )


def test_visualization(tmp_dir):
    mask = np.array([[0, 7], [255, 3]], dtype=np.uint8)
    rgb = viz.decode_mask(mask, ignore_index=IGNORE_INDEX)
    check("viz: decode_mask shape", rgb.shape == (2, 2, 3))
    check("viz: class 0 -> Bareland color", tuple(rgb[0, 0]) == (128, 0, 0))
    check("viz: ignore -> black", tuple(rgb[1, 0]) == (0, 0, 0))

    tensor = torch.rand(3, 8, 8)  # simulate a normalized CHW image tensor
    denorm = viz.denormalize_image(tensor)
    check("viz: denormalize -> uint8 HWC", denorm.shape == (8, 8, 3) and denorm.dtype == np.uint8)

    # save_prediction concatenates image and masks side by side, so their
    # heights must match; use an 8x8 mask to pair with the 8x8 image tensor.
    panel_mask = np.random.randint(0, NUM_CLASSES, size=(8, 8)).astype(np.uint8)
    out = tmp_dir / "panel.png"
    viz.save_prediction(str(out), tensor, panel_mask, ground_truth=panel_mask, ignore_index=IGNORE_INDEX)
    check("viz: save_prediction writes a file", out.exists())


def test_loss():
    if not HAS_SMP:
        skip("loss forward", "segmentation_models_pytorch not installed")
        return

    from src.losses.loss_factory import build_loss

    criterion = build_loss(base_config())
    logits = torch.randn(2, NUM_CLASSES, 16, 16, requires_grad=True)
    target = torch.randint(0, NUM_CLASSES, (2, 16, 16))
    target[0, 0, 0] = IGNORE_INDEX
    loss = criterion(logits, target)
    loss.backward()
    check("loss: scalar output", loss.dim() == 0)
    check("loss: finite and non-negative", torch.isfinite(loss) and loss.item() >= 0)
    check("loss: gradient flows", logits.grad is not None)


class DummySegModel(torch.nn.Module):
    """Minimal FCN stand-in so the inference paths can be tested without smp.

    A single 1x1 conv maps ``in_channels -> num_classes`` while preserving the
    spatial size, which is all ``predict_full`` / ``predict_tiled`` rely on.
    """

    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.head = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)


def test_model_factory():
    if not HAS_SMP:
        skip("model factory", "segmentation_models_pytorch not installed")
        return

    from src.models.model_factory import build_model

    model = build_model(base_config())
    out = model(torch.randn(1, 3, 64, 64))
    check("model factory: output shape", out.shape == (1, NUM_CLASSES, 64, 64))


def test_inference(tmp_dir):
    import predict

    device = torch.device("cpu")
    model = DummySegModel().to(device).eval()
    transform = predict.build_transform(base_config())

    # A non-multiple-of-32 size verifies reflect-padding and crop-back.
    image = (np.random.rand(100, 120, 3) * 255).astype(np.uint8)

    full = predict.predict_full(model, image, transform, device)
    check("predict: full-image mask shape matches input", full.shape == (100, 120))
    check("predict: full-image classes in range", full.max() < NUM_CLASSES)

    tiled = predict.predict_tiled(
        model, image, transform, device,
        tile_size=64, overlap=16, num_classes=NUM_CLASSES,
    )
    check("predict: tiled mask shape matches input", tiled.shape == (100, 120))
    check("predict: tiled classes in range", tiled.max() < NUM_CLASSES)

    out = tmp_dir / "pred.png"
    viz.save_mask(str(out), tiled, ignore_index=IGNORE_INDEX)
    check("predict: save_mask writes a file", out.exists())


def main():
    import tempfile

    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        test_metrics()
        test_visualization(tmp_dir)
        test_loss()
        test_model_factory()
        test_inference(tmp_dir)

    print("-" * 50)
    print(f"Result: {passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
