import torch
from tqdm import tqdm

from src.metrics.segmentation_metrics import SegmentationMetrics

def validate_one_epoch(
        model,
        dataloader,
        criterion,
        device,
        num_classes,
        ignore_index=255,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, masks)

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
            })
    
    avg_loss = total_loss / max(num_batches, 1)

    result = metrics.compute()
    result["loss"] = avg_loss

    return result
