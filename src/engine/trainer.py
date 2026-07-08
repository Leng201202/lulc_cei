import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler=None,
    mixed_precision=False,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()

        use_amp = (
            mixed_precision
            and scaler is not None
            and device.type == "cuda"
        )

        if use_amp:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / max(num_batches, 1)

    return average_loss