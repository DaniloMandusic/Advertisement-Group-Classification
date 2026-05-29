import copy
import math
import torch


def _get_batch(train_loader, device):
    batch = next(iter(train_loader))
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )


def sanity_check_output_shape(model, train_loader, num_classes, device="cpu"):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, _ = _get_batch(train_loader, device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    expected = (input_ids.shape[0], num_classes)
    actual   = tuple(logits.shape)

    if actual != expected:
        raise RuntimeError(
            f"Output shape mismatch: expected {expected}, got {actual}. "
            f"Check num_classes or classifier head."
        )

    print(f"[OK] Output shape: {actual}")


def sanity_check_init_loss(model, train_loader, criterion, num_classes, device="cpu", tolerance=0.5):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, labels = _get_batch(train_loader, device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

    expected = math.log(num_classes)
    actual   = loss.item()

    print(f"[--] Expected init loss: {expected:.4f} | Actual: {actual:.4f}")

    if abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"Initial loss {actual:.4f} is far from expected ln({num_classes})={expected:.4f}. "
            f"Check labels, criterion, or weight initialization."
        )

    print(f"[OK] Initial loss is reasonable.")


def sanity_check_gradients(model, train_loader, criterion, device="cpu"):
    model.train()
    input_ids, attention_mask, labels = _get_batch(train_loader, device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss   = criterion(logits, labels)
    loss.backward()

    no_grad   = []
    zero_grad = []

    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
        elif param.grad.abs().max() == 0:
            zero_grad.append(name)

    if no_grad:
        print(f"[WARN] No gradient:   {no_grad}")
    if zero_grad:
        print(f"[WARN] Zero gradient: {zero_grad}")
    if not no_grad and not zero_grad:
        print(f"[OK] All gradients flowing.")

    model.zero_grad()


def sanity_check_overfit(model, train_loader, criterion, num_classes, device="cpu", steps=300, lr=1e-2, threshold=0.1):
    original_state = copy.deepcopy(model.state_dict())

    model.train()
    input_ids, attention_mask, labels = _get_batch(train_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"[--] Overfit final loss: {final_loss:.4f} (threshold: {threshold})")

    model.load_state_dict(original_state)
    print(f"[--] Model weights restored.")

    if final_loss > threshold:
        raise RuntimeError(
            f"Model failed to overfit a single batch. "
            f"Final loss {final_loss:.4f} > {threshold} after {steps} steps. "
            f"Check model architecture or data pipeline."
        )

    print(f"[OK] Overfit single batch.")



def run_sanity_checks(model, train_loader, criterion, num_classes, device="cpu"):
    print("=" * 50)
    print("Running sanity checks...")
    print("=" * 50)

    sanity_check_output_shape(model, train_loader, num_classes, device)
    sanity_check_init_loss(model, train_loader, criterion, num_classes, device)
    sanity_check_gradients(model, train_loader, criterion, device)
    sanity_check_overfit(model, train_loader, criterion, num_classes, device)

    print("=" * 50)
    print("All sanity checks passed.")
    print("=" * 50)


