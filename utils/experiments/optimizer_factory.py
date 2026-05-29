import torch

def get_optimizer(optimizer_cfg, model):
    if optimizer_cfg["name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_cfg["lr"],
        )

    elif optimizer_cfg["name"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
        )

    elif optimizer_cfg["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            momentum=0.9,
        )

    else:
        raise ValueError("Unknown optimizer")

    return optimizer