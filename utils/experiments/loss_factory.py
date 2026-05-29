import torch.nn as nn

def get_loss_fn(loss_cfg):
    if loss_cfg["name"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    elif loss_cfg["name"] == "label_smoothing":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

    else:
        raise ValueError("Unknown loss")

    return criterion