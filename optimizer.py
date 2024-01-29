import torch
from torch.optim import Adam, AdamW

def build_optimizer(model, config):
    lr = config.model.learning_rate
    weight_decay = config.model.weight_decay
    optimizer_name = config.model.optimizer
    if optimizer_name == "Adam":
        optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if optimizer_name == "AdamW":
        optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return optimizer