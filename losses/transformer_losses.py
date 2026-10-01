import torch
import torch.nn.functional as F


def compute_ntp_loss(next_pred, x, padding_mask, loss_type="mse"):
    pred = next_pred[:, :-1]
    target = x[:, 1:]
    valid = (~padding_mask)[:, 1:].float().unsqueeze(-1)

    if loss_type == "mse":
        per_elem = (pred - target) ** 2 * valid
        denom = valid.sum() * pred.size(-1) + 1e-6
        return per_elem.sum() / denom
    if loss_type == "cosine":
        p = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        cos = (p * t).sum(-1, keepdim=True)
        return ((1.0 - cos) * valid).sum() / (valid.sum() + 1e-6)
    raise ValueError(f"Unknown ntp_loss_type: {loss_type}")
