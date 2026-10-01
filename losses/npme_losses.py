import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(tensor, num_classes):
    if tensor.dim() == 5:
        tensor = tensor.squeeze(1)
    b, d, h, w = tensor.shape
    one_hot = torch.zeros(b, num_classes, d, h, w, device=tensor.device)
    one_hot.scatter_(1, tensor.unsqueeze(1).long(), 1.0)
    return one_hot


class HybridLoss(nn.Module):
    def __init__(self, num_classes, trigger_iter, geo_weight):
        super().__init__()
        self.num_classes = num_classes
        self.trigger_iter = trigger_iter
        self.geo_weight = geo_weight
        weights = torch.tensor([1.0, 10.0, 20.0], dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.ce = nn.CrossEntropyLoss(weight=self.weights)

    def get_grad_loss(self, probs, targets_oh):
        grad_p_y = torch.abs(probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :])
        grad_p_x = torch.abs(probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :])
        grad_p_z = torch.abs(probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1])
        with torch.no_grad():
            grad_t_y = torch.abs(targets_oh[:, :, 1:, :, :] - targets_oh[:, :, :-1, :, :])
            grad_t_x = torch.abs(targets_oh[:, :, :, 1:, :] - targets_oh[:, :, :, :-1, :])
            grad_t_z = torch.abs(targets_oh[:, :, :, :, 1:] - targets_oh[:, :, :, :, :-1])
        return F.mse_loss(grad_p_y, grad_t_y) + F.mse_loss(grad_p_x, grad_t_x) + F.mse_loss(grad_p_z, grad_t_z)

    def forward(self, logits, targets, current_iter, geo_pred=None, geometry=None):
        ce_loss = self.ce(logits, targets.squeeze(1).long())

        probs = F.softmax(logits, dim=1)
        targets_oh = to_one_hot(targets, self.num_classes)

        dice_loss = 0.0
        for c in range(1, self.num_classes):
            p_c, t_c = probs[:, c, ...], targets_oh[:, c, ...]
            inter = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()
            dice_loss += (1.0 - (2.0 * inter + 1e-5) / (union + 1e-5))
        dice_loss /= (self.num_classes - 1)

        grad_loss = torch.tensor(0.0, device=logits.device)
        if current_iter >= self.trigger_iter:
            grad_loss = self.get_grad_loss(probs, targets_oh)
            total_loss = 0.3 * ce_loss + 0.7 * dice_loss + 0.2 * grad_loss
        else:
            total_loss = 0.3 * ce_loss + 0.7 * dice_loss

        geo_loss = torch.tensor(0.0, device=logits.device)
        if geo_pred is not None and geometry is not None:
            geo_loss = F.mse_loss(geo_pred.float(), geometry.float())
            total_loss = total_loss + self.geo_weight * geo_loss

        return total_loss, dice_loss, grad_loss, geo_loss
