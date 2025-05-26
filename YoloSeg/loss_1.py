import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloSegLoss(nn.Module):
    """
    Instance segmentation loss combining BCE, Dice, and Focal loss.
    Usage:
        loss_fn = YoloSegLoss(dice_weight=0.5, use_focal=True)
        loss = loss_fn(pred_masks, gt_masks)
    Args:
        dice_weight (float): Weight of dice loss relative to BCE.
        use_focal (bool): Whether to use focal modulation on BCE.
        gamma (float): Focusing parameter for focal loss.
    """
    def __init__(self, dice_weight=0.5, use_focal=False, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.gamma = gamma

    def forward(self, pred_masks, gt_masks):
        """
        Args:
            pred_masks: [B, N, H, W] logits
            gt_masks:   [B, N, H, W] binary masks
        Returns:
            Combined loss value (scalar)
        """
        # gt_masks = gt_masks.float()
        bce = self.focal_bce_loss(pred_masks, gt_masks) if self.use_focal else F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        dice = self.dice_loss(pred_masks, gt_masks)
        return bce * (1 - self.dice_weight) + dice * self.dice_weight

    def focal_bce_loss(self, logits, targets):
        """Compute Binary Focal Loss from logits."""
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p if t==1 else 1-p
        focal_weight = (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') # BCE loss
        return (focal_weight * bce).mean()

    def dice_loss(self, pred_masks, gt_masks, eps=1e-6):
        pred = torch.sigmoid(pred_masks)
        inter = (pred * gt_masks).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice = (2 * inter + eps) / (union + eps)
        return 1 - dice.mean()

