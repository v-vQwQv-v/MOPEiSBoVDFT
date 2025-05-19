import torch
import torch.nn.functional as F

def calculate_segmentation_loss(pred_coeffs, proto, gt_masks):
    """
    Args:
        pred_coeffs: [B, C, N]  → mask coefficients
        proto:       [B, C, H, W] → shared mask bases
        gt_masks:    [B, N, H, W] → ground truth binary masks
    """
    pred_masks = torch.einsum('bcn,bchw->bnhw', pred_coeffs, proto)  # → [B, N, H, W]
    loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())
    return loss
