import torch
import torch.nn.functional as F

class BceDiceLoss(torch.nn.Module):
    def __init__(self, pos_weight=1, reduction=False):
        super(BceDiceLoss, self).__init__()
        self.weights = torch.tensor([pos_weight]).to("cuda:0")
        self.reduction = reduction
    def forward(
        self, 
        pred, 
        target, 
        smooth = 1e-5,
    ):
        pred = pred.to(torch.float32).to("cuda:0").squeeze()
        target = target.to(torch.float32).to("cuda:0").squeeze()
        # binary cross entropy loss
        bce = F.binary_cross_entropy_with_logits(
            pred, 
            target, 
            reduction="none",  
            pos_weight=self.weights
        ).mean(dim=(-1,-2))

        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(axis=-1).sum(axis=-1)
        union = (pred + target).sum(axis=-1).sum(axis=-1)
        
        dice = (2.0 * (intersection + smooth) / (union + smooth))
        dice_loss = 1.0 - dice
        loss = bce + dice_loss
        if self.reduction:
            return loss, bce, dice_loss
        return loss.mean()

class DiceLoss(torch.nn.Module):
    def __init__(self, pos_weight=None, reduction=False):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
    def forward(
        self, 
        pred, 
        target, 
        smooth = 1e-5,
    ):
        pred = pred.to(torch.float32).to("cuda:0").squeeze()
        target = target.to(torch.float32).to("cuda:0").squeeze()
        
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(axis=-1).sum(axis=-1)
        union = (pred + target).sum(axis=-1).sum(axis=-1)
        
        dice = (2.0 * (intersection + smooth) / (union + smooth))
        loss = 1.0 - dice
        if self.reduction:
            return [loss]
        return loss.mean()