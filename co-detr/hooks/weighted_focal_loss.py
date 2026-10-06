import torch
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss, QualityFocalLoss
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.registry import MODELS
from mmengine.dataset import ClassBalancedDataset
from mmengine.registry import DATASETS

# ClassBalancedDataset lives in mmengine but is not auto-registered in the
# DATASETS registry used by MMDetection configs. Register it manually so that
# type="ClassBalancedDataset" works in config files.
if DATASETS.get('ClassBalancedDataset') is None:
    DATASETS.register_module()(ClassBalancedDataset)


@MODELS.register_module()
class ClassWeightedFocalLoss(FocalLoss):
    """FocalLoss with per-class weight.

    Expands class_weight to (N, C) and multiplies element-wise with the
    per-element focal loss before reduction. Compatible with the CUDA
    sigmoid_focal_loss used in Co-DETR.
    """

    def __init__(self, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight_list = class_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        if self.class_weight_list is not None:
            cw = pred.new_tensor(self.class_weight_list)          # (C,)
            cw_expanded = cw.unsqueeze(0).expand(pred.size(0), -1)  # (N, C)
            if weight is not None:
                weight = weight.view(-1, 1) * cw_expanded         # (N, C)
            else:
                weight = cw_expanded                               # (N, C)
        return super().forward(
            pred, target, weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override)


@MODELS.register_module()
class ClassWeightedQualityFocalLoss(QualityFocalLoss):
    """QualityFocalLoss with per-class weight support.

    Applies class_weight BEFORE summing over the class dimension,
    which is the only point where per-class weighting can be inserted.
    The standard QualityFocalLoss reduces (N, C) → (N,) before the
    per-sample weight is applied, making class-level weighting impossible
    via the normal 'weight' parameter.

    Only supports the tuple target form (label, score) used by CoDINOHead.
    """

    def __init__(self, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight_list = class_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        if self.class_weight_list is None:
            return super().forward(pred, target, weight=weight,
                                   avg_factor=avg_factor,
                                   reduction_override=reduction_override)

        assert not isinstance(target, torch.Tensor), (
            "ClassWeightedQualityFocalLoss only supports tuple target "
            "(label, score) as passed by CoDINOHead.")

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        label, score = target

        # Replicate quality_focal_loss internals at (N, C) resolution
        pred_sigmoid = pred.sigmoid()
        zerolabel = pred_sigmoid.new_zeros(pred.shape)
        # Background loss: all classes treated as negative
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction='none') * pred_sigmoid.pow(self.beta)

        # Foreground: override matched class with quality (IoU) score
        bg_class_ind = pred.size(1)
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        if len(pos) > 0:
            pos_label = label[pos].long()
            scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
            loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
                pred[pos, pos_label], score[pos],
                reduction='none') * scale_factor.abs().pow(self.beta)

        # Apply per-class weight BEFORE summing over class dim  ← key step
        cw = pred.new_tensor(self.class_weight_list)   # (C,)
        loss = loss * cw.unsqueeze(0)                   # (N, C)

        # Sum over class dim → (N,), then apply per-sample weight + reduction
        loss = loss.sum(dim=1)
        loss = self.loss_weight * weight_reduce_loss(
            loss, weight, reduction, avg_factor)
        return loss
