import torch.nn as nn
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
import mmcv
import torch


class ImgExtraction(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = ResNet(**model_cfg.BACKBONE_ARGS)
        self.neck = FPN(**model_cfg.NECK_ARGS)
        
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
            **kwargs:
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        images = batch_dict["images"]
        features = self.backbone(images)
        features = self.neck(features)
        batch_dict["img_feats"] = features
        # mmcv.imwrite(images[0].permute(1,2,0).cpu().numpy(),"aaa.jpg")
        # mmcv.imwrite(images[1].permute(1,2,0).cpu().numpy(),"bbb.jpg")
        return batch_dict
