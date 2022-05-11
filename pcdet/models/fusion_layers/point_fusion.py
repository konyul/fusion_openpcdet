# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from . import apply_3d_transformation
import cv2
import mmcv

def get_proj_mat_by_coord_type(img_meta, coord_type):
    """Obtain image features using points.
    Args:
        img_meta (dict): Meta info.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Can be case-insensitive.
    Returns:
        torch.Tensor: transformation matrix.
    """
    coord_type = coord_type.upper()
    mapping = {'LIDAR': 'lidar2img', 'DEPTH': 'depth2img', 'CAMERA': 'cam2img'}
    assert coord_type in mapping.keys()
    return img_meta[mapping[coord_type]]

def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.
    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.
    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 sample_idx,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    sample_idx = sample_idx
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, sample_idx, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1
    
    # import numpy as np
    # image = img_meta['images'][0].cpu().permute((1,2,0)).numpy()
    # img2 = image.copy()
    # for idx in range(len(img_coors[:,0])):
    #     point_coors = (int(img_coors[:,0][idx]),int(img_coors[:,1][idx]))
    #     cv2.circle(img2,point_coors,1,(255,255,255),-1)
    # cv2.imwrite("ccc.jpg", img2)
    

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()


class PointFusion(nn.Module):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__()
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(pts)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas,i))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta,i):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        sample_idx = i
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][sample_idx][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)[sample_idx].detach().cpu().numpy()
        if not 'img_shape' in img_meta:
            img_meta['img_shape'] = torch.tile(torch.tensor(img_meta['images'].shape[-2:]),(img_meta['images'].shape[0],1))
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['images'].shape[-2:],
            img_shape=img_meta['img_shape'][sample_idx][:2].detach().cpu(),
            sample_idx = sample_idx,
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts
