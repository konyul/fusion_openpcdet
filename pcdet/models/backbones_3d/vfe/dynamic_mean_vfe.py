import torch

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from pcdet.ops.voxel import Voxelization
import torch.nn.functional as F
import numpy as np
from pcdet.ops.voxel import DynamicScatter
from torch import nn
from pcdet.models.fusion_layers.point_fusion import PointFusion

class DynamicMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        # # debug
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict


class DynamicVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.voxelization = Voxelization(voxel_size,point_cloud_range,-1,-1)
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
        
        self.feat_channels=model_cfg.FEAT_CHANNELS
        self.in_channels = model_cfg.IN_CHANNELS
        self.with_cluster_center = model_cfg.WITH_CLUSTER_CENTER
        self.with_voxel_center = model_cfg.WITH_VOXEL_CENTER
        self.with_distance = model_cfg.WITH_DISTANCE
        if self.with_cluster_center:
            self.in_channels += 3
        if self.with_voxel_center:
            self.in_channels += 3
        if self.with_distance:
            self.in_channels += 1
        self.feat_channels = [self.in_channels] + list(self.feat_channels)
        vfe_layers = []
        for i in range(len(self.feat_channels) - 1):
            in_filters = self.feat_channels[i]
            out_filters = self.feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_layer = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        mode = 'max'
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.fusion_layer = PointFusion(img_channels=model_cfg.FUSION_LAYER_CONFIG['img_channels'],
            pts_channels=model_cfg.FUSION_LAYER_CONFIG['pts_channels'],
            mid_channels=model_cfg.FUSION_LAYER_CONFIG['mid_channels'],
            out_channels=model_cfg.FUSION_LAYER_CONFIG['out_channels'],
            img_levels=model_cfg.FUSION_LAYER_CONFIG['img_levels'],
            align_corners=model_cfg.FUSION_LAYER_CONFIG['align_corners'],
            activate_out=model_cfg.FUSION_LAYER_CONFIG['activate_out'],
            fuse_out=model_cfg.FUSION_LAYER_CONFIG['fuse_out'])
        self.out_channels = model_cfg.OUT_CHANNELS
        
        
    def get_output_feature_dim(self):
        return self.out_channels
    
    def voxelize(self, batch_dict):
        cur_device = batch_dict['img_feats'][0].device
        points = batch_dict['points']
        coors = []
        for res in points:
            #res = torch.from_numpy(res).to(device=cur_device)
            res_coors = self.voxelization(res)
            coors.append(res_coors)
        #points = torch.from_numpy(np.vstack(points)).to(device=cur_device)
        points = torch.vstack(points)
        batch_dict['concat_points'] = points
        
        
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        batch_dict['coors_batch'] = coors_batch

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point
    

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_dict['points'] = [torch.from_numpy(batch_dict['points'][i]).to(device=batch_dict['img_feats'][0].device) for i in range(len(batch_dict['points']))]
        self.voxelize(batch_dict)
        batch_size = batch_dict['batch_size']
        features = batch_dict['concat_points'] # (batch_idx, x, y, z, i, e)
        points = batch_dict['points']
        coors = batch_dict['coors_batch']
        img_feats = batch_dict["img_feats"]

        features_ls = [features]
        voxel_mean, mean_coors = self.cluster_scatter(features, coors)
        points_mean = self.map_voxel_center_to_point(
            coors, voxel_mean, mean_coors)
        # TODO: maybe also do cluster for reflectivity
        f_cluster = features[:, :3] - points_mean[:, :3]
        features_ls.append(f_cluster)

        f_center = features.new_zeros(size=(features.size(0), 3))
        f_center[:, 0] = features[:, 0] - (
            coors[:, 3].type_as(features) * self.vx + self.x_offset)
        f_center[:, 1] = features[:, 1] - (
            coors[:, 2].type_as(features) * self.vy + self.y_offset)
        f_center[:, 2] = features[:, 2] - (
            coors[:, 1].type_as(features) * self.vz + self.z_offset)
        features_ls.append(f_center)


        if False:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                batch_dict)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if False:
            return point_feats
        
        batch_dict['voxel_features'] = voxel_feats
        batch_dict['voxel_coords'] = voxel_coors
        return batch_dict
