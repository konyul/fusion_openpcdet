from .detector3d_template import Detector3DTemplate
from ..backbones_2d import img_extraction
from pcdet.models import dense_heads

class MVXNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'imgextraction', 'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def build_imgextraction(self, model_info_dict):
        if self.model_cfg.get('IMGEXTRACTION', None) is None:
            return None, model_info_dict

        imgextraction_module = img_extraction.__all__[self.model_cfg.IMGEXTRACTION.NAME](
            model_cfg=self.model_cfg.IMGEXTRACTION,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(imgextraction_module)
        return imgextraction_module, model_info_dict


class MVXNet_PGD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'imgextraction', 'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head', 'dense_head_3d'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss = loss_rpn
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        
        if getattr(self, 'dense_head_3d', None):
            loss_rpn_3d, tb_dict = self.dense_head_3d.get_loss(batch_dict, tb_dict)
            tb_dict['loss_rpn3d'] = loss_rpn_3d.item()
            loss += loss_rpn_3d

        return loss, tb_dict, disp_dict
    
    def build_imgextraction(self, model_info_dict):
        if self.model_cfg.get('IMGEXTRACTION', None) is None:
            return None, model_info_dict

        imgextraction_module = img_extraction.__all__[self.model_cfg.IMGEXTRACTION.NAME](
            model_cfg=self.model_cfg.IMGEXTRACTION,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(imgextraction_module)
        return imgextraction_module, model_info_dict
    
    def build_dense_head_3d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_3D', None) is None:
            return None, model_info_dict
        if self.model_cfg.DENSE_HEAD_3D.NAME == 'MMDet3DHead':
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_3D.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD_3D
            )
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict
        else:
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_3D.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD_3D,
                input_channels=32,
                num_class=self.num_class,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            )
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict
