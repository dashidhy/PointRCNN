import numpy as np
import torch
import torch.nn as nn
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
from lib.config import cfg


def get_model(input_channels=6, use_xyz=True):
    return Pointnet2MSG_AC(input_channels=input_channels, use_xyz=use_xyz)

class Pointnet2MSG_SUB(nn.Module):
    def __init__(self, stage, input_channels=6, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS[stage].__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[stage][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[stage][k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[stage][k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[stage][k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS[stage].__len__()):
            pre_channel = cfg.RPN.FP_MLPS[stage][k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS[stage]) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[stage][k])
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz[0], l_features[0]


class Pointnet2MSG_AC(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()

        self.pointnet2_stages = nn.ModuleList()
        self.rpn_cls_layers = nn.ModuleList()
        channel_in = input_channels
        for stage in range(cfg.RPN.STAGES):

            # pointnet++
            self.pointnet2_stages.append(Pointnet2MSG_SUB(stage=stage, input_channels=channel_in, use_xyz=use_xyz))
            channel_in = cfg.RPN.FP_MLPS[stage][0][-1]
            
            # classification layer
            cls_layers = []
            pre_channel = channel_in
            for k in range(cfg.RPN.CLS_FC[stage].__len__()):
                cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[stage][k], bn=cfg.RPN.USE_BN))
                pre_channel = cfg.RPN.CLS_FC[stage][k]
            cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
            if cfg.RPN.DP_RATIO >= 0:
                cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
            self.rpn_cls_layers.append(nn.Sequential(*cls_layers))
    
    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            for stage in range(cfg.RPN.STAGES):
                nn.init.constant_(self.rpn_cls_layers[stage][-1].conv.bias, -np.log((1 - pi) / pi))

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        '''
        pointcloud: (B, N, 3+C)
        '''

        batch_size, num_pts = pointcloud.size(0), pointcloud.size(1)
        device = pointcloud.device
        pc_attention = pointcloud
        pts_idx_in = torch.arange(num_pts).repeat(batch_size, 1).to(device) # (B, N)
        row_sort = torch.arange(batch_size).repeat(num_pts, 1).transpose(0, 1).to(device) # (B, N)
        rpn_cls_list = []
        pts_idx_list = []
        
        # stages
        for stage in range(cfg.RPN.STAGES):

            if stage > 0:
                # sort previous stage cls
                sorted_stage_cls_idx = torch.argsort(stage_cls.view(-1, num_pts), dim=1, descending=True) # (B, N)
                pts_idx_in = pts_idx_in[row_sort, sorted_stage_cls_idx]

                num_attention = int(num_pts/(2**stage))
                attention_pts_idx = sorted_stage_cls_idx[:, :num_attention] # (B, num_attention)
                prestage_pts_idx = sorted_stage_cls_idx[:, num_attention:] # (B, num_pts - num_attention)
                row_attention = torch.arange(batch_size).repeat(num_attention, 1).transpose(0, 1) # (B, num_attention)
                row_prestage = torch.arange(batch_size).repeat(num_pts - num_attention, 1).transpose(0, 1) # (B, num_pts - num_attention)

                features_trans = features.transpose(1, 2) # (B, N, C)
                features_attention = features_trans[row_attention, attention_pts_idx]
                features_prestage = features_trans[row_prestage, prestage_pts_idx]
                xyz_attention = xyz[row_attention, attention_pts_idx]
                xyz_prestage = xyz[row_prestage, prestage_pts_idx]

                pc_attention = torch.cat((xyz_attention, features_attention), dim=2)

            xyz, features = self.pointnet2_stages[stage](pc_attention) # (B, N, 3), (B, C, N)

            if stage > 0: # cancatenate features from previous stages
                xyz = torch.cat((xyz, xyz_prestage), dim=1)
                features = torch.cat((features, features_prestage.transpose(1, 2)), dim=2)
            
            stage_cls = self.rpn_cls_layers[stage](features).transpose(1, 2).contiguous() # (B, N, 1)
            rpn_cls_list.append(stage_cls)
            pts_idx_list.append(pts_idx_in)
        

        return xyz, features, rpn_cls_list, pts_idx_list
