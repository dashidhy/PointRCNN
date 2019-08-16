import numpy as np
from pyquaternion import Quaternion
from lib.datasets.nusc_dataset import nuScenesDataset
import lib.utils.nusc_utils as nusc_utils
from lib.config import cfg

class nuScenesRCNNDataset(nuScenesDataset):

    def __init__(self, dataroot, split, mode, verbose=False, npoints=16384, logger=None):
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        super(nuScenesRCNNDataset, self).__init__(dataroot=dataroot, split=split, verbose=verbose)
        
        self.mode = mode
        self.classes = {0:'background', 1:'barrier', 2:'trafficcone', 3:'bicycle', 4:'motorcycle', 
                        5:'pedestrian', 6:'car', 7:'bus', 8:'construction', 9:'trailer', 10:'truck'}
        self.num_class = self.classes.__len__()
        self.npoints = npoints
        self.logger = logger
        
        # TODO: add code for rcnn training
    
    def __len__(self):
        return self.sample_tokens.__len__()
    
    def __getitem__(self, index):
        sample_info = self.nusc.get('sample', self.sample_tokens[index])
        return self.get_rpn_sample(sample_info)
    
    @staticmethod
    def remove_useless_points(pc):
        # remove close points
        x_range = np.abs(pc.points[0]) < 1.0
        y_range = np.abs(pc.points[1]) < 2.5
        pc.points = pc.points[:, np.logical_not(np.logical_and(x_range, y_range))]
        # remove far away points
        dist = np.linalg.norm(pc.points[:2], axis=0)
        pc.points = pc.points[:, dist < 55.0]
        return pc

    @staticmethod
    def anns_to_bboxes(ann_infos):
        if ann_infos:
            ann_bboxes = []
            ann_labels = []
            for ann_info in ann_infos:
                yaw = np.arccos(ann_info['rotation'][0]) * 2.0
                yaw = yaw if ann_info['rotation'][3] >= 0.0 else -yaw
                if yaw > np.pi:
                    yaw -= 2 * np.pi
                elif yaw < -np.pi:
                    yaw += 2 * np.pi
                ann_bboxes.append(ann_info['translation'] + ann_info['size'] + [yaw])
                ann_labels.append(ann_info['category_label'])
            ann_bboxes = np.array(ann_bboxes) # (N, 7)
            ann_labels = np.array(ann_labels) # (N)
        else:
            ann_bboxes = np.zeros((0, 7))
            ann_labels = np.zeros((0))

        return ann_bboxes, ann_labels
    
    @staticmethod
    def pc_trans(lidar_pc, lidar_ep_info, lidar_cs_info):
        rot1 = Quaternion(lidar_cs_info['rotation'])
        rot2 = Quaternion(lidar_ep_info['rotation'])
        lidar_pc.rotate(rot1.rotation_matrix)
        lidar_pc.translate(lidar_cs_info['translation'])
        lidar_pc.rotate(rot2.rotation_matrix)
        lidar_points = lidar_pc.points[:3].T # (N, 3)
        lidar_intensity = lidar_pc.points[3] # (N) # currently don't know how to use it
        return lidar_points, lidar_intensity
    
    @staticmethod
    def ann_trans(ann_infos, ref_ep_info):
        for ann_info in ann_infos:
            for dim in range(ann_info['translation'].__len__()):
                ann_info['translation'][dim] -= ref_ep_info['translation'][dim]
        return ann_infos

    
    def filterate_anns(self, ann_infos, min_pts=10):
        valid_ann_infos = []
        for ann_info in ann_infos:
            # remove super sparse anns
            if ann_info['num_lidar_pts'] < min_pts:
                continue
            # filterate and change category names
            for label in self.classes.keys():
                if self.classes[label] in ann_info['category_name']:
                    #ann_info['category_name'] = self.classes[label]
                    ann_info['category_label'] = label
                    valid_ann_infos.append(ann_info)
                    break
        return valid_ann_infos
    
    def get_rpn_sample(self, sample_info):
        
        # get original data and preprocess point cloud and labels
        sample_outputs = {'token': sample_info['token']}
        sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info = self.get_lidar(sample_info)
        sample_lidar_pc = self.remove_useless_points(sample_lidar_pc)
        sample_ann_infos = self.get_anns(sample_info)
        sample_ann_infos = self.filterate_anns(sample_ann_infos)

        # point cloud coordinate transform for better usage of labels
        sample_lidar_points, sample_lidar_intensity = self.pc_trans(sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info)

        # annotation translate
        sample_ann_infos = self.ann_trans(sample_ann_infos, sample_lidar_ep_info)

        # points sampling
        if self.npoints < sample_lidar_points.__len__():
            sample_lidar_points_dist = np.linalg.norm(sample_lidar_points[:, :2], axis=1)
            sample_lidar_points_isfar = sample_lidar_points_dist > 15.0
            far_idxs = np.where(sample_lidar_points_isfar)[0]
            near_idxs = np.where(sample_lidar_points_isfar == False)[0]
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - far_idxs.__len__(), replace=False)
            idxs_choice = np.concatenate((near_idxs_choice, far_idxs))
        else:
            idxs_choice = np.arange(sample_lidar_points.__len__())
            if self.npoints > sample_lidar_points.__len__():
                extra_choice = np.random.choice(idxs_choice, self.npoints - sample_lidar_points.__len__(), replace=False)
                idxs_choice = np.concatenate((idxs_choice, extra_choice))
        np.random.shuffle(idxs_choice)
        sample_lidar_points = sample_lidar_points[idxs_choice] # (npoints, 3)
        sample_lidar_intensity = sample_lidar_intensity[idxs_choice] # (npoints)

        # gether annotations
        sample_ann_bboxes, sample_ann_labels = self.anns_to_bboxes(sample_ann_infos)

        # data augmentation, points and bboxes changed
        if cfg.AUG_DATA:
            sample_lidar_points, sample_ann_bboxes, aug_method = self.data_augmentation(sample_lidar_points, sample_ann_bboxes)
            sample_outputs['aug_method'] = aug_method

        # gether outputs
        rpn_cls_label, rpn_prt_label = self.generate_rpn_training_labels(sample_lidar_points, sample_ann_bboxes)
        sample_outputs['pts_input'] = sample_lidar_points.astype(np.float32)
        sample_outputs['pts_rect'] = sample_lidar_points.astype(np.float32)
        sample_outputs['pts_features'] = sample_lidar_intensity.astype(np.float32)
        sample_outputs['rpn_cls_label'] = rpn_cls_label.astype(np.float32)
        sample_outputs['rpn_prt_label'] = rpn_prt_label.astype(np.float32)
        sample_outputs['gt_boxes3d'] = sample_ann_bboxes.astype(np.float32)
        sample_outputs['gt_labels'] = sample_ann_labels.astype(np.int32)

        return sample_outputs
    
    def generate_rpn_training_labels(self, points, bboxes): # TODO: add anchor labels
        cls_label = np.zeros((points.__len__()))
        prt_label = np.zeros((points.__len__(), 3))
        if bboxes.__len__() == 0:
            return cls_label, prt_label
        corners = nusc_utils.boxes3d_to_corners3d(bboxes)
        extend_bboxes = nusc_utils.enlarge_box3d(bboxes, extra_width=0.2)
        extend_corners = nusc_utils.boxes3d_to_corners3d(extend_bboxes)
        for k in range(bboxes.__len__()):
            box_corners = corners[k]
            fg_pt_flag = nusc_utils.in_hull(points, box_corners) # (N,)
            fg_points = points[fg_pt_flag] # (M, 3)
            cls_label[fg_pt_flag] = 1 # (N,)

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_corners[k]
            fg_enlarge_flag = nusc_utils.in_hull(points, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center = bboxes[k][:3]  # (x, y, z)
            size = bboxes[k][[4, 3, 5]] # (l, w, h)
            yaw = bboxes[k][6]

            # part label encoding
            local_coor = fg_points - center
            local_coor_cano = nusc_utils.rotate_pc_along_z(local_coor, yaw) # reverse rotation
            prt_label[fg_pt_flag] = local_coor_cano / size + 0.5 # x, y, z in range [0., 1.]

        return cls_label, prt_label.clip(0.0, 1.0)
    
    def data_augmentation(self, points, bboxes, mustaug=False):
        """
        :param points: (N, 3)
        :param bboxes: (N, 7)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            points = nusc_utils.rotate_pc_along_z(points, angle)
            bboxes = nusc_utils.rotate_pc_along_z(bboxes, angle)
            bboxes[:, 6] -= angle
            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            points *= scale
            bboxes[:, :6] *= scale
            aug_method.append(['scaling', scale])
        
        # TODO: if 'translating' in aug_list

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            points[:, 0] = -points[:, 0]
            bboxes[:, 0] = -bboxes[:, 0]
            bboxes[:, 6] = -bboxes[:, 6]
            
            aug_method.append('flip')
        
        bboxes[:, 6][bboxes[:, 6] >  np.pi] -= 2 * np.pi
        bboxes[:, 6][bboxes[:, 6] < -np.pi] += 2 * np.pi
        
        return points, bboxes, aug_method

    def collate_batch(self, sample_list):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert sample_list.__len__() == 1
            return sample_list[0]

        batch_size = sample_list.__len__()
        ans_dict = {}

        for key in sample_list[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for sample in sample_list:
                    max_gt = max(max_gt, sample[key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i, sample in enumerate(sample_list):
                    batch_gt_boxes3d[i, :sample[key].__len__(), :] = sample[key]
                ans_dict[key] = batch_gt_boxes3d
                continue
            
            if key == 'gt_labels':
                max_gt = 0
                for sample in sample_list:
                    max_gt = max(max_gt, sample[key].__len__())
                batch_gt_labels = np.zeros((batch_size, max_gt), dtype=np.int32)
                for i, sample in enumerate(sample_list):
                    batch_gt_labels[i, :sample[key].__len__()] = sample[key]
                ans_dict[key] = batch_gt_labels
                continue

            if isinstance(sample_list[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = sample_list[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([sample[key][np.newaxis, ...] for sample in sample_list], axis=0)

            else:
                ans_dict[key] = [sample[key] for sample in sample_list]
                if isinstance(sample_list[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(sample_list[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
