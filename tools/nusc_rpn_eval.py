import _init_path
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

from lib.net.point_rcnn import PointRCNN
from lib.datasets.nusc_rcnn_dataset import nuScenesRCNNDataset
from lib.config import cfg, cfg_from_file

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/nusc.yaml')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--aug_data', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--fg_thd', type=float, required=True)
parser.add_argument('--save_name', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # set cfg
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
    cfg.AUG_DATA = args.aug_data

    # dataset
    DATA_ROOT = os.path.join('../', 'data', 'nuScenes')
    nusc_dset = nuScenesRCNNDataset(dataroot=DATA_ROOT, npoints=cfg.RPN.NUM_POINTS, split=args.split, mode='EVAL',logger=None)

    # load model
    MODEL_PATH = os.path.join(args.output_dir, 'rpn', cfg.TAG, 'ckpt', 'checkpoint_epoch_{}.pth'.format(args.epoch))
    ckpt = torch.load(MODEL_PATH)
    model_state = ckpt['model_state']
    model = PointRCNN(num_classes=11, use_xyz=True, mode='TRAIN')
    model.load_state_dict(model_state)
    model.cuda().eval()
    rpn_backbone = model.rpn

    # eval
    input_data = {}
    IoU = []
    PA = []
    mIoU = 0
    mPA = 0
    with tqdm(total=len(nusc_dset), leave=False, desc='eval') as pbar:
        for i in range(len(nusc_dset)):
            sample = nusc_dset.__getitem__(i)
            pts_input = torch.from_numpy(sample['pts_input']).cuda(non_blocking=True).float().unsqueeze(0)
            input_data['pts_input'] = pts_input

            ret_dict = rpn_backbone(input_data)
            rpn_cls = torch.sigmoid(ret_dict['rpn_cls']).detach().cpu().squeeze().numpy()

            rpn_fg = rpn_cls > args.fg_thd
            gt_fg = sample['rpn_cls_label'] > 0.0

            # output: mIoU, mPA, sorted IoU, sorted PA
            sIoU = np.sum(np.logical_and(rpn_fg, gt_fg)) / np.sum(np.logical_or(rpn_fg, gt_fg)) if np.sum(np.logical_or(rpn_fg, gt_fg)) != 0 else 1.0
            sPA = np.sum(np.logical_not(np.logical_xor(rpn_fg, gt_fg))) / rpn_fg.size

            # record data
            IoU.append((sample['token'], sIoU))
            PA.append((sample['token'], sPA))
            mIoU += sIoU
            mPA += sPA

            # update progress bar
            pbar.update()
            pbar.set_postfix(dict(sIoU=sIoU, sPA=sPA))
    
    mIoU /= len(nusc_dset)
    mPA /= len(nusc_dset)
    IoU = sorted(IoU, key=lambda x: x[1])
    PA = sorted(PA, key=lambda x: x[1])

    save_dir = os.path.join(args.output_dir, 'rpn', cfg.TAG, 'statistics')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, args.save_name+'.txt')

    with open(save_file, 'w') as f:
        f.write('split: {}\n'.format(args.split))
        f.write('aug_data: {}\n'.format(args.aug_data))
        f.write('fg_thd: %f\n' % args.fg_thd)
        f.write('epoch: %d\n' % args.epoch)
        f.write('mIoU: %f\n' % mIoU)
        f.write('mPA: %f\n' % mPA)
        
        f.write('\nIoUs:\n')
        for token, iou in tqdm(IoU, desc='write IoU'):
            f.write(token + ' ' + '%f\n' % iou)
        
        f.write('\nPAs:\n')
        for token, pa in tqdm(PA, desc='write PA'):
            f.write(token + ' ' + '%f\n' % pa)