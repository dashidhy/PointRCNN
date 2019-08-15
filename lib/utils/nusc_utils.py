import numpy as np
from scipy.spatial import Delaunay
import scipy
import torch


def rotate_pc_along_z(points, yaw): # NOTE: this is reverse rotation
    """
    params points: (N, 3+C) or (N, 3)
    params yaw: rad scalar
    Output points: updated pc with XYZ rotated
    """
    cosval = np.cos(yaw)
    sinval = np.sin(yaw)
    rotmat = np.array([[cosval, -sinval], 
                       [sinval,  cosval]])
    points[:, :2] = points[:, :2].dot(rotmat)
    return points

def boxes3d_to_corners3d(boxes3d, rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, yaw]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    w, l, h = boxes3d[:, 3] / 2.0, boxes3d[:, 4] / 2.0, boxes3d[:, 5] / 2.0
    x_corners = np.array([l, -l, -l, l, l, -l, -l, l]).T  # (N, 8)
    y_corners = np.array([w, w, -w, -w, w, w, -w, -w]).T  # (N, 8)
    z_corners = np.array([-h, -h, -h, -h, h, h, h, h]).T  # (N, 8)

    yaw = boxes3d[:, 6]
    zeros, ones = np.zeros(yaw.size), np.ones(yaw.size)
    rot_list = np.array([[np.cos(yaw), -np.sin(yaw), zeros],
                         [np.sin(yaw),  np.cos(yaw), zeros],
                         [      zeros,        zeros,  ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 1, 8), 
                                   y_corners.reshape(-1, 1, 8),
                                   z_corners.reshape(-1, 1, 8)), axis=1)  # (N, 3, 8)
    rotated_corners = np.matmul(R_list, temp_corners)  # (N, 3, 8)
    
    rotated_corners = rotated_corners + boxes3d[:, :3][..., np.newaxis]
    rotated_corners = np.transpose(rotated_corners, (0, 2, 1)) # (N, 8, 3)
    return rotated_corners


def boxes3d_to_corners3d_torch(boxes3d, flip=False):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, yaw]
    :return: corners_rotated: (N, 8, 3)
    """
    w, l, h, yaw = boxes3d[:, 3:4] / 2.0, boxes3d[:, 4:5] / 2.0, boxes3d[:, 5:6] / 2.0, boxes3d[:, 6:7] # (N, 1)
    if flip:
        yaw = yaw + np.pi
    centers = boxes3d[:, :3]
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    x_corners = torch.cat([l, -l, -l, l, l, -l, -l, l], dim=1)  # (N, 8)
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)  # (N, 8)
    z_corners = torch.cat([-h, -h, -h, -h, h, h, h, h], dim=1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim=1), y_corners.unsqueeze(dim=1), z_corners.unsqueeze(dim=1)), dim=1) # (N, 3, 8)

    cosa, sina = torch.cos(yaw), torch.sin(yaw)
    raw_1 = torch.cat([ cosa, -sina, zeros], dim=1)
    raw_2 = torch.cat([ sina,  cosa, zeros], dim=1)
    raw_3 = torch.cat([zeros, zeros,  ones], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim=2)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, yaw]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, yaw]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 4] / 2.0, boxes3d[:, 3] / 2.0
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, yaw]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    return large_boxes3d


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def get_iou3d(corners3d, query_corners3d, need_bev=False):
    """	
    :param corners3d: (N, 8, 3) in rect coords	
    :param query_corners3d: (M, 8, 3)	
    :return:	
    """
    from shapely.geometry import Polygon
    A, B = corners3d, query_corners3d
    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype=np.float32)
    iou_bev = np.zeros((N, M), dtype=np.float32)

    min_h_a = A[:, 0:4, 2].sum(axis=1) / 4.0
    max_h_a = A[:, 4:8, 2].sum(axis=1) / 4.0
    min_h_b = B[:, 0:4, 2].sum(axis=1) / 4.0
    max_h_b = B[:, 4:8, 2].sum(axis=1) / 4.0

    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_h_a[i], min_h_b[j]])
            min_of_max = np.min([max_h_a[i], max_h_b[j]])	
            h_overlap = np.max([0, min_of_max - max_of_min])
            if h_overlap == 0:
                continue

            bottom_a, bottom_b = Polygon(A[i, :4, :2].T), Polygon(B[j, :4, :2].T)
            if bottom_a.is_valid and bottom_b.is_valid:
                # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.
            overlap3d = bottom_overlap * h_overlap
            union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area * (max_h_b[j] - min_h_b[j]) - overlap3d
            iou3d[i][j] = overlap3d / union3d
            iou_bev[i][j] = bottom_overlap / (bottom_a.area + bottom_b.area - bottom_overlap)

    if need_bev:
        return iou3d, iou_bev

    return iou3d
