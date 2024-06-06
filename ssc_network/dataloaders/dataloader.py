#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Class of pytorch data loader
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
Aug 10, 2019
"""

import glob
import imageio
import numpy as np
import numpy.matlib
import torch.utils.data
from pathlib import Path

from torchvision import transforms

from config import colorMap

# C_NUM = 12  # number of classes
# 'empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs'
#                0  1  2  3  4   5  6  7  8  9 10  11  12  13  14  15 16 17  18  19  20
seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11,
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11]  # 0 - 11
#                21  22  23  24  25  26  27  28  29 30  31  32 33  34  35  36


class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, root, istest=False):
        self.param = {'voxel_size': (240, 144, 240),
                      'voxel_unit': 0.02,            # 0.02m, length of each grid == 20mm
                      'cam_k': [[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
                                [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
                                [0, 0, 1]],          # fx = K(1,1); fy = K(2,2);
                      }
        #
        self.subfix = 'npz'
        self.istest = istest
        self.downsample = 4  # int, downsample = 4, in labeled data, get 1 voxel from each 4

        self.filepaths = self.get_filelist(root, self.subfix)

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] \
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print('Dataset:{} files'.format(len(self.filepaths)))

        correct = False
        if self.subfix == 'npz' and correct:
            self._correct_labels()

    
    def _correct_labels(self):
        print ("Correcting labels with projection indices!")
        N = len(self.filepaths)
        for index in range(N):
            print ("Correcting {}/{} label..".format(index+1, N))
            #name = self.filepaths[index][48:-11]
            filepath = Path(self.filepaths[index])
            target_file = filepath.parent.with_name(filepath.parent.stem + "_fixed") / filepath.stem
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if target_file.with_suffix('.npz').exists():
                print("{} exists! skipping...".format(target_file.with_suffix('.npz')))
                continue

            with np.load(self.filepaths[index]) as npz_file:
                # print(npz_file.files)
                rgb_tensor = npz_file['rgb']
                depth_tensor = npz_file['depth']
                tsdf_hr = npz_file['tsdf_hr']  # flipped TSDF, (240, 144, 240, 1)
                # target_hr = npz_file['target_hr']
                tsdf_lr = npz_file['tsdf_lr']
                target_lr = npz_file['target_lr']
                position = npz_file['position']

                vox_origin, cam_pose, _ = self._read_rle('/media/scratch1/mcheem/datasets/depthbin/{}/{}.bin'.format("NYUtest" if self.istest  else "NYUtrain", filepath.stem[:-7]))
                depth = self._read_depth('/media/scratch1/mcheem/datasets/depthbin/{}/{}.png'.format("NYUtest" if self.istest  else "NYUtrain", filepath.stem[:-7]))  
                _, _, position2, position4 = self._depth2voxel(depth, cam_pose, vox_origin, self.param)
                np.savez_compressed(target_file , rgb=rgb_tensor, depth=depth_tensor, tsdf_hr=tsdf_hr, target_lr=target_lr, position=position2, tsdf_lr=tsdf_lr)



    def __getitem__(self, index):
        _name = self.filepaths[index][:-4]
        # print(_name)

        # ---------------------------------------------------------------------------
        # Processing repackaged data provided by DDRNet
        # ---------------------------------------------------------------------------
        if self.subfix == 'npz':
            with np.load(self.filepaths[index]) as npz_file:
                # print(npz_file.files)
                rgb_tensor = npz_file['rgb']
                depth_tensor = npz_file['depth']
                tsdf_hr = npz_file['tsdf_hr']  # flipped TSDF, (240, 144, 240, 1)
                # target_hr = npz_file['target_hr']
                target_lr = npz_file['target_lr']
                position = npz_file['position']

                if self.istest:
                    tsdf_lr = npz_file['tsdf_lr']  # ( 60,  36,  60)
                    # nonempty = self.get_nonempty(tsdf, 'TSDF')
                    nonempty = self.get_nonempty2(tsdf_lr, target_lr, 'TSDF')  # 这个更符合SUNCG的做法
                    return rgb_tensor, depth_tensor, tsdf_hr, target_lr.T, nonempty.T, position, _name + '.png'
            return rgb_tensor, depth_tensor, tsdf_hr, target_lr.T, position, _name + '.png'

        # else:
        #
        # ---------------------------------------------------------------------------
        # Processing data provided by SSCNet
        # ---------------------------------------------------------------------------
        # --- read depth, shape: (h, w)
        depth = self._read_depth(_name + '.png')  #
        depth_tensor = depth.reshape((1,) + depth.shape)

        # --- read rgb image, shape: (h, w, 3)
        # rgb = self._read_rgb(_name + '.jpg')  #
        rgb = self._read_rgb(_name[:-4] + 'rgb.png')
        rgb_tensor = self.transforms_rgb(rgb)  # channel first, shape: (3, h, w)

        # --- read ground truth
        vox_origin, cam_pose, rle = self._read_rle(_name + '.bin')

        target_hr = self._rle2voxel(rle, self.param['voxel_size'], _name + '.bin')
        target_lr = self._downsample_label(target_hr, self.param['voxel_size'], self.downsample)

        binary_vox, _, position, position4 = self._depth2voxel(depth, cam_pose, vox_origin, self.param)

        npz_file = np.load(_name + '.npz')
        tsdf_hr = npz_file['tsdf']  # SUNCG (W, H, D)

        if self.istest:
            tsdf_lr = self._downsample_tsdf(tsdf_hr, self.downsample)
            # nonempty = self.get_nonempty(tsdf, 'TSDF')
            nonempty = self.get_nonempty2(tsdf_lr, target_lr, 'TSDF')  # 这个更符合SUNCG的做法
            return rgb_tensor, depth_tensor, tsdf_hr, target_lr.T, nonempty.T, position, _name + '.png'

        return rgb_tensor, depth_tensor, tsdf_hr, target_lr.T, position, _name + '.png'

    def __len__(self):
        return len(self.filepaths)

    def get_filelist(self, root, subfix):
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")
        _filepaths = list()
        if isinstance(root, list):  # 将多个root
            for root_i in root:
                fp = glob.glob(root_i + '/*.' + subfix)
                fp.sort()
                _filepaths.extend(fp)
        elif isinstance(root, str):
            _filepaths = glob.glob(root + '/*.' + subfix)  # List all files in data folder
            _filepaths.sort()

        if len(_filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))

        return _filepaths

    @staticmethod
    def _read_depth(depth_filename):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        # depth = misc.imread(depth_filename) / 8000.0  # numpy.float64
        depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        # assert depth.shape == (img_h, img_w), 'incorrect default size'
        depth = np.asarray(depth)
        return depth

    @staticmethod
    def _read_rgb(rgb_filename):  # 0.01s
        r"""Read a RGB image with size H x W
        """
        # rgb = misc.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        rgb = imageio.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        # rgb = np.rollaxis(rgb, 2, 0)  # (H, W, 3)-->(3, H, W)
        return rgb

    @staticmethod
    def _read_rle(rle_filename):  # 0.0005s
        r"""Read RLE compression data
        Return:
            vox_origin,
            cam_pose,
            vox_rle, voxel label data from file
        Shape:
            vox_rle, (240, 144, 240)
        """
        fid = open(rle_filename, 'rb')
        vox_origin = np.fromfile(fid, np.float32, 3).T  # Read voxel origin in world coordinates
        cam_pose = np.fromfile(fid, np.float32, 16).reshape((4, 4))  # Read camera pose
        vox_rle = np.fromfile(fid, np.uint32).reshape((-1, 1)).T  # Read voxel label data from file
        vox_rle = np.squeeze(vox_rle)  # 2d array: (1 x N), to 1d array: (N , )
        fid.close()
        return vox_origin, cam_pose, vox_rle

    # this version takes 0.9s
    @classmethod
    def _rle2voxel(cls, rle, voxel_size=(240, 144, 240), rle_filename=''):
        r"""Read voxel label data from file (RLE compression), and convert it to fully occupancy labeled voxels.
        In the data loader of pytorch, only single thread is allowed.
        For multi-threads version and more details, see 'readRLE.py'.
        output: seg_label: 3D numpy array, size 240 x 144 x 240
        """
        # ---- Read RLE
        # vox_origin, cam_pose, rle = cls._read_rle(rle_filename)
        # ---- Uncompress RLE, 0.9s
        seg_label = np.zeros(voxel_size[0] * voxel_size[1] * voxel_size[2], dtype=np.uint8)  # segmentation label
        vox_idx = 0
        for idx in range(int(rle.shape[0] / 2)):
            check_val = rle[idx * 2]
            check_iter = rle[idx * 2 + 1]
            if check_val >= 37 and check_val != 255:  # 37 classes to 12 classes
                print('RLE {} check_val: {}'.format(rle_filename, check_val))
            # seg_label_val = 1 if check_val < 37 else 0  # 37 classes to 2 classes: empty or occupancy
            # seg_label_val = 255 if check_val == 255 else seg_class_map[check_val]
            seg_label_val = seg_class_map[check_val] if check_val != 255 else 255  # 37 classes to 12 classes
            seg_label[vox_idx: vox_idx + check_iter] = np.matlib.repmat(seg_label_val, 1, check_iter)
            vox_idx = vox_idx + check_iter
        seg_label = seg_label.reshape(voxel_size)  # 3D array, size 240 x 144 x 240
        return seg_label

    # this version takes 3s
    @classmethod  # method 2, new
    def _depth2voxel(cls, depth, cam_pose, vox_origin, param):
        cam_k = param['cam_k']
        voxel_size = param['voxel_size']  # (240, 144, 240)
        unit = param['voxel_unit']  # 0.02
        # ---- Get point in camera coordinate
        H, W = depth.shape
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        # pt_world2 = pt_world
        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        # pt_world2[:, :, 0] = pt_world[:, :, 1]  # x 原始paper方法
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        # pt_world2[:, :, 2] = pt_world[:, :, 0]  # z

        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_size, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros([_ + 1 for _ in voxel_size], dtype=np.float32)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_size + (3,), dtype=np.float32)  # (W, H, D, 3)
        position = np.zeros((H, W), dtype=np.int32)
        position4 = np.zeros((H, W), dtype=np.int32)
        # position44 = np.zeros((H/4, W/4), dtype=np.int32)

        voxel_size_lr = (voxel_size[0] // 4, voxel_size[1] // 4, voxel_size[2] // 4)
        for h in range(H):
            for w in range(W):
                i_x, i_y, i_z = point_grid[h, w, :]
                if 0 <= i_x < voxel_size[0] and 0 <= i_y < voxel_size[1] and 0 <= i_z < voxel_size[2]:
                    voxel_binary[i_x, i_y, i_z] = 1  # the bin has at least one point (bin is not empty)
                    voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                    # position[h, w, :] = point_grid[h, w, :]  # 记录图片上的每个像素对应的voxel位置
                    # 记录图片上的每个像素对应的voxel位置
                    position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
                    # TODO 这个project的方式可以改进
                    position4[h, ] = np.ravel_multi_index((point_grid[h, w, :] / 4).astype(np.int32), voxel_size_lr)
                    # position44[h / 4, w / 4] = np.ravel_multi_index(point_grid[h, w, :] / 4, voxel_size_lr)

        # output --- 3D Tensor, 240 x 144 x 240
        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid  # Release Memory
        return voxel_binary, voxel_xyz, position, position4  # (W, H, D), (W, H, D, 3)

    # this version takes about 0.6s on CPU
    @staticmethod
    def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
        r"""downsample the labeled data,
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        ds = downscale
        small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)

        for i in range(small_size[0]*small_size[1]*small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            # z, y, x = np.unravel_index(i, small_size)  # 速度更慢了
            # print(x, y, z)

            label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            label_bin = label_i.flatten()  # faltten 返回的是真实的数组，需要分配新的内存空间
            # label_bin = label_i.ravel()  # 将多维数组变成 1维数组，而ravel 返回的是数组的视图

            # zero_count_0 = np.sum(label_bin == 0)
            # zero_count_255 = np.sum(label_bin == 255)
            zero_count_0 = np.array(np.where(label_bin == 0)).size  # 要比sum更快
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
                label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale

    @staticmethod
    def _downsample_tsdf(tsdf, downscale=4):  # 仅在Get None empty　时会用到
        r"""
        Shape:
            tsdf, (240, 144, 240)
            tsdf_downscale, (60, 36, 60), (stsdf.shape[0]/4, stsdf.shape[1]/4, stsdf.shape[2]/4)
        """
        if downscale == 1:
            return tsdf
        # TSDF_EMPTY = np.float32(0.001)
        # TSDF_SURFACE: 1, sign >= 0
        # TSDF_OCCLUD: sign < 0  np.float32(-0.001)
        ds = downscale
        small_size = (int(tsdf.shape[0] / ds), int(tsdf.shape[1] / ds), int(tsdf.shape[2] / ds))
        tsdf_downscale = np.ones(small_size, dtype=np.float32) * np.float32(0.001)  # init 0.001 for empty
        s01 = small_size[0] * small_size[1]
        tsdf_sr = np.ones((ds, ds, ds), dtype=np.float32)  # search region
        for i in range(small_size[0] * small_size[1] * small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            tsdf_sr[:, :, :] = tsdf[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            tsdf_bin = tsdf_sr.flatten()
            # none_empty_count = np.array(np.where(tsdf_bin != TSDF_EMPTY)).size
            none_empty_count = np.array(np.where(np.logical_or(tsdf_bin <= 0, tsdf_bin == 1))).size
            if none_empty_count > 0:
                # surface_count  = np.array(np.where(stsdf_bin == 1)).size
                # occluded_count = np.array(np.where(stsdf_bin == -2)).size
                # surface_count = np.array(np.where(tsdf_bin > 0)).size  # 这个存在问题
                surface_count  = np.array(np.where(tsdf_bin == 1)).size
                # occluded_count = np.array(np.where(tsdf_bin < 0)).size
                # tsdf_downscale[x, y, z] = 0 if surface_count > occluded_count else np.float32(-0.001)
                tsdf_downscale[x, y, z] = 1 if surface_count > 2 else np.float32(-0.001)  # 1 or 0 ?
            # else:
            #     tsdf_downscale[x, y, z] = empty  # TODO 不应该将所有值均设为0.001
        return tsdf_downscale

    @staticmethod
    def get_nonempty(voxels, encoding):  # Get none empty from depth voxels
        data = np.zeros(voxels.shape, dtype=np.float32)  # init 0 for empty
        # if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
        #     data[voxels == 1] = 1
        #     return data
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels != 0] = 1
            surface = np.array(np.where(voxels == 1))  # surface=1
        elif encoding == 'TSDF':
            data[np.where(np.logical_or(voxels <= 0, voxels == 1))] = 1
            surface = np.array(np.where(voxels == 1))  # surface
            # surface = np.array(np.where(np.logical_and(voxels > 0, voxels != np.float32(0.001))))  # surface
        else:
            raise Exception("Encoding error: {} is not validate".format(encoding))

        min_idx = np.amin(surface, axis=1)
        max_idx = np.amax(surface, axis=1)
        # print('min_idx, max_idx', min_idx, max_idx)
        # data[:a], data[a]不包含在内, data[b:], data[b]包含在内
        # min_idx = min_idx
        max_idx = max_idx + 1
        # 本该扩大一圈就够了，但由于GT标注的不是很精确，故在高分辨率情况下，多加大一圈
        # min_idx = min_idx - 1
        # max_idx = max_idx + 2
        min_idx[min_idx < 0] = 0
        max_idx[0] = min(voxels.shape[0], max_idx[0])
        max_idx[1] = min(voxels.shape[1], max_idx[1])
        max_idx[2] = min(voxels.shape[2], max_idx[2])
        data[:min_idx[0], :, :] = 0  # data[:a], data[a]不包含在内
        data[:, :min_idx[1], :] = 0
        data[:, :, :min_idx[2]] = 0
        data[max_idx[0]:, :, :] = 0  # data[b:], data[b]包含在内
        data[:, max_idx[1]:, :] = 0
        data[:, :, max_idx[2]:] = 0
        return data

    @staticmethod
    def get_nonempty2(voxels, target, encoding):  # Get none empty from depth voxels
        data = np.ones(voxels.shape, dtype=np.float32)  # init 1 for none empty
        data[target == 255] = 0
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels == 0] = 0
        elif encoding == 'TSDF':
            # --0
            # data[voxels == np.float32(0.001)] = 0
            # --1
            # data[voxels > 0] = 0
            # --2
            # data[voxels >= np.float32(0.001)] = 0
            # --3
            data[voxels >= np.float32(0.001)] = 0
            data[voxels == 1] = 1

        return data

    @staticmethod
    def _get_xyz(size):
        """x 水平 y高低  z深度"""
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h                 # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w                 # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d                 # z, front-back flip
        return _x, _y, _z

    @classmethod
    def labeled_voxel2ply(cls, vox_labeled, ply_filename):  #
        """Save labeled voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
           vox_labeled.shape: (W, H, D)
        """  #
        # ---- Check data type, numpy ndarray
        if type(vox_labeled) is not np.ndarray:
            raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
        # ---- Check data validation
        if np.amax(vox_labeled) == 0:
            print('Oops! All voxel is labeled empty.')
            return
        # ---- get size
        size = vox_labeled.shape
        # print('vox_labeled.shape:', vox_labeled.shape)
        # ---- Convert to list
        vox_labeled = vox_labeled.flatten()
        # ---- Get X Y Z
        _x, _y, _z = cls._get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        # print('_x.shape', _x.shape)
        # ---- Get R G B
        vox_labeled[vox_labeled == 255] = 0  # empty
        # vox_labeled[vox_labeled == 255] = 12  # ignore
        _rgb = colorMap[vox_labeled[:]]
        # print('_rgb.shape:', _rgb.shape)
        # ---- Get X Y Z R G B
        xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # python2.7
        xyz_rgb = list(xyz_rgb)  # python3
        # print('xyz_rgb.shape-1', xyz_rgb.shape)
        # xyz_rgb = zip(_z, _y, _x, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # 将X轴和Z轴交换，用于meshlab显示
        # ---- Get ply data without empty voxel

        xyz_rgb = np.array(xyz_rgb)
        # print('xyz_rgb.shape-1', xyz_rgb.shape)
        ply_data = xyz_rgb[np.where(vox_labeled > 0)]

        if len(ply_data) == 0:
            raise Exception("Oops!  That was no valid ply data.")
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        # ---- Save ply data to disk
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
        del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head
        # print('Saved-->{}'.format(ply_filename))


if __name__ == '__main__':
    # ---- Data loader
    data_dir = '/home/amax/jie/Data_zoo/NYU_SSC/NYUCADval40'

    # ------------------------------------------------
    data_loader = torch.utils.data.DataLoader(
        dataset=NYUDataset(data_dir),
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    for step, (rgb_tesnor, depth, target_lr, position, _filename) in enumerate(data_loader):
        print('step:', step, _filename)












