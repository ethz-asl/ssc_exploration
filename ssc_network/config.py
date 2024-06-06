import numpy as np
import torch

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'nyu':
            # folder that contains dataset/.
            return {'train': '/home/schmluk/data/NYUtrain_npz',
                    'val': '/home/schmluk/data/NYUtest_npz'}

        elif dataset == 'nyucad':
            return {'train': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADtrain_npz',
                    'val': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADtest_npz'}

        # debug
        elif dataset == 'debug':
            return {'train': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADval40_npz',
                    'val': '/home/jsg/jie/Data_zoo/NYU_SSC/NYUCADval40_npz'}

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


# ssc: color map
colorMap = np.array([[22, 191, 206],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling
                     [43, 160, 4],      # 2 floor
                     [158, 216, 229],   # 3 wall
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     [153, 153, 153],     # 12 Accessible area, or label==255, ignore
                     ]).astype(np.int32)

# ###########################################################################################

class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])



