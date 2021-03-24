# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:54:54 2021

@author: richie bao-paper data processing
ref_material classification:pytorch-material-classification https://github.com/jiaxue1993/pytorch-material-classification
"""

from easydict import EasyDict as edict
import os
import os.path as osp


C=edict()
config=C

C.root=os.getcwd()
C.kitti_imgs_dir_source=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\data'
C.kitti_imgs_dir='./data/imgs_KITTI'
C.results_path=r'./data/results'
C.mark_boundaries_dir=r'./data/results/mark_boundaries'

#DEP-a deep encoding pooling network
C.dataset_path=r'D:\dataset\gtos-mobile'

C.cuda=True
C.batch_size=128

C.lr = 1e-2
C.lr_decay = 40

C.seed = 0
C.gpu = '0'

C.momentum = 0.9
C.weight_decay = 1e-4

C.resume ='./model_saved/checkpoint.pth.tar' #False
C.start_epoch = 1

C.eval = False

C.start_epoch = 1
C.nepochs = 500

C.snapshot_dir='./model_saved/' 

#Cluster_resnet5
C.recs_segs='./data/results/recs_segs'
C.recs_segs_selection='./data/results/recs_segs_selection'
C.recs_segs_selection_cluster='./data/results/recs_segs_selection_cluster'


#sementic_segmentation
C.seg_sp=r'./data/results/seg_sp'
C.blended_seg_sp=r'./data/results/blended_seg_sp'
C.seg_raw_fn=r'./data/results/segs_raw.pkl'
