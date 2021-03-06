# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:55:42 2021

@author: richie bao-paper data processing
"""
import pickle,os
from multiprocessing import Pool
import numpy as np
import segs2jpg_pool
from tqdm import tqdm
import glob
from pathlib import Path
import copy

#Chicago
seg_Path=r'D:\data_paper\paper_01_segsCluster\results/all_seg_9.pkl'

#Madison
# seg_Path=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE\segs\seg_9.pkl'

def segs_unique_elements(seg_Path):
    seg_Path=seg_Path
    with open(seg_Path,'rb') as f:
        segs=pickle.load(f)
    unique_elements, counts_elements=np.unique(segs, return_counts=True)
    print(unique_elements, counts_elements)
    
    return unique_elements

#Chicago
seg_imgs_root=r'D:\data_paper\paper_01_segsCluster\sentinel2_RGB_all'
def check_file(data_list,data_root,suffix='jpg'):
    fns=glob.glob(data_root+"/*.{}".format(suffix))
    fns_stem=[int(Path(fn).stem) for fn in fns]
    none_existed=set(data_list)-set(fns_stem)
    
    return list(none_existed)


if __name__ == '__main__':
    unique_elements=segs_unique_elements(seg_Path)
    UE_=copy.deepcopy(unique_elements) #[:5000] #5000,10000,
    UE=check_file(UE_,seg_imgs_root)
    

    with Pool(8) as p:
        p.map(segs2jpg_pool.segs2rgb, tqdm(UE))
    