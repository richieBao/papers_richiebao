# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:07:28 2021

@author: richie bao-paper data processing
"""
import os,pickle,utils

#Madison
# seg_imgs_root=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE\segs_imgs\rgb_9'

# bands_root=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE\cropped'
# bands_fn=[os.path.join(bands_root,fn) for fn in [
#     r'T15TYH_20200826T164901_B04_10m_crop.jp2',
#     r'T15TYH_20200826T164901_B03_10m_crop.jp2',
#     r'T15TYH_20200826T164901_B02_10m_crop.jp2'
#     ]]

# seg_Path=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE\segs\seg_9.pkl'

#Chicago
seg_imgs_root=r'D:\data_paper\paper_01_segsCluster\sentinel2_RGB_all'

MTD_MSIL2A_20201007_fn=r'D:\RSi\S2B_MSIL2A_20201007T164109_N0214_R126_T16TDM_20201007T210310.SAFE\MTD_MSIL2A.xml'
band_fns_list,band_fns_dict=utils.Sentinel2_bandFNs(MTD_MSIL2A_20201007_fn)
sentinel2_root=r'D:\RSi\S2B_MSIL2A_20201007T164109_N0214_R126_T16TDM_20201007T210310.SAFE'
bands_fn=[          
          os.path.join(sentinel2_root,band_fns_dict['B04_10m']),
          os.path.join(sentinel2_root,band_fns_dict['B03_10m']),
          os.path.join(sentinel2_root,band_fns_dict['B02_10m'])]

seg_Path=r'D:\data_paper\paper_01_segsCluster\results/all_seg_9.pkl'


def bandsComposite(bands_fn):
    import rasterio as rio
    import numpy as np    
    import matplotlib.pyplot as plt
    from skimage import exposure
    '''
    function - 使用rasterio库读取与显示多个波段组合
    '''
    # Function to normalize the grid values
    def normalize(array):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))
    
    def img_exposure(img):
        p2, p98=np.percentile(img, (2,98))
        return exposure.rescale_intensity(img, in_range=(p2, p98)) / 65535
    
    rasters=[normalize(img_exposure(rio.open(fn).read(1))) for fn in bands_fn]
    # Create RGB natural color composite
    rgb=np.dstack(rasters)   
    
    return rgb

img_rgb=bandsComposite(bands_fn) 

with open(seg_Path,'rb') as f:
    segs=pickle.load(f)

seg_fail_list=[]

def segs2rgb(seg):
    import copy
    import matplotlib    
    import numpy as np

    try:
        seg_mask=np.squeeze(segs==seg)
        i, j = np.where(seg_mask)
        indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                              np.arange(min(j), max(j) + 1),
                              indexing='ij')      
        
        seg_img=img_rgb[tuple(indices)]
        seg_mask_rec=seg_mask[tuple(indices)]
        i_, j_ = np.where(~seg_mask_rec)   
        seg_img[tuple([i_,j_])]=0    
        
        
        matplotlib.image.imsave(os.path.join(seg_imgs_root,"{}.jpg".format(seg)),seg_img)
    except:
        seg_fail_list.append(seg)