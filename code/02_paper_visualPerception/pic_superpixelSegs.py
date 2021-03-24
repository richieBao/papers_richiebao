# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:39:07 2021

@author: richie bao-paper data processing
"""
from config import config

def retrieve_data_fp(data_root,suffix=['jpg','png']):
    from glob import glob
    import os
    
    fns_rel=[]
    for suff in suffix:
        fns_rel.extend(glob(data_root+"/*.{}".format(suff)))
    
    fns_abs=[os.path.join(config.root,f) for f in fns_rel]
    return fns_rel,fns_abs

def img_watershed_segs(img_fps,save_path,mark_boundaries_sp,show_seg=False,**kwargs):
    from skimage.filters import sobel
    from skimage.color import rgb2gray
    from skimage.segmentation import felzenszwalb, watershed
    from PIL import Image
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    import pickle,os    
    
    segs_dic={}
    for img_fp in tqdm(img_fps):
        img_pil=Image.open(img_fp)
        # segments_fz=felzenszwalb(img_pil, scale=kwargs['scale'], sigma=kwargs['sigma'], min_size=kwargs['min_size'])
        img_pil_=sobel(rgb2gray(np.array(img_pil)))
        segments_watershed=watershed(img_pil_, markers=kwargs['markers'], compactness=kwargs['compactness'])
        # img_name_stem=Path(img_fp).stem
        segs_dic[img_fp]=segments_watershed
        
        if show_seg:
            fig, ax=plt.subplots() #figsize=(10*2, 10*2)
            ax.imshow(mark_boundaries(img_pil, segments_watershed))  
            plt.show()
            fig.savefig(os.path.join(mark_boundaries_sp,'{}'.format(Path(img_fp).name)))
            
    
    save_fp=os.path.join(save_path,'superpixel_segs.pkl')
    with open(save_fp,'wb') as f:
        pickle.dump(segs_dic,f)
    print("results has been saved into {}".format(save_fp))
    
    return save_fp

def segs2imgs_b0(segs_fp,save_path):
    from tqdm import tqdm
    import pickle,os
    from tqdm import tqdm
    import matplotlib
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import copy
    
    imgs_segs_path=os.path.join(save_path,'imgs_segs')
    if not os.path.exists(imgs_segs_path):
        os.makedirs(imgs_segs_path)
    
    with open(segs_fp,'rb') as f:
        imgs_segs=pickle.load(f)

    for f,segs in tqdm(imgs_segs.items()):
        img_segs_path=os.path.join(imgs_segs_path,Path(f).stem)
        if not os.path.exists(img_segs_path):
            os.makedirs(img_segs_path)
        # print(k,seg)
        img_rgb=np.array(Image.open(f))
        unique_elements, counts_elements=np.unique(segs, return_counts=True)
        for seg in unique_elements:
            seg_mask=np.squeeze(segs==seg)
            i, j = np.where(seg_mask)
            indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                  np.arange(min(j), max(j) + 1),
                                  indexing='ij')    
            i_, j_ = np.where(~seg_mask)
            
            img_deepCopy=copy.deepcopy(img_rgb)
            img_deepCopy[tuple([i_,j_])]=0
            seg_img=img_deepCopy[tuple(indices)]              
            
            matplotlib.image.imsave(os.path.join(img_segs_path,"{}.jpg".format(seg)),seg_img)            
            
    print("\nfinished conversion...")
    
def KITTI_imgs_selection(data_root_source,data_root_destination,suffix=['jpg','png'],interval=100):
    from glob import glob
    import os,shutil
    from pathlib import Path
    
    if os.path.isdir(data_root_destination):
        shutil.rmtree(data_root_destination)
    os.mkdir(data_root_destination)
    
    fns_abs=[]
    for suff in suffix:
        fns_abs.extend(glob(data_root_source+"/*.{}".format(suff)))
    
    fns_dict={Path(f).stem:f for f in fns_abs}
    fns_idx_selection=list(fns_dict.keys())[::interval]
    fns_selection=[shutil.copy(fns_dict[k],data_root_destination) for k in fns_idx_selection]
    # print(fns_selection)
    print("The copy completed successfully.")

if __name__ == "__main__":
    KITTI_imgs_selection(config.kitti_imgs_dir_source,data_root_destination=config.kitti_imgs_dir)
    
    fns_rel,fns_abs=retrieve_data_fp(config.kitti_imgs_dir)
    # img_Felzenszwalb_segs(fns_rel[3],scale=100,sigma=0.5,min_size=50)
    segs_fp=img_watershed_segs(fns_rel,config.results_path,config.mark_boundaries_dir,show_seg=True,markers=250, compactness=0.0010)
    segs2imgs_b0(segs_fp,config.results_path)
