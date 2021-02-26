# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:52:30 2021

@author: richie bao-paper data processing
"""
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import inscribed_rec_imgMask_pool
import utils
import pandas as pd
import sys,os
import glob
from pathlib import Path

# np.set_printoptions(threshold=sys.maxsize)

# img_fn=r'D:\data_paper\crop_seg_imgs\9_RGB\9_RGB\149.jpg'
imgs_root=r'D:\data_paper\paper_01_segsCluster\sentinel2_RGB_all'
suffix='jpg'
imgs_rec_save_path=r'D:\data_paper\paper_01_segsCluster\sentinel2_RGB_all_rec'
    
def pixel_boudnary(img_bool):
    from PIL import Image, ImageFilter, ImageChops
    import math
    
    a=img_bool
    sz=(len(a[0]), len(a))
    flat_list = [j==False for i in a for j in i]
    image=Image.new('1', sz)
    image.putdata(flat_list)
    # print(np.array(image))7
    contour=ImageChops.difference(image, image.filter(ImageFilter.MinFilter(3)))
    contour_list=list(contour.getdata())
    points=[divmod(i,sz[0]) for i in range(len(contour_list)) if contour_list[i]]
    # points_x,points_y=zip(*points)
    # avg=lambda x: sum(x)/len(x)
    # mean_x=avg(points_x)
    # mean_y=avg(points_y)
    # phase=[(math.atan2(points_y[i]-mean_y, points_x[i]-mean_x),i) \
    #        for i in range(len(points))] 
    # phase.sort()
    
    return points

def pixels_min_max(idx_nz):
    import pandas as pd
    
    def displacement(row):
        return [row[1],row[0]]
    
    rows=np.unique(idx_nz[:,0])
    # print(rows)
    cols=np.unique(idx_nz[:,1])
    # print(cols)
    pixel_df=pd.DataFrame(idx_nz)
    # print(pixel_df)
    px_group_r_min=pixel_df.groupby(by=[0]).min().reset_index().to_numpy()
    px_group_r_max=pixel_df.groupby(by=[0]).max().reset_index().to_numpy()
    # print(px_group_min)
    # print(px_group_max)
    px_group_c_min=pixel_df.groupby(by=[1]).min().reset_index().to_numpy()
    px_group_c_max=pixel_df.groupby(by=[1]).max().reset_index().to_numpy()
    
    px_group_c_min=np.apply_along_axis(displacement, -1, px_group_c_min)
    px_group_c_max=np.apply_along_axis(displacement, -1, px_group_c_max)
    # print(px_group_c_min)
    # print(px_group_c_max)
    
    return np.vstack((px_group_r_min,px_group_r_max,px_group_c_min,px_group_c_max))
    
def pixel_reduction_random(idx_nz,size=300):
    idx_df=pd.DataFrame(idx_nz)
        
    return idx_df.sample(n=size).to_numpy()   

def pixel_reduction_interval(idx_nz,interval=100):
    idx_df=pd.DataFrame(idx_nz)
    # print(idx_df)
    idx=list(range(0,len(idx_df),interval))
    # print(len(idx))
    
    return idx_df.iloc[idx].to_numpy()

flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

def numConcat(num):     
    num1, num2=num
    return np.array("({},{})".format(num1,num2),dtype='object')
fns=glob.glob(imgs_root+"/*.{}".format(suffix))
# fns_stem=[int(Path(fn).stem) for fn in fns]


if __name__ == '__main__':
    for img_fn in tqdm(fns[:10]):  
        fn_stem=int(Path(img_fn).stem)
        img=Image.open(img_fn)
        # img.show()
        
        img_array=np.asarray(img)
        # print(img_array.shape)
        # print(img_array)
        rectangles=[]
        img_mean=np.mean(img_array,axis=-1)
        img_bool=img_mean==0 
        # print(img_mean.shape)
        row_n=img_mean.shape[0]
        col_n=img_mean.shape[1]
        # print(row_n,col_n)
        idx_nz=np.argwhere(img_mean!=0) #np.nonzero(img_mean)
        # print(idx_nz.shape)
        idx_nz_list=idx_nz.tolist()
        
        pixel_ri=pixel_reduction_interval(idx_nz,interval=50)    
    
        idx=pixel_ri
        # s_t=utils.start_time()
        prod_args=partial(inscribed_rec_imgMask_pool.recs, args=[row_n,col_n,idx_nz_list])
        with Pool(8) as p:
            result = p.map(prod_args, idx)  
        
        # with Pool(8) as p:
        #     result = p.map(inscribed_rec_imgMask_pool.recs, idx)     
        # utils.duration(s_t)
        
        recs_len=[np.prod(np.array(lst).shape[:2]) for lst in result]
        
        max_len_idx=recs_len.index(max(recs_len))
        max_rec_idx=result[max_len_idx]    
        max_rec_idx_concat=flatten_lst([[numConcat(i) for i in idx] for idx in max_rec_idx])       
        
        img_coords=np.dstack(np.meshgrid(range(row_n),range(col_n))).transpose((1,0,2))    
        img_coords_concat=np.apply_along_axis(numConcat,-1,img_coords)
        
        mask=np.isin(img_coords_concat,max_rec_idx_concat)  
        i, j = np.where(mask)
        indices = np.meshgrid( np.arange(min(i), max(i) + 1),np.arange(min(j), max(j) + 1),indexing='ij')  
        
        img_max_rec=img_array[tuple(indices)] 
        img_max_rec_pil=Image.fromarray(img_max_rec.astype('uint8'), 'RGB')
        # img_max_rec_pil.show()
        img_max_rec_pil.save(os.path.join(imgs_rec_save_path,'{}.jpg'.format(fn_stem)))    
    