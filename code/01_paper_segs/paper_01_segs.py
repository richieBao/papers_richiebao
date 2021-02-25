'''
@@ -4,134 +4,46 @@ Created on Wed Feb 24 17:41:51 2021

@author: richie bao-paper data processing
'''

import utils,os,argparse,glob
import utils,os,argparse
import earthpy.spatial as es
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from skimage.segmentation import quickshift
import pickle
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage import exposure


def config():
    global args
    parser=argparse.ArgumentParser(description='config for paper_segs')
    parser.add_argument('--sentinel2_root', default=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE',type=str,metavar='sentinel2',help='sentinel-2 remote sensing images file root.')
    parser.add_argument('--crop_boundary', default=r'D:\RSi\paper_01\GIS_data',type=str,metavar='cropBoudnary',help='crop boundary .shp format with WGS84')
    parser.add_argument('--cropped_path',default='',type=str,help='cropped remote sensing images saved path.')
    parser.add_argument('--kernel_sizes',default=[5],type=list,help='superpixel segmentation quickshift kernel sizes.')
    parser.add_argument('--quickshift_ratio', default=0.5,type=float,help='superpixel segmentation quickshift ratio.')
    parser.add_argument('--quickshift_max_dist', default=1000,type=float,help='superpixel segmentation quickshift max dist.')
    parser.add_argument('--segs_save_path', default='',type=str,help='segmemtation results save path.')
    
    
    args=parser.parse_args()    
    return args

def sentinel2_crop(args,):
def sentinel2_crop(args):
    MTD_fn=os.path.join(args.sentinel2_root,'MTD_MSIL2A.xml')
    crop_folder=r'cropped'
    crop_save_fp=os.path.join(args.sentinel2_root,crop_folder)
    if not os.path.exists(crop_save_fp):
        os.makedirs(crop_save_fp)    
    args.cropped_path=crop_save_fp
        
    band_fns_list,band_fns_dict=utils.Sentinel2_bandFNs(MTD_fn)   
    
    bands_selection=["B02_10m","B03_10m","B04_10m"]
    stack_bands=[os.path.join(args.sentinel2_root,band_fns_dict[b]) for b in bands_selection]
    array_stack, meta_data=es.stack(stack_bands)
    print("meta_data:\n",meta_data)    
    imgs_crs=meta_data['crs']
    
    crop_bound=gpd.read_file(args.crop_boundary)    
    crop_bound_proj=crop_bound.to_crs(imgs_crs)
    
    print("start cropping...")
    crop_imgs=es.crop_all([os.path.join(args.sentinel2_root,f+'.jp2') for f in band_fns_list], crop_save_fp, crop_bound_proj, overwrite=True) #对所有波段band执行裁切
    print("finished cropping...")    
    

# Function to normalize the grid values
def normalize_(array):
    """
    function - 数组标准化 Normalizes numpy arrays into scale 0.0 - 1.0
    """
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))
def sentinel2_processing(MTD_fn,crop=True):
    band_fns_list,band_fns_dict=utils.Sentinel2_bandFNs(MTD_fn)
    # print(band_fns_dict)

def superpixel_segmentation_quickshift_NDVI(args):
    bands_selection=["B02_10m","B03_10m","B04_10m","B08_10m"] 
    croppedImgs_fns=glob.glob(args.cropped_path+"/*.jp2")
    croppedBands_fnsDict={f.split('_')[-3]+'_'+f.split('_')[-2]:f for f in croppedImgs_fns}
    cropped_stack_bands=[croppedBands_fnsDict[b] for b in bands_selection]
    cropped_array_stack,_=es.stack(cropped_stack_bands)

    cropped_array_stack_float=cropped_array_stack.astype(float)
    NDVI=(cropped_array_stack_float[3]-cropped_array_stack_float[2])/(cropped_array_stack_float[3]+cropped_array_stack_float[2])
    img_=np.append(cropped_array_stack[:2],np.expand_dims(NDVI,axis=0),axis=0)
    img=np.stack([normalize_(array) for array in img_]).transpose(1,2,0)
    
    s_t=utils.start_time()
    segments=[quickshift(img, kernel_size=k,ratio=args.quickshift_ratio,max_dist=args.quickshift_max_dist) for k in tqdm(args.kernel_sizes)]
    segments_stack=np.stack(segments)
    print("total time:{}".format(utils.duration(s_t)))

    segs_folder='segs'
    segs_save_fp=os.path.join(args.sentinel2_root,segs_folder)
    if not os.path.exists(segs_save_fp):
        os.makedirs(segs_save_fp)  
    args.segs_save_path=segs_save_fp
    with open(os.path.join(segs_save_fp,'seg_{}.pkl'.format('_'.join([str(i) for i in args.kernel_sizes]))),'wb') as f:
        pickle.dump(segments_stack,f)

def img_exposure(img):
    '''
    function - 拉伸图像
    '''
    p2, p98=np.percentile(img, (2,98))
    return exposure.rescale_intensity(img, in_range=(p2, p98)) / 65535
def main():
    #00-arguments cofigure
    args=config()   
    #01-crop
    sentinel2_crop(args)
    

def segs_mark_boundaries_show(args,seg_name):
    with open(os.path.join(args.segs_save_path,seg_name),'rb') as f:
        segs=pickle.load(f)    

    bands_selection=["B02_10m","B03_10m","B04_10m"] 
    croppedImgs_fns=glob.glob(args.cropped_path+"/*.jp2")
    croppedBands_fnsDict={f.split('_')[-3]+'_'+f.split('_')[-2]:f for f in croppedImgs_fns}
    cropped_stack_bands=[croppedBands_fnsDict[b] for b in bands_selection]
    cropped_array_stack,_=es.stack(cropped_stack_bands)      
    rasters=[normalize_(img_exposure(band)) for band in cropped_array_stack]
    rasters.reverse()
    img_rgb=np.dstack(rasters)
    
    fig, ax=plt.subplots(figsize=(50*2, 50*2))
    ax.imshow(mark_boundaries(img_rgb, segs[0]))  
    plt.show()
    
    return img_rgb
  

def segs2jpgs(args,seg_name,img_rgb):
    with open(os.path.join(args.segs_save_path,seg_name),'rb') as f:
        segs=pickle.load(f) 
        
    unique_elements, counts_elements=np.unique(segs, return_counts=True)
    print(unique_elements, counts_elements)        
    


if __name__ == '__main__':
    #00-arguments cofigure
    args=config()   
    #01-crop
    sentinel2_crop(args)    
    #02-superpixel_segmentation_quickshift
    superpixel_segmentation_quickshift_NDVI(args)
    #03-show segmentation mark boundaries
    seg_name=r'seg_5.pkl'
    img_rgb=segs_mark_boundaries_show(args,seg_name)
    #04-save segs as RGB.jpg masked with value=0
    main()