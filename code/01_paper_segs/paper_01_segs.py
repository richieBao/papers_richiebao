# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:41:51 2021

@author: richie bao-paper data processing
"""
import utils,os,argparse
import earthpy.spatial as es

def config():
    global args
    parser=argparse.ArgumentParser(description='config for paper_segs')
    parser.add_argument('--sentinel2_root', default=r'D:\RSi\paper_01\S2A_MSIL2A_20200826T164901_N0214_R026_T15TYH_20200826T211750.SAFE',type=str,metavar='sentinel2',help='sentinel-2 remote sensing images file root.')
    
    
    args=parser.parse_args()    
    return args

def sentinel2_crop(args):
    MTD_fn=os.path.join(args.sentinel2_root,'MTD_MSIL2A.xml')
    crop_folder=r'cropped'
    crop_save_fp=os.path.join(args.sentinel2_root,crop_folder)
    if not os.path.exists(crop_save_fp):
        os.makedirs(crop_save_fp)    
    band_fns_list,band_fns_dict=utils.Sentinel2_bandFNs(MTD_fn)   
    
    
    

def sentinel2_processing(MTD_fn,crop=True):
    band_fns_list,band_fns_dict=utils.Sentinel2_bandFNs(MTD_fn)
    # print(band_fns_dict)

    


def main():
    #00-arguments cofigure
    args=config()   
    #01-crop
    sentinel2_crop(args)
    





if __name__ == '__main__':
    main()