# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:36:29 2021

@author: richie bao-paper data processing
"""
import pickle
import cityscapes_labels
from pathlib import Path
from tqdm import tqdm
import numpy as np
import util
from config import config
import shutil,os
        
def segs_match(superpixel_segs,segs_raw):
    segs_match_dict_ID={}
    segs_match_dict_label={}
    for k in tqdm(segs_raw.keys()):
        superpixel_seg=superpixel_segs[k]
        seg_raw=segs_raw[k]
        superpixel_seg_unique=np.unique(superpixel_seg)
        # print(superpixel_seg_unique)
        segs_match_subdict={}
        segs_match_subdict_label={}
        for s_seg in superpixel_seg_unique:
            s_seg_mask=superpixel_seg==s_seg
            # print(s_seg_mask)
            seg_raw_extraction=seg_raw[s_seg_mask]
            # print(np.unique(seg_raw_extraction))
            unique_elements, counts_elements=np.unique(seg_raw_extraction, return_counts=True)
            # print(unique_elements, counts_elements)
            seg_raw_idx=unique_elements[counts_elements.tolist().index(max(counts_elements))]
            # print(seg_raw_idx)
            segs_match_subdict[s_seg]=seg_raw_idx
            segs_match_subdict_label[s_seg]=cityscapes_labels.trainId2label[seg_raw_idx].name
            # break        
        segs_match_dict_ID[k]=segs_match_subdict
        segs_match_dict_label[k]=segs_match_subdict_label
        
        # break    
    # print(segs_match_dict)        
    save_fp='./data/results/segs_match.pkl'
    with open(save_fp,'wb') as f:
        pickle.dump([segs_match_dict_ID,segs_match_dict_label],f)
    return segs_match_dict_ID,segs_match_dict_label
        
def recs_segs_selection(segs_match_dict_label,select_labels,recs_segs_root,save_fp):
    selection_fn={}
    for fn in tqdm(segs_match_dict_label.keys()):
        fp=os.path.join(recs_segs_root,fn)
        # print(fp)
        selection=[os.path.join(fp,str(k)+'.jpg') for k in segs_match_dict_label[fn].keys() if segs_match_dict_label[fn][k] in select_labels]
        # print(len(selection),len(segs_match_dict_label[fn].keys()))
        
        save_subfp=os.path.join(save_fp,fn)
        if not os.path.exists(save_subfp):
            os.makedirs(save_subfp)
            
        selection_fn[fn]=[shutil.copy(f,save_subfp) for f in selection if os.path.exists(f)]

    return selection_fn
        
if __name__ == "__main__":
    # superpixel_segs_fn=r'./data/results/superpixel_segs.pkl'
    # with open(superpixel_segs_fn,'rb') as f:
    #     superpixel_segs_=pickle.load(f)
    # superpixel_segs={Path(k).stem:v for k,v in superpixel_segs_.items()}    
        
        
    # segs_raw_fn=r'./data/results/segs_raw.pkl'
    # with open(segs_raw_fn,'rb') as f:
    #     segs_raw=pickle.load(f)
        
    # segs_match_dict_ID,segs_match_dict_label=segs_match(superpixel_segs,segs_raw)
    segs_match_dict_fn='./data/results/segs_match.pkl'
    with open(segs_match_dict_fn,'rb') as f:
        segs_match_dict_ID,segs_match_dict_label=pickle.load(f)
    selection_label=['building','road'] 
    selection_fn=recs_segs_selection(segs_match_dict_label,selection_label,config.recs_segs,config.recs_segs_selection)
    