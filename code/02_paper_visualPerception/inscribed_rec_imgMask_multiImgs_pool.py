# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:59:16 2021

@author: richie bao-paper data processing
"""
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from multiprocessing import Pool


def recs(idx,args):
    row_n,col_n,idx_nz_list=args
    
    rows=[]
    for i in range(idx[0],row_n+1):
        # print(i)
        if [i,idx[1]] in idx_nz_list:
            rows.append(i)
        else:
            break    
    cols=[] 
    if len(rows)>3:     
        for k in rows:
            col=[]
            for j in range(idx[1],col_n+1):
                if [k,j] in idx_nz_list:
                    col.append([k,j])
                else:
                    break
            cols.append(col)              
     
    if cols:
        cols_len_min=min([len(lst) for lst in cols])
        if cols_len_min>3:
            cols_extraction=[lst[:cols_len_min] for lst in cols]  

    try:
        return cols_extraction
    except:
        pass
    
    
def recs_(idx):
    # rec_temp=[]
    #A
    rows=[]
    for i in range(idx[0],row_n+1):
        if [i,idx[1]] in idx_nz_list:
            rows.append(i)
        else:
            break
    
    cols=[] 
    if len(rows)>3:      
        for k in rows:
            col=[]
            for j in range(idx[1],col_n+1):
                if [k,j] in idx_nz_list:
                    col.append([k,j])
                else:
                    break
            cols.append(col)               
        
    if cols:
        cols_len_min=min([len(lst) for lst in cols])
        if cols_len_min>3:
            cols_extraction=[lst[:cols_len_min] for lst in cols]              
            # rec_temp.append(cols_extraction)            
    
    #B-np.rot90()
    cols_=[]
    for j in range(idx[1],col_n+1):
        if [idx[0],j] in idx_nz_list:
            cols_.append(j)
        else:
            break
    rows_=[]
    if len(cols_)>3:
        for k in cols_:
            row_=[]
            for i in range(idx[0],row_n+1):
                if[i,k] in idx_nz_list:
                    row_.append([i,k])
                else:
                    break
            rows_.append(row_)
            
    if rows_:
        rows_len_min=min([len(lst) for lst in rows_])
        if rows_len_min>3:
            rows_extraction=[lst[:rows_len_min] for lst in rows_]
            # rec_temp.append(rows_extraction)
            
    try:
        return [cols_extraction,rows_extraction]
    except:
        pass
    
 