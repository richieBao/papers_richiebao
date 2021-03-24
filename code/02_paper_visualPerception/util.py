# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:28:26 2021

@author: richie bao-paper data processing
"""
from config import config

def filePath_extraction(dirpath,fileType):
    import os
    '''funciton-以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 '''
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): #os.walk()遍历目录，使用help(os.walk)查看返回值解释
       i+=1
       if fileNames: #仅当文件夹中有文件时才提取
           tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
           if tempList: #剔除文件名列表为空的情况,即文件夹下存在不为指定文件类型的文件时，上一步列表会返回空列表[]
               filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info

def retrieve_data_fp(data_root,suffix=['jpg','png']):
    from glob import glob
    import os
    
    fns_rel=[]
    for suff in suffix:
        fns_rel.extend(glob(data_root+"/*.{}".format(suff)))
    
    fns_abs=[os.path.join(config.root,f) for f in fns_rel]
    return fns_rel,fns_abs