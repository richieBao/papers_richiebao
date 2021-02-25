# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:10:00 2021

@author: richie bao-paper data processing
"""

def Sentinel2_bandFNs(MTD_MSIL2A_fn):
    import xml.etree.ElementTree as ET
    '''
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Paras:
    MTD_MSIL2A_fn - MTD_MSIL2A 文件路径
    
    Returns:
    band_fns_list - 波段相对路径列表
    band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典
    '''
    Sentinel2_tree=ET.parse(MTD_MSIL2A_fn)
    Sentinel2_root=Sentinel2_tree.getroot()

    print("GENERATION_TIME:{}\nPRODUCT_TYPE:{}\nPROCESSING_LEVEL:{}".format(Sentinel2_root[0][0].find('GENERATION_TIME').text,
                                                           Sentinel2_root[0][0].find('PRODUCT_TYPE').text,                 
                                                           Sentinel2_root[0][0].find('PROCESSING_LEVEL').text
                                                          ))
    
    # print("MTD_MSIL2A.xml 文件父结构:")
    for child in Sentinel2_root:
        print(child.tag,"-",child.attrib)
    print("_"*50)    
    band_fns_list=[elem.text for elem in Sentinel2_root.iter('IMAGE_FILE')] #[elem.text for elem in Sentinel2_root[0][0][11][0][0].iter()]
    band_fns_dict={f.split('_')[-2]+'_'+f.split('_')[-1]:f+'.jp2' for f in band_fns_list}
    # print('get sentinel-2 bands path:\n',band_fns_dict)
    
    return band_fns_list,band_fns_dict  


def start_time():
    import datetime
    '''
    function-计算当前时间
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):
    import datetime
    '''
    function-计算持续时间
    
    Paras:
    start_time - 开始时间
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)