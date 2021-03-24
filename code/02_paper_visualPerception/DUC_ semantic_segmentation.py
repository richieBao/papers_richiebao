# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:47:20 2021

@author: richie bao-paper data processing
"""
import mxnet as mx
import cv2 as cv
import numpy as np
import os
from PIL import Image
import math
from collections import namedtuple
from mxnet.contrib.onnx import import_model
import cityscapes_labels

import util
from config import config

def preprocess(im,rgb_mean):
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    # convert to ndarray
    test_img = mx.ndarray.array(test_img)
    return test_img

def get_palette():
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def predict(imgs,result_shape,im):
    # get input and output dimensions
    result_height, result_width = result_shape
    _, _, img_height, img_width = imgs.shape
    # set downsampling rate
    ds_rate = 8
    # set cell width
    cell_width = 2
    # number of output label classes
    label_num = 19
    
    # Perform forward pass
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([imgs]),is_train=False)
    labels = mod.get_outputs()[0].asnumpy().squeeze()

    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width),:int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])
    
    # get softmax output
    softmax = labels
    
    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results

    # comput confidence score
    confidence = float(np.max(softmax, axis=0).mean())

    # generate segmented image
    result_img = Image.fromarray(colorize(raw_labels)).resize(result_shape[::-1])
    
    # generate blended image
    blended_img = Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0))

    return confidence, result_img, blended_img, raw_labels

def get_model(ctx, model_path):
    # import ONNX model into MXNet symbols and params
    sym,arg,aux = import_model(model_path)
    # define network module
    mod = mx.mod.Module(symbol=sym, data_names=['data'], context=ctx, label_names=None)
    # bind parameters to the network
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, im.shape[0], im.shape[1]))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg, aux_params=aux,allow_missing=True, allow_extra=True)
    return mod

def img_show(img_fn):
    im=cv.imread(img_fn)[:, :, ::-1]   
    PIL_img=Image.fromarray(im)
    PIL_img.show()
    
def batch_pre(img_fns,seg_sp,blended_seg_sp,seg_raw_fn):    
    from pathlib import Path
    import os
    from tqdm import tqdm
    import pickle
    
    if not os.path.exists(seg_sp):
        os.makedirs(seg_sp)    
    if not os.path.exists(blended_seg_sp):
        os.makedirs(blended_seg_sp)     
    
    seg_raw={}
    for f in tqdm(img_fns):
        im=cv.imread(f)[:, :, ::-1]
        rgb_mean=cv.mean(im)
        pre=preprocess(im,rgb_mean)
        result_shape=[im.shape[0],im.shape[1]]
        conf,result_img,blended_img,raw=predict(pre,result_shape,im)
        
        result_img.save(os.path.join(seg_sp,Path(f).name))
        blended_img.save(os.path.join(blended_seg_sp,Path(f).name))
        seg_raw[Path(f).stem]=raw
        
    with open(seg_raw_fn,'wb') as f:
        pickle.dump(seg_raw,f)
    print("seg_raw has been saved into {}".format(seg_raw_fn))


if __name__ == "__main__":
    fns_rel,fns_abs=util.retrieve_data_fp(config.kitti_imgs_dir)
    # img_show(fns_rel[0])

    if len(mx.test_utils.list_gpus())==0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(0)    
        
    im=cv.imread(fns_rel[0])[:, :, ::-1]    
    mod=get_model(ctx, r'C:\Users\richi\omen_richiebao\omen_github\model_bak\ResNet101_DUC_HDC.onnx')
    print("The model is loaded...")     
    
    batch_pre(fns_rel,config.seg_sp,config.blended_seg_sp,config.seg_raw_fn)
    
    
    
    
    
    
    
