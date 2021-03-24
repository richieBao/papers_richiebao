# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:38:12 2021

@author: richie bao-paper data processing
ref:PyTorch
"""
import os
from config import config
import util
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 

from torch.optim import Adam
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms as T
from torch.autograd import Variable
import torch
from torchvision.utils import make_grid

from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import shutil

limit_images=30000
# clustering
pca_dim=10 #50
kmeans_clusters=5 #100
# convnet
batch_size=32 #64
num_classes=100 #100

transforms = T.Compose([T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor()])



flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

class imgs_dataset(Dataset):
    def __init__(self, image_paths, transforms=None, labels=[], limit=None,file_type='jpg',shuffle=False):
        self.image_paths=[Path(fp) for fp in image_paths]
        self.file_type=file_type
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.labels = labels
        self.transforms = transforms
        
        self.classes = set([path.parts[-2] for path in self.image_paths])
        self.shuffle=shuffle
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index] if self.labels else 0
        if self.file_type=='npy':
            image_array=np.load(image_path)    
            image=Image.fromarray((image_array*255/np.max(image_array)).astype('uint8'))
            #image=np.load(image_path)
        else:
            image=Image.open(image_path)
        if self.transforms:
            return self.transforms(image), str(image_path),label
        return image,str(image_path),label
            
    def __len__(self):
        return len(self.image_paths) 

def resnet_fine_tuning():
    model=resnet50() #resnet18()  pretrained=True
    model.fc=nn.Linear(2048, num_classes) #512
    model.cuda();
    
    return model

def extract_features(model, dataset, batch_size=32):
    """
    Gets the output of a pytorch model given a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    image_path_list=[]
    for image,image_path,_ in tqdm(loader, desc='extracting features'):
        output = model(Variable(image).cuda())
        features.append(output.data.cpu())
        image_path_list.append([i for i in image_path])
    return torch.cat(features).numpy(),image_path_list

def cluster(pca, kmeans, model, dataset, batch_size, return_features=False):
    features,image_path_list=extract_features(model, dataset, batch_size)  
    reduced=pca.fit_transform(features)
    #reduced=features
    pseudo_labels=list(kmeans.fit_predict(reduced))
    if return_features:
        return pseudo_labels, features,image_path_list
    return pseudo_labels,image_path_list

def show_cluster(cluster, labels, dataset, limit=32,figsize=(15, 10)):
    images = []
    labels = np.array(labels)
    indices = np.where(labels==cluster)[0]
    
    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None
    
    for i in indices[:limit]:
        image,image_path_list, _ = dataset[i]
        images.append(image)
        
    gridded = make_grid(images)
    plt.figure(figsize=figsize)
    plt.title(f'cluster: {cluster}')
    plt.imshow(gridded.permute(1, 2, 0))
    plt.axis('off')
    
def cluster_imgs_sorted_store(pseudo_labels,image_path_list,save_root):
    img_path=[Path(fp) for fp in image_path_list]
    cluster_labels=np.unique(pseudo_labels)
    print('cluster labels:',cluster_labels)
    for idx in tqdm(range(len(img_path))):
        sub_folder=os.path.join(save_root,image_path_list[idx].split(os.sep)[-2])
        # print(sub_folder)
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
            for label in cluster_labels:
                os.mkdir(os.path.join(sub_folder,str(label)))
        # print(img_path[idx])
        shutil.copy(image_path_list[idx],os.path.join(sub_folder,str(pseudo_labels[idx])))  
        # break

if __name__ == "__main__":
    fileType=["jpg"]
    imgs_fn=util.filePath_extraction(config.recs_segs_selection, fileType)
    image_paths=flatten_lst([[os.path.join(k,fn) for fn in v] for k,v in imgs_fn.items()])
    dataset=imgs_dataset(image_paths=image_paths,transforms=transforms,limit=limit_images,file_type='jpg',shuffle=True)
    
    # img, image_path,_ = dataset[100]
    # img.show()
    model=resnet_fine_tuning()
    
    pca=IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
    kmeans=MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=512, init_size=3*kmeans_clusters)
    optimizer=Adam(model.parameters())
    pseudo_labels,features,image_path_list_=cluster(pca, kmeans, model, dataset, batch_size, return_features=True)
    image_path_list=flatten_lst(image_path_list_)

    plt.hist(pseudo_labels, bins=kmeans_clusters)
    plt.title('cluster membership counts'); 
    
    print('clustering frequency:\n',Counter(pseudo_labels).most_common())

    counts=Counter(pseudo_labels)
    for i in range(len(counts.most_common())):        
        show_cluster(counts.most_common()[i][0], pseudo_labels, dataset,limit=200,figsize=(50,50))

    cluster_imgs_sorted_store(pseudo_labels,image_path_list,config.recs_segs_selection_cluster) 
