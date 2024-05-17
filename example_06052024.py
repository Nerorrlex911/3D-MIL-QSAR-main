#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Multi-instance (MI) machine learning approaches can be used to solve the issues of representation of each molecule by multiple conformations (instances) and automatic selection of the most relevant ones. In the multi-instance approach, an example (i.e., a molecule) is presented by a bag of instances (i.e., a set of conformations), and a label (a molecule property value) is available only for a bag (a molecule), but not for individual instances (conformations).
# 
# In this study, we have implemented several multi-instance algorithms, both conventional and based on deep learning, and investigated their performance. We have compared the performance of MI-QSAR models with those based on the classical single-instance QSAR (SI-QSAR) approach in which each molecule is encoded by either 2D descriptors computed for the corresponding molecular graph or 3D descriptors issued for a single lowest-energy conformation. 
# 
# <img src="img/toc.png" width="600"/>

# # Descriptors
# 
# Сonformations representing each molecule were generated using the algorithm implemented in RDKit. In our study, we generated up to 100 conformations and removed conformations with RMSD values below 0.5Å to the remaining ones to reduce redundancy.For the descriptor representation of conformations, we used previously developed 3D pharmacophore signatures. Each conformation is represented by a set of pharmacophore features (H-bond donor/acceptor, the center of positive/negative charge, hydrophobic, and aromatic) determined by applying the corresponding SMARTS patterns. All possible quadruplets of features of a particular conformation were enumerated. Distances between features were binned to allow fuzzy matching of quadruplets with small differences in the position of features. Here we used the 1Å bin step as it demonstrated reasonable performance in our previous studies. These signatures consider distances between features and their spatial arrangement to recognize the stereo configuration of quadruplets. We counted the number of identical 3D pharmacophore quadruplet signatures for each conformation and used the obtained vectors as descriptors for model building. 3D pharmacophore descriptors used in this study were implemented in the pmapper Python package (https://github.com/DrrDom/pmapper). 
# To build 2D models, we chose binary Morgan fingerprints (MorganFP) of radius 2 and size 2048 calculated with RDKit because they are widely used 2D descriptors. For comparative purpose we also used 2D physicochemical descriptors (PhysChem) and binary 2D pharmacophore fingerprints (PharmFP) calculated with RDKit.

# # 0. Code installation
# 
# Using **conda** and **pip** is the easiest way to install all required packages. Create a new environment (named "exp") with Python 3.6. Note the issues related to RDKit installation https://stackoverflow.com/questions/70202430/rdkit-importerror-dll-load-failed. <br/>
# 
# Run these commands in the command line:
# 
# `conda create -n exp python=3.6`<br/>
# `conda activate exp` <br/>
# 
# Install RDKit package: <br/>
# 
# `conda install -c conda-forge rdkit` <br/>
# 
# Install PyTorch package: <br/>
# 
# `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` <br/>
# `pip install torch_optimizer` <br/>
# 
# Install software to calculate 3D pmapper descriptors: <br/>
# 
# `conda install -c conda-forge openbabel` <br/>
# ` pip install networkx` <br/>
# `pip install pmapper` <br/>

# # 1. Descriptor calculation

# In[ ]:


import os
from miqsar.utils import calc_3d_pmapper

#Choose dataset to be modeled and create a folder where the descriptors will be stored

nconfs_list = [1, 5] #number of conformations to generate; calculation is time consuming, so here we set 5, for real tasks set 25..100
ncpu = 2 # set number of CPU cores 

dataset_file = os.path.join('datasets', 'CHEMBL1075104.smi')
descriptors_folder = os.path.join('descriptors')
# os.mkdir(descriptors_folder)

out_fname = calc_3d_pmapper(input_fname=dataset_file, nconfs_list=nconfs_list, energy=100,  descr_num=[4], ncpu=ncpu, path=descriptors_folder)


# The descriptor folder contains several files:
# 
# `conf-CHEMBL1075104_1.pkl` - pickle file with RDKit the lowest-energy conformations<br/>
# `conf-CHEMBL1075104_5.pkl` - pickle file with RDKit the generated conformations<br/>
# `conf-5_CHEMBL1075104_log.pkl` - pickle file with the conformation energies<br/>
# 
# `PhFprPmapper_conf-CHEMBL1075104_1.txt` - SVM file with pmapper 3D descriptors for the lowest-energy conformatons<br/>
# `PhFprPmapper_conf-CHEMBL1075104_1.colnames` - names of pmapper 3D descriptors for the lowest-energy conformatons<br/>
# `PhFprPmapper_conf-catalyst_data_1.rownames` - ids of the lowest-energy conformatons<br/>
# 
# `PhFprPmapper_conf-CHEMBL1075104_5.txt` - SVM file with pmapper 3D descriptors for generated conformations<br/>
# `PhFprPmapper_conf-CHEMBL1075104_5.colnames` - names of pmapper 3D descriptors for generated conformations<br/>
# `PhFprPmapper_conf-CHEMBL1075104_5.rownames` - ids of generated conformations<br/>

# # 2. Preparation of training and test set

# In[ ]:


import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_svm_data(fname):
    
    def str_to_vec(dsc_str, dsc_num):

        tmp = {}
        for i in dsc_str.split(' '):
            tmp[int(i.split(':')[0])] = int(i.split(':')[1])
        #
        tmp_sorted = {}
        for i in range(dsc_num):
            tmp_sorted[i] = tmp.get(i, 0)
        vec = list(tmp_sorted.values())

        return vec
    
    #
    with open(fname) as f:
        dsc_tmp = [i.strip() for i in f.readlines()]

    with open(fname.replace('txt', 'rownames')) as f:
        mol_names = [i.strip() for i in f.readlines()]
    #
    labels_tmp = [float(i.split(':')[1]) for i in mol_names]
    idx_tmp = [i.split(':')[0] for i in mol_names]
    dsc_num = max([max([int(j.split(':')[0]) for j in i.strip().split(' ')]) for i in dsc_tmp])
    #
    bags, labels, idx = [], [], []
    for mol_idx in list(np.unique(idx_tmp)):
        bag, labels_, idx_ = [], [], []
        for dsc_str, label, i in zip(dsc_tmp, labels_tmp, idx_tmp):
            if i == mol_idx:
                bag.append(str_to_vec(dsc_str, dsc_num))
                labels_.append(label)
                idx_.append(i)
                
        bags.append(np.array(bag).astype('uint8'))
        labels.append(labels_[0])
        idx.append(idx_[0])

    return np.array(bags), np.array(labels), np.array(idx)


# split data into a training and test set
dsc_fname = os.path.join('descriptors', 'PhFprPmapper_conf-CHEMBL1075104_5.txt') # descriptors file
bags, labels, idx = load_svm_data(dsc_fname)
print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')

# 首先，将数据集划分为训练集和测试集（70%训练，30%测试）
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx, test_size=0.3, random_state=42)
# 接下来，将测试集划分为验证集和测试集（10%验证，20%测试）
x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(x_test, y_test, idx_test, test_size=0.67, random_state=42)

print(f'There are {len(x_train)} training molecules and {len(x_test)} test molecules and {len(x_val)} validation molecules')


# # 3. Model training

# For better training of the neural network, the descriptors should be scaled:

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def scale_data_simple(x_train, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)
    return np.array(x_train_scaled), np.array(x_test_scaled)

def scale_data(x_train, x_val, x_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_val):
        x_val_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)
    return np.array(x_train_scaled), np.array(x_val_scaled), np.array(x_test_scaled)


x_train_scaled, x_val_scaled, x_test_scaled = scale_data(x_train, x_val, x_test)


# One should implement a protocol for optimizing the hyperparameters of the neural network. Here we assign the default hyperparameters found with the grid search technique.

# In[ ]:


from miqsar.estimators.wrappers import InstanceWrapperMLPRegressor
from miqsar.estimators.utils import set_seed
set_seed(43)

ndim = (x_train_scaled[0].shape[1], 512, 256, 128, 64) # number of hidden layers and neurons in the main network
pool = 'mean'                                     # type of pulling of instance descriptors
n_epoch = 1000                                    # maximum number of learning epochs
lr = 0.01                                        # learning rate
weight_decay = 0.001                              # l2 regularization
batch_size = 99999999                             # batch size
init_cuda = False                                  # True if GPU is available


net = InstanceWrapperMLPRegressor(ndim=ndim, pool=pool, init_cuda=init_cuda)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, explained_variance_score
from miqsar.estimators.earlystop import EarlyStopping
import torch
def cross_val(x,y,model,k=10,epochs=100):
    np.random.seed(0)
    num_val = len(x)//k
    r2 = []
    for i in range(k):
        print('Processing fold #',i)
        val_x = x[num_val*i:num_val*(i+1)]
        val_y = y[num_val*i:num_val*(i+1)]

        train_x = np.concatenate([x[:num_val*i],x[(i+1)*num_val:]],axis=0)
        train_y = np.concatenate([y[:num_val*i],y[(i+1)*num_val:]],axis=0)
        
        train_x_scaled, val_x_scaled = scale_data_simple(train_x, val_x)
        """
        val_y_pred = model.predict(val_x_scaled)
        mae_val = mean_absolute_error(val_y, val_y_pred)
        train_y_pred = model.predict(train_x_scaled)
        mae_train = mean_absolute_error(train_y, train_y_pred)
        all_scores.append([mae_val,mae_train])
        """
        pred_train = model.predict(train_x_scaled)
        pred_val = model.predict(val_x_scaled)
        r2_train = r2_score(train_y,pred_train)
        r2_val = r2_score(val_y,pred_val)
        r2.append([r2_train,r2_val])
    return r2
earlystopping = EarlyStopping(patience=30,verbose=True)
train_loss = []
test_loss = []
val_loss = []
for epoch in range(n_epoch):
    net.fit(x_train_scaled, y_train, 
        n_epoch=1, 
        lr=lr,
        weight_decay=weight_decay,
        batch_size=16)
    net.estimator.eval()
    with torch.no_grad():
        yval_pred = net.predict(x_val_scaled)
        val_loss.append(mean_absolute_error(y_val,yval_pred))
        avg_val_loss = np.average(val_loss)
        earlystopping(val_loss=avg_val_loss,model=net.estimator)
        if earlystopping.early_stop:
            print("此时早停。")
            break

import csv

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error

y_test_pred = net.predict(x_test_scaled)
y_val_pred = net.predict(x_val_scaled)
y_train_pred = net.predict(x_train_scaled)

error_list = ["r2_score","mean_absolute_error","mean_squared_error","explained_variance_score","median_absolute_error"]
tensor_list = ["train","val","test"]
import pandas as pd
model_description = pd.DataFrame(columns=error_list,index=tensor_list)
for ierror,error in enumerate(error_list):
    for itensor,tensor in enumerate(tensor_list):
        model_description.iloc[itensor,ierror] = eval(f'{error}(y_{tensor},y_{tensor}_pred)')
model_description.to_csv(f'model_description_{lr}_{weight_decay}.csv')




