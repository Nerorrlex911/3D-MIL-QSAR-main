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
ncpu = 4 # set number of CPU cores 

#dataset_file = os.path.join('datasets', 'predict.smi')
#print(predict.smi)
dataset_file='./datasets/predict.smi'
descriptors_folder = os.path.join('descriptors')
# os.mkdir(descriptors_folder)

out_fname = calc_3d_pmapper(input_fname=dataset_file, nconfs_list=nconfs_list, energy=100,  descr_num=[4],
                            path=descriptors_folder, ncpu=ncpu)
#print(input_fname)
print(out_fname)

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
dsc_fname = os.path.join('descriptors', 'PhFprPmapper_conf-predict_5.txt') # descriptors file
bags, labels, idx = load_svm_data(dsc_fname)
print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')


# # 3. Model training

# For better training of the neural network, the descriptors should be scaled:

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def scale_predict(x_train):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = x_train.copy()
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    return np.array(x_train_scaled)

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

x_scaled = scale_predict(bags)


# One should implement a protocol for optimizing the hyperparameters of the neural network. Here we assign the default hyperparameters found with the grid search technique.

# In[ ]:

import pickle
import csv
from miqsar.estimators.utils import set_seed
set_seed(43)

with open(f'model_0.01_0.001.pkl', 'rb') as model_file:
    net = pickle.load(model_file)
net.init_cuda = False
y_pred = net.predict(x_scaled)

# 指定要保存的CSV文件名
pred_path = f'pred.csv'

# 将数组写入CSV文件
with open(pred_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(y_pred.reshape(-1,1))

