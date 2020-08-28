#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys, json, resource
import pandas as pd
import numpy as np


# In[2]:


isup_classes= ['0', '1', '2', '3', '4', '5']
num_classes = 6
panda_seed = 137


# In[1]:


#base folders:
nn_path = "/run/media/admin/kagg/nn"
base_path = "/run/media/admin/kagg/panda"
note_path = "/home/admin/pca"
temp_path = os.path.join(base_path, 'temp')

train_path = os.path.join(base_path, 'orig_images') #24 samples on local WS
mask_path = os.path.join(base_path, 'orig_masks') #24 samples on local WS


# In[ ]:


#resized images folders
#! - ALL images in these base folders on the local WS are rotated aka h > w

#ALL images folders:
train_size1 = os.path.join(base_path, 'train_size1')
train_size2 = os.path.join(base_path, 'train_size2')

#ALL masks with size1, size2
mask_size1 = os.path.join(base_path, 'mask_size1')
mask_size2 = os.path.join(base_path, 'mask_size2')


# In[6]:


#base folders for processed images:

#no-white tiled images
train_nwh_small_size1 = os.path.join(base_path, 'train_nwh_small_size1')
train_nwh_large_size1 = os.path.join(base_path, 'train_nwh_large_size1')
train_nwh_small_size2 = os.path.join(base_path, 'train_nwh_small_size2')
train_nwh_large_size2 = os.path.join(base_path, 'train_nwh_large_size2')

#no-white tiled images after mask application:
train_s2_isup = os.path.join(base_path, 'train_s2_isup')
train_s1_isup = os.path.join(base_path, 'train_s1_isup')

#cacerous areas:
cancer_size2 = os.path.join(base_path, 'cancer_size2')


# In[5]:


#folder with resized processed images, for CNN education:
#!: these images are resized to equal size (and rotated). This allows to pay less for expensive GPU machine
train_size2_prepare = os.path.join(base_path, 'train_size2_prepare')

train_nwh_large_size2_prepare = os.path.join(base_path, 'train_nwh_large_size2_prepare')
train_nwh_large_size2_prepare_inv = os.path.join(base_path, 'train_nwh_large_size2_prepare_inv')
train_size2_16x512_inv = os.path.join(base_path, 'train_size2_16x512_inv')
train_size2_16x256_inv = os.path.join(base_path, 'train_size2_16x256_inv')
train_size2_16x128_inv = os.path.join(base_path, 'train_size2_16x128_inv')
train_size2_16x320_inv = os.path.join(base_path, 'train_size2_16x320_inv')

#no-white tiled images after mask application:
train_s2_isup_prepare = os.path.join(base_path, 'train_s2_isup_prepare')
train_s1_isup_prepare = os.path.join(base_path, 'train_s1_isup_prepare')


# In[5]:


#base dataframes with data:
primary_train_labels = pd.read_csv(os.path.join(base_path, 'train.csv')) #original df, don't touch
train_labels = pd.read_csv(os.path.join(base_path, 'train_corr.csv')) #some useful columns added, ALL rows
mask_labels = pd.read_csv(os.path.join(base_path, 'mask_labels.csv'))
test_cnn_labels = pd.read_csv(os.path.join(base_path, 'test_cnn_labels.csv'))


# In[8]:


#base files lists:
prime_trains = os.listdir(train_path)
prime_masks = os.listdir(mask_path)


# In[ ]:


#CNN train, validation and test folds:
#!: for test purpose only, no more then 2 epochs

train_cnn = os.path.join(base_path, 'trainf')
valid_cnn = os.path.join(base_path, 'validf')
test_cnn = os.path.join(base_path, 'testf')
test_no_classes_cnn = os.path.join(base_path, 'testf_no_classes')


# In[9]:


def getmem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0,1)
    )


# In[1]:


import os
module_name = 'panda_bvv_config'

