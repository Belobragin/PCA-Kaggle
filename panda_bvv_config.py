#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Local or AWS machine:
#DEBUG = False
DEBUG = True


# In[2]:


#DL machine or not:
#DLM = False
DLM = True


# In[3]:


import os, sys, json, resource
import pandas as pd
import numpy as np
from random import shuffle


# In[4]:



# In[5]:


isup_classes= ['0', '1', '2', '3', '4', '5']
gs_classes = ['0', '3', '4', '5']
gs_scores = ['0_0', '3_3', '3_4', '4_3', '4_4', '3_5', '4_5', '5_4', '5_5']
choices_=[0, 1, 2, 3, 4, 5]
num_classes = 6
npseed = 136
random_state_split=101011
val_size_proportion = 0.15


# In[23]:


#(2437+3563+2939+859)/22


# In[24]:


#(431+629+519+152)/22


# In[6]:


isup_class_weights  = {0: 0.6118, 1: 0.66367, 2: 1.31745, 3: 1.42458, 4: 1.4166, 5: 1.44553}
gl_class_weights = {0: 1.03311, 1: 0.68307, 2: 0.83271, 3: 2.72356}
isup_bias = np.array([2.448, 2.367, 1.681, 1.603, 1.608, 1.588])
trivial_isup_bias = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
trivial_gl_class_bias = np.array([1.0, 1.0, 1.0, 1.0])


# In[7]:


trivial_class_weights_gleason = {
 0:1.0,
 1:1.0,
 2:1.0,
 3:1.0,
 4:1.0,
 5:1.0,
 6:1.0,
 7:1.0,
 8:1.0,  
}
gl_score_bias = np.array([2.608, 2.5267, 1.84, -0.549, 1.764, 1.665, 1.382, 0.152, -0.518])


# In[8]:


if DEBUG:
    #base folders for local:
    base_path = "/run/media/admin/kagg/panda"
    note_path = "/home/admin/pca"
    nn_path = "/run/media/admin/kagg/nn"
    test_size1 = os.path.join(base_path, 'test_size1')
    test_size2 = os.path.join(base_path, 'test_size2')
else:
    # base folders for AWS:
    base_path = "/kagg/ebsvol/contest/panda"
    note_path = "/kagg/ebsvol/mynote/panda_notes"
    train_size1 = os.path.join(base_path, 'train_size1')
    train_size2 = os.path.join(base_path, 'train_size2') 


# In[9]:


temp_path = os.path.join(base_path, 'temp')
model_path = os.path.join(base_path, 'models')
gleason_path = os.path.join(base_path, 'gs') #this is for gleason CLASSES, i.e 0, 3, 4, 5


# In[10]:


#resized images folders
#! - ALL images in these base folders on the local WS are rotated aka h > w

#ALL images folders:
test_cnn = os.path.join(base_path, 'testf')
train_cnn = os.path.join(base_path, 'trainf')
valid_cnn = os.path.join(base_path, 'validf')
#ALL masks with size1, size2
mask_size1 = os.path.join(base_path, 'mask_size1')
mask_size2 = os.path.join(base_path, 'mask_size2')


# In[11]:


#base dataframes with data:
primary_train_labels = pd.read_csv(os.path.join(base_path, 'train.csv')) #original df, don't touch
train_labels = pd.read_csv(os.path.join(base_path, 'train_corr.csv')) #some useful columns added, ALL rows
mask_labels = pd.read_csv(os.path.join(base_path, 'mask_labels.csv'))
test_cnn_labels = pd.read_csv(os.path.join(base_path, 'test_cnn_labels.csv'))
test_gleason_labels = pd.read_csv(os.path.join(base_path, 'gleason_test.csv'))
gl_class_labels = pd.read_csv(os.path.join(base_path, 'gl_class.csv'))
gl_score_labels = pd.read_csv(os.path.join(base_path, 'gl_score.csv'))


# In[12]:


cancer_s2 = os.path.join(base_path, 'cancer_s2')
cancer_s1 = os.path.join(base_path, 'cancer_s1')


# In[13]:


if DLM:
    id_label_map_gl_class = {k:v for k,v in zip(gl_class_labels.gl_id.values,                                           gl_class_labels.gleason_score.values)}
    id_label_map_gl_scores = {k:v for k,v in zip(gl_score_labels.image_id.values,                                           gl_score_labels.gleason_score.values)}
    id_label_map_isup = {k:v for k,v in zip(train_labels.image_id.values, train_labels.isup_grade.values)}
    
    from bvv_utils import *
    


# In[14]:


#CNN training:
if DLM:
    train_dict = {'effnB0_test':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':2,
                            'num_earlyStop':2,
                            'num_reduceOnPlateu':8,
                            'learn_rate':5e-4,
                            'stop_patience':14,
                            'inp_label_smooth':0.01,
                            'BS': 10,
                            's_per_epoch':20,
                            'val_steps':8,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB0_panda_check',
                            'weights_file': 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':None,
                            'file_for_struct':'model_effnB0_panda_struct.json',
                            'file_for_weights':'model_effnB0_panda_weights.h5',
                            'history_file':'history_effnB0.json',
                            'save_plot_file':'plot_edu_effnb0.png',
                            'from_folder_train':'testdata_grey/gs2_16x320',
                            'from_folder_val':'testdata_grey/gs2_16x320',   
                            'num_logits':6,
                            'trdatagen': LightImgAugDataGeneratorMC,
                            'valdatagen':LightImgAugDataGeneratorMC,
                            },
                  'effnB0':{'image_sizey':512,
                            'image_sizex':512,
                            'num_epochs':50,
                            'num_reduceOnPlateu':10,
                            'learn_rate':3e-3,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':400,
                            'val_steps':71,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':trivial_isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB0_panda_check',
                            'weights_file': 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':None,
                            'file_for_struct':'model_effnB0_panda_struct.json',
                            'file_for_weights':'model_effnB0_panda_weights.h5',
                            'history_file':'history_effnB0.json',
                            'save_plot_file':'plot_edu_effnb0.png',
                            'from_folder_train':'ts1_16x512',
                            'from_folder_val':'ts1_16x512',   
                            'num_logits':6,
                            'trdatagen': DeepImgAugDataGeneratorLR,
                            'valdatagen':DeepImgAugDataGeneratorLR,
                            },
                'effnB3_gs_test':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':2,
                            'num_reduceOnPlateu':6,
                            'learn_rate':1e-4,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 4,
                            's_per_epoch':8,
                            'val_steps':3,
                            'id_label_map':id_label_map_isup,
                            'class_weights':gl_class_weights,
                            'output_bias':trivial_gl_class_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':'effnB3_check20_best_level0_weights.npy',
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'gs_proc_inv',
                            'from_folder_val':None,
                            'num_logits':4,
                            'trdatagen': classic_train_datagen,
                            'valdatagen':classic_val_datagen,
                            },
                   'effnB3_gs':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':8,
                            'num_reduceOnPlateu':6,
                            'learn_rate':1e-4,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':445,
                            'val_steps':78,
                            'id_label_map':id_label_map_isup,
                            'class_weights':gl_class_weights,
                            'output_bias':trivial_gl_class_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':'effnB3_check20_best_level0_weights.npy',
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'gs_proc_inv',
                            'from_folder_val':None,
                            'num_logits':4,
                            'trdatagen': classic_train_datagen,
                            'valdatagen':classic_val_datagen,
                            },
                  'effnB3_da':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':30,
                            'num_reduceOnPlateu':6,
                            'learn_rate':1e-4,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':400,
                            'val_steps':71,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':trivial_isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':'model30_weights.h5',
                            'level0_file':None,
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'ts2_16x320',
                            'from_folder_val':None,
                            'num_logits':6,
                            'trdatagen': LightImgAugDataGeneratorMC,
                            'valdatagen':LightImgAugDataGeneratorMC,
                            },
                  #this option is to educate best model on samples from canser_s2 folder, but
                  #validate on odinary samples from 
                  'effnB3_cs':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':40,
                            'num_reduceOnPlateu':6,
                            'learn_rate':3e-3,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':400,
                            'val_steps':71,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':'model20_weights.h5',
                            'level0_file':None,
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'cs2_16x320',
                            'from_folder_val':'ts2_16x320',
                            'num_logits':6,
                            'trdatagen': LightImgAugDataGeneratorMC,
                            'valdatagen':LightImgAugDataGeneratorMC,
                            },
                'effnB3_grey':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':40,
                            'num_reduceOnPlateu':8,
                            'learn_rate':3e-3,
                            'stop_patience':18,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':400,
                            'val_steps':71,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':'model20_weights.h5',
                            'level0_file':None,
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'grey2_16x320',
                            'from_folder_val':'grey2_16x320',
                            'num_logits':6,
                            'trdatagen': LightImgAugDataGeneratorMC,
                            'valdatagen':LightImgAugDataGeneratorMC,
                            },
                  'effnB3regr':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':20,
                            'num_earlyStop':20,
                            'num_reduceOnPlateu':8,
                            'learn_rate':5e-5,
                            'stop_patience':14,
                            'inp_label_smooth':0.01,
                            'BS': 22,
                            's_per_epoch':400,
                            'val_steps':72,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':'effnB3_check20_best_level0_weights.npy',
                            'file_for_struct':'model_effnB3regr_panda_struct.json',
                            'file_for_weights':'model_effnB3regr_panda_weights.h5',
                            'history_file':'history_effnB3regr.json',
                            'save_plot_file':'plot_edu_effnb3regr.png',
                            'from_folder_train':'ts2_16x320_inv',
                            'num_logits':6,
                            'trdatagen': DeepImgAugDataGeneratorLR,
                            'valdatagen':DeepImgAugDataGeneratorLR,
                            },
                  'effnB3regr_test':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':2,
                            'num_earlyStop':2,
                            'num_reduceOnPlateu':8,
                            'learn_rate':5e-4,
                            'stop_patience':14,
                            'inp_label_smooth':0.01,
                            'BS': 10,
                            's_per_epoch':20,
                            'val_steps':8,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':None,
                            'level0_file':'effnB3_check20_best_level0_weights.npy',
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights.h5',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'testdata320/testf',
                            'num_logits':6,
                            'trdatagen': DeepImgAugDataGeneratorLR,
                            'valdatagen':DeepImgAugDataGeneratorLR,
                            },
                  'effnB3_test':{'image_sizey':320,
                            'image_sizex':320,
                            'num_epochs':2,
                            'num_earlyStop':2,
                            'num_reduceOnPlateu':8,
                            'learn_rate':5e-4,
                            'stop_patience':14,
                            'inp_label_smooth':0.01,
                            'BS': 10,
                            's_per_epoch':20,
                            'val_steps':8,
                            'id_label_map':id_label_map_isup,
                            'class_weights':isup_class_weights,
                            'output_bias':isup_bias,
                            'model_name': 'model_panda.h5',
                            'checkpoint_name': 'model_effnB3_panda_check',
                            'weights_file': 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                            'bestmodel_weights':'best12_weights.h5',
                            'level0_file':None,
                            'file_for_struct':'model_effnB3_panda_struct.json',
                            'file_for_weights':'model_effnB3_panda_weights',
                            'history_file':'history_effnB3.json',
                            'save_plot_file':'plot_edu_effnb3.png',
                            'from_folder_train':'gs2_16x320',
                            'from_folder_val':'gs2_16x320',   
                            'num_logits':6,
                            'trdatagen': LightImgAugDataGeneratorMC,
                            'valdatagen':LightImgAugDataGeneratorMC,
                            },
                  }


# In[15]:


# 'effnB2_da_now':{'image_sizey':260,
#                             'image_sizex':260,
#                             'num_epochs':40,
#                             'num_reduceOnPlateu':15,
#                             'learn_rate':3e-3,
#                             'stop_patience':34,
#                             'inp_label_smooth':0.01,
#                             'BS': 32,
#                             's_per_epoch':280,
#                             'val_steps':49,
#                             'id_label_map':id_label_map_gl_scores,
#                             'class_weights':trivial_class_weights_gleason,
#                             'output_bias':gl_score_bias,
#                             'model_name': 'model_panda.h5',
#                             'checkpoint_name': 'model_effnB2_panda_check',
#                             'weights_file': 'efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
#                             'bestmodel_weights':None,
#                             'level0_file':None,
#                             'file_for_struct':'model_effnB2_panda_struct.json',
#                             'file_for_weights':'model_effnB2_panda_weights.json',
#                             'history_file':'history_effnB2.json',
#                             'save_plot_file':'plot_edu_effnb2.png',
#                             'from_folder_train':'ts2_16x260_inv',
#                             'num_logits':9,
#                             'trdatagen': LightImgAugDataGeneratorMC,
#                             'valdatagen':LightImgAugDataGeneratorMC,
#                             },
#                   'effnB2_test':{'image_sizey':260,
#                             'image_sizex':260,
#                             'num_epochs':2,
#                             'num_earlyStop':2,
#                             'num_reduceOnPlateu':8,
#                             'learn_rate':3e-3,
#                             'stop_patience':14,
#                             'inp_label_smooth':0.01,
#                             'BS': 10,
#                             's_per_epoch':12,
#                             'val_steps':4,
#                             'id_label_map':id_label_map_gl_scores,
#                             'class_weights':trivial_class_weights_gleason,
#                             'output_bias':gl_score_bias,
#                             'model_name': 'model_panda.h5',
#                             'checkpoint_name': 'model_effnB2_panda_check',
#                             'weights_file': 'efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
#                             'bestmodel_weights':None,
#                             'level0_file':None,
#                             'file_for_struct':'model_effnB2_panda_struct.json',
#                             'file_for_weights':'model_effnB2_panda_weights.json',
#                             'history_file':'history_effnB2.json',
#                             'save_plot_file':'plot_edu_effnb2.png',
#                             'from_folder_train':'testdata256/testf',
#                             'num_logits':9,
#                             'trdatagen': LightImgAugDataGeneratorMC,
#                             'valdatagen':LightImgAugDataGeneratorMC,    
#                             },


# In[17]:


import os
module_name = 'panda_bvv_config'

