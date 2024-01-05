'''
Description: 
# Created By: Amey Vasulkar
# Created On: 2023-10-17
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
#%%
import os
import cv2
import copy
import torch
import random

from datetime import datetime

import matplotlib.pyplot as plt
from PIL import ImageFile, Image


ImageFile.LOAD_TRUNCATED_IMAGES = True
import detectron2
from detectron2 import model_zoo
from modules.config import setup
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from modules import register_datasets
from detectron2.config import get_cfg


#%%

#%%

##Register


#Registering the available datasets with annotations
std_sets = [i for i in MetadataCatalog.list()]

root_dir='/media/share/datastore/DS_DataBank/CVAT_Tasks'
number_plate_annotation_dir=os.path.join(root_dir, 'VehicleNumberPlate')

train_data_directory = number_plate_annotation_dir
register_datasets.register_mlt_coco_datasets(train_data_directory)
custom_sets = [i for i in MetadataCatalog.list()]
mysets = [i for i in custom_sets if i not in std_sets]
train_set= [i for i in mysets if 'train' in i]
test_set= [i for i in mysets if 'test' in i]

#%%
## Variables. Still implmenet the gridsearch here

run_name = 'Kappa-3'
cfg = get_cfg()
cfg.INPUT.MASK_FORMAT= 'bitmask'
# cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
#%%
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
MODEL='/home/amey/Projects/number-plate-blur/output/'+run_name
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, os.path.join(MODEL,"model_final.pth"))  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8# set a custom testing threshold
# point_rend.add_pointrend_config(cfg)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)
# %%

#%%
dataset_dicts=DatasetCatalog.get('AA7_val')
number_plate_metadata=MetadataCatalog.get('AA4_val')
# dataset_dicts=DatasetCatalog.get('AA1_train')
# number_plate_metadata=MetadataCatalog.get('AA1_train')
#%%
# test_fname='Job_Bizetbuurt_EO_Track01_Left Backward_00146.jpg'
save_dir=os.path.join(MODEL, 'test_inference')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for d in dataset_dicts[:]:
    print(d["file_name"])
    # if d["file_name"].split('/')[-1]==test_fname:
    im=cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                metadata=number_plate_metadata, 
                scale=0.75)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img=out.get_image()[:, :, ::-1]
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #show high res image
    plt.figure(figsize=(20,20))
    #save the image in a directory called 'test_inference' 
    #in the model folder
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, d["file_name"].split('/')[-1]), bbox_inches='tight', pad_inches=0)
    plt.close()
    # # break
        
# %%
