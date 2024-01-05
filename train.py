'''
Description: Training script
# Created By: Amey Vasulkar
# Created On: 2023-10-17
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
#%%
import os

import torch
torch.cuda.empty_cache()

# Set the max_split_size_mb value (in megabytes)
torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmark for faster training
torch.cuda.empty_cache()  # Empty the GPU cache
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from modules import register_datasets
from modules.config import setup
from modules.trainer_evaluator import MyNumberPlateTrainer,dumpConfig,generate_run_name,trainset_and_time_info
from detectron2.data import DatasetCatalog, MetadataCatalog
#%%

## Variables. Still implmenet the gridsearch here
SOLVER_IMS_PER_BATCH = 1
SOLVER_BASE_LR = 0.0025
SOLVER_MAX_ITER = 3000
MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE = 64
MODEL_ROI_HEADS_NUM_CLASSES = 1
MODEL_POINT_HEAD_NUM_CLASSES = 1
run_name = None



#Registering the available datasets with annotations
std_sets = [i for i in MetadataCatalog.list()]

root_dir='/media/share/datastore/DS_DataBank/CVAT_Tasks'
number_plate_annotation_dir=os.path.join(root_dir, 'VehicleNumberPlate')

train_data_directory = number_plate_annotation_dir
register_datasets.register_mlt_coco_datasets(train_data_directory)
#%%
custom_sets = [i for i in MetadataCatalog.list()]
mysets = [i for i in custom_sets if i not in std_sets]
train_set= [i for i in mysets if 'train' in i]
test_set= [i for i in mysets if 'test' in i]
#%%
#Defining a modelname/runname
while run_name == None:
    run_name = generate_run_name()
    if os.path.exists(os.getcwd() + os.path.sep + run_name):
        run_name = None

assert len(custom_sets) > 1, "We found no available data for your training..."

#%%
cfg=setup(SOLVER_IMS_PER_BATCH,
            SOLVER_BASE_LR,
            SOLVER_MAX_ITER,
            MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE,
            MODEL_ROI_HEADS_NUM_CLASSES,
            MODEL_POINT_HEAD_NUM_CLASSES,
            run_name,
            train_set,
            test_set)
trainset_and_time_info(cfg, train_set)

def do_train(cfg):
    
    trainer = MyNumberPlateTrainer(cfg)
    trainer.resume_or_load(resume=False)
    dumpConfig(cfg.OUTPUT_DIR, cfg)
    trainer.train()

do_train(cfg)

# %%
