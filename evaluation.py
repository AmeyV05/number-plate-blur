'''
Description: Evaluation of the models to get statistics and metrics.
# Created By: Amey Vasulkar
# Created On: 2023-10-18
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
#%%
import os
import cv2
from modules import register_datasets
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from modules.config import setup
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import json
#%%

#
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
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, os.path.join(MODEL,"model_final.pth"))  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7# set a custom testing threshold
# point_rend.add_pointrend_config(cfg)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)

# Create a COCOEvaluator for your dataset
evaluator = COCOEvaluator("AA7_val", cfg, False, output_dir=MODEL)

#%%

# Create a test data loader for your dataset
data_loader = build_detection_test_loader(cfg, "AA7_val")

#%%
# Run inference and evaluation
predictor=DefaultPredictor(cfg)
results = inference_on_dataset(predictor.model, data_loader, evaluator)
print(results)

#%%
#save the ordered dict in a text file in the output directory

json_string=json.dumps(results)
with open(os.path.join(MODEL,'results.json'), 'w') as f:
   f.write(json_string) 
# %%

# %%
