'''
Description: Setting up the configuration for the model.
# Created By: Amey Vasulkar
# Created On: 2023-10-17
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
import os
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2 import model_zoo


def setup(SOLVER_IMS_PER_BATCH:int,
            SOLVER_BASE_LR:float,
            SOLVER_MAX_ITER:int,
            MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE:int,
            MODEL_ROI_HEADS_NUM_CLASSES:int,
            MODEL_POINT_HEAD_NUM_CLASSES:int,
            run_name:str,
            trainsets:list,
            testsets:list):

    cfg = get_cfg()

    cfg.MODEL.DEVICE= 'cuda'
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     cfg.merge_from_file (model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS=('/home/amey/Projects/number-plate-blur/output/Lambda-4/model_final.pth')

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS=('/home/amey/Projects/number_plate_blurring/output/Jitter-3/model_final.pth')

    
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = tuple(trainsets)
    cfg.DATASETS.TEST = tuple(testsets)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.SOLVER_STEPS=[]
    cfg.SOLVER.STEPS = (2000,)

        
    ### - Train for COCO - ###

    cfg.SOLVER.IMS_PER_BATCH = SOLVER_IMS_PER_BATCH  
    cfg.SOLVER.BASE_LR = SOLVER_BASE_LR
    cfg.SOLVER.MAX_ITER = SOLVER_MAX_ITER

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = MODEL_ROI_HEADS_NUM_CLASSES
 
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + os.path.sep + run_name

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg