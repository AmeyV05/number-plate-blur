'''
Description: An off the shelf car detecting model from dectectron2 model zoo is used here.
WE used the mask rcnn model. This script is made using that model.
# Created By: Amey Vasulkar
# Created On: 2023-10-19
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
#%% importing libraries
import os
import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
##Jupyter interactive widgets
from IPython.display import display, Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper, MetadataCatalog, DatasetCatalog


