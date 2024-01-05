'''
Description: Getting the annotated dataset. Registering them in to train test validataion datasets.

# Created By: Amey Vasulkar
# Created On: 2023-10-11
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
##IMPORTS##
#%%
import os
import cv2
import copy
import torch
import random
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

#%%

root_dir='/media/share/datastore/DS_DataBank/CVAT_Tasks'
number_plate_annotation_dir=os.path.join(root_dir, 'VehicleNumberPlate')


#%%
def register_mlt_coco_datasets(number_plate_annotation_dir):
    #Registering the annotations as datasets with train test and validation splits
    #AA1..AA4 are training
    #AA5- is test
    #AA6- is validation
    directory=number_plate_annotation_dir

    for fol in os.listdir(directory):        
        subfol = os.path.join(directory, fol)
        # print(subfol)
        items = os.listdir(subfol)
        annotation=os.path.join(subfol, 'instances_default.json')
        if os.path.isfile(annotation):
            print(f'I found usable annotations for {fol}! Adding it as a dataset...')
            if fol in ['AA1']:
                register_coco_instances(f'{fol}_train', {}, subfol + os.path.sep + "instances_default.json", subfol)
            elif fol in ['AA4']: #some issue with this annotation. 'AA4
                register_coco_instances(f'{fol}_test', {}, subfol + os.path.sep + "instances_default.json", subfol)
            elif fol in ['AA8','AA7']:
                register_coco_instances(f'{fol}_val', {}, subfol + os.path.sep + "instances_default.json", subfol)


if __name__ == "__main__":
    register_mlt_coco_datasets(number_plate_annotation_dir)
    dataset_dicts = DatasetCatalog.get("AA1_train")
    for d in random.sample(dataset_dicts, 1):
        print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("AA1_train"), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('test', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# %%
