'''
Description: Setting up the trainer and evaluator for the model.
# Created By: Amey Vasulkar
# Created On: 2023-10-17
# Copyright (C) 2023 Rise3D BV,
# This file is part of the Orthos Project.
# It cannot be copied and/or distributed without the express permission of Rise3D BV (www.rise3d.nl).
'''
import os
import random
from datetime import datetime
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper, MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.data.transforms as T

# custom_transform_list = [
#             T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
#         ]
# ## AUGMENTATION  from TIM##
# custom_transform_list = [
#             T.RandomBrightness(0.8, 1.2),
#             T.RandomContrast(0.8, 1.2),
#             T.RandomSaturation(0.8, 1.2),
#             T.RandomRotation(angle=[90, 180, 270]),
#             T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#             T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
#         ]
custom_transform_list = []

    
class MyNumberPlateTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        
        with open(cfg.OUTPUT_DIR + os.path.sep + "AugConfig.txt", "w") as output:
            output.write(str(custom_transform_list))
        
        # return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, recompute_boxes = True,
        #                                                         	    augmentations = custom_transform_list))
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                	    augmentations = custom_transform_list))

    @classmethod
    def build_evaluator(cfg, dataset_name, output_folder=None):
        # Disable automatic evaluation by returning None
        # if output_folder is None:
        #     output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # return COCOEvaluator(dataset_name, ("bbox", "segm"), output_dir=output_folder)
        return None

    # @classmethod
    # def do_test(cls, cfg):
    #     data_loader = build_detection_test_loader(cfg, dataset_name=cfg.DATASETS.TEST[0])
    #     evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR)
    #     results = inference_on_dataset(cfg.model, data_loader, evaluator)
    #     return results
    
    @classmethod
    def after_train(cls):
        # results = do_test(cfg)
        # Process and log results as needed
        # return results
        return "Done! For now, no evaluation yet. Thanks for training :)."
    
def dumpConfig(directory, cfg):
    
    configinfo = cfg.dump()
    filename = "config_dump.txt"
    
    print("Exporting the config")
    with open(directory + os.path.sep + filename, "w") as file:
        file.write(configinfo)

def generate_run_name():
    models = ['Arctic', 'Beam', 'Chrome', 'Digi', 'Encrypt', 'Flash', 'Ghost', 'Holo', 'Jitter', 'Kappa', 'Lambda']
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    word = random.choice(models)
    num = random.choice(numbers)
    return word + "-" + str(num)

def trainset_and_time_info(cfg, sets):
    """
    Write the current date and time to a .txt file at the given path,
    followed by the elements of the dataset_list.

    Args:
        path (str): The path where the .txt file will be created.
        dataset_list (list): A list of items to be written in the file.
    """
    direc = cfg.OUTPUT_DIR
    current_datetime = datetime.now().strftime('%d-%m-%Y %H:%M')
    filename = "datetime_file.txt"

    file_path = os.path.join(direc, filename)

    with open(file_path, "w") as file:
        file.write(current_datetime + "\n")

    with open(file_path, "a") as file:
        file.write("__________________\n")

    with open(file_path, "a") as file:
        file.write("Datasets:\n")
        for item in sets:
            file.write(item + "\n")