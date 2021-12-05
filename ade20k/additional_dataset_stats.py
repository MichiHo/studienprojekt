"""Create classes_new.pkl from all img annotation jsons, containing two fields:

-   classes: contains for each class:
    -   name
    -   scenes: scenes it is present in, 
    -   parents: dict of IDs of classes which are parent to this class and the count how often 
    -   object_count: total number of class instances in the dataset
-   scenes: dict of scene-names mapped to image count. only the first element of each images 'scene'
    attribute is used, which mostly contains either 'indoor' or 'outdoor'
"""
import json
import os
import pickle
import time
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import ade_utils as utils
from utils import path_arg, conf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-path', type=path_arg, default=conf.ade_classes_new_path,
    help='path of the output pkl file. (default from configuration)')
args = parser.parse_args()

ade_index = utils.adeindex.load()

start_time = time.time()

classes = dict()
missing_parents = 0
scenes = dict()

for img_index in tqdm(range(utils.num_images)):
    print(f"{img_index:5} {utils.progress_bar(img_index,utils.num_images,length=30)}\r",end="")
    imgdata = utils.imgdata.load(ade_index['folder'][img_index],ade_index['filename'][img_index][:-4])
    scene = imgdata['scene'][0]
    if not scene in scenes: scenes[scene] = 0
    scenes[scene] += 1
        
    for obj in imgdata['object']:
        class_id = obj['name_ndx']
        if not class_id in classes.keys():
            classes[class_id] = {
                'name': ade_index['objectnames'][class_id],
                'scenes': {},
                'parents': {
                -1 : 0
                },
                'object_count': 0
            }
        classes[class_id]['object_count'] += 1
        if not scene in classes[class_id]['scenes'].keys(): classes[class_id]['scenes'][scene] = 0
        classes[class_id]['scenes'][scene] += 1
            
        if type(obj['parts']['ispartof']) == int:
            parent = utils.imgdata.find_obj_by_id(imgdata,obj['parts']['ispartof'])
            if not parent:
                missing_parents += 1
                continue
            parent_class_id = parent['name_ndx']
            if not parent_class_id in classes[class_id]['parents'].keys():
                classes[class_id]['parents'][parent_class_id] = 1
            else:
                classes[class_id]['parents'][parent_class_id] += 1
        else:
            classes[class_id]['parents'][-1] += 1

print("--- %s seconds ---                                                          " % (time.time() - start_time))
print("Missing parents:",missing_parents)
with open(args.out_path, 'wb') as json_file:
    pickle.dump({
        'classes': classes,
        'scenes': scenes,
        'missing_parents': missing_parents
    }, json_file)
    
#print(classes)
