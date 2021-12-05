"""Change all annotations to P-Type png and copy them, along with input images, into the filestructure
for MMSegmentation.
"""
import os
import pickle
import shutil

from PIL import Image

import ade_utils as utils

stats_filepath = "annotation_new/stats.pkl"
in_folder = "annotation_new/RGB-mode"
out_folder = "annotation_new/inout"
if not os.path.exists(in_folder):
    print("Input folder doesnt exist",in_folder)
    exit()
    
if not os.path.exists(stats_filepath):
    print("Stats file doesnt exist",stats_filepath)
    exit()
    
if not os.path.exists(out_folder): os.makedirs(out_folder)

ade_index = utils.adeindex.load()
with open(stats_filepath,"rb") as stats_file:
    stats = pickle.load(stats_file)
    
for image in stats['images']:
    ann_folder = os.path.join(in_folder,image['scene'])
    foldername = ade_index['folder'][image['id']]
    filename = ade_index['filename'][image['id']][:-4]
    
    if "train" in filename: subdir = "train"
    elif "val" in filename: subdir = "val"
    else:
        print("not train or val!",filename)
        continue
    
    ann_image = Image.open(os.path.join(ann_folder,filename+"_reseg.png"))
    
    shutil.copy(
        ,
        os.path.join(out_folder,"img_dir",subdir,filename+".png"))
