import os
import pickle
import shutil
import sys

import numpy as np

import ade_utils as utils

data = utils.adeindex.load()

obj_class = None
if len(sys.argv)>1:
    obj_class = sys.argv[1]
    try:
        obj_class_id = utils.adeindex.class_index(data,obj_class)
    except ValueError:
        print(f"No class named {obj_class}!")
        exit()
    print(f"Computing stats for '{obj_class}' (#{obj_class_id})")

if not obj_class:
    # Stats for all classes
    count_matrix = np.array(data['objectPresence'])
    img_count = np.count_nonzero(count_matrix,axis=0)
    inst_count = np.sum(count_matrix,axis=0)
    if img_count.shape[0] != utils.num_classes:
        print("pick axis 1!")
        img_count = np.count_nonzero(count_matrix,axis=1)
        inst_count = np.sum(count_matrix,axis=1)


    out_file = "data_stats.csv"
    with open(out_file,"w") as file:
        file.write("'objectnames'; 'objectcounts'; 'proportionClassIsPart'; img count (from 'objectPresence'); inst count (from 'objectPresence')\n")
        for i,name in enumerate(data['objectnames']):
            file.write(f"{name}; {data['objectcounts'][i]}; {data['proportionClassIsPart'][i]}; {img_count[i]}; {inst_count[i]}\n")
else:
    # Stats for one class
    #objects = utils.objects.load()
    # How often is obj of class part of each other class?
    no_parent = 0
    with_parent = 0
    parents = {
    }
    imgs = 0
    for i in range(utils.num_images):
        if data['objectPresence'][obj_class_id,i] > 0:
            imgs += 1
            # Found image containing obj_class
            imgdata = utils.imgdata.load(data['folder'][i],os.path.splitext(data['filename'][i])[0])
            for obj in utils.imgdata.objects_of_class(imgdata,obj_class_id):
                if obj['parts']['ispartof'] == []:
                    no_parent += 1
                else:
                    parent = utils.imgdata.find_obj_by_id(imgdata,obj['parts']['ispartof'])
                    with_parent += 1
                    if not parent['name'] in parents:
                        parents[parent['name']] = 1
                    else:
                        parents[parent['name']] += 1
            print(f"img {imgs:5} no_parent {no_parent:5} with_parent {with_parent:5}{'.'*80}\r",end="")
    print("")
    
    out_file = f"class_stats_{obj_class}.csv"
    with open(out_file,"w") as file:
        file.write("parent class; count\n")
        file.write(f"NONE; {no_parent}\n")
        for name,count in parents.items():
            file.write(f"{name}; {count};\n")
    
    print("Encodings found:")
    print(utils.imgdata_encs)
