"""Takes classes of the ADE20k dataset (by name) and lists all classes objects of this class can be part of (parents). It can be written to csv file and printed to console."""
import argparse
import os
import pickle
import shutil
import sys
from tqdm import tqdm

import numpy as np

from utils import *
import ade_utils as utils


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('classnames', type=str, nargs='+', help='names of the classes to list parents of, separated by colons (comma and whitespace can be part of an ADE20k classname).')
parser.add_argument('--out-dir', type=path_arg, default="", help='the folder to store the csv files in. (defaults to this folder)')
parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction, help='whether to store the stats as csv file.')
parser.add_argument('--display', default=False, action=argparse.BooleanOptionalAction, help='whether to print the stats to console.')
args = parser.parse_args()

args.classnames = colon_separated(args.classnames)

data = utils.AdeIndex.load()

class_ids = []
for classname in args.classnames:
    try:
        class_id = utils.AdeIndex.class_index(data,classname)
        class_ids.append(class_id)
    except ValueError:
        print(f"No class named {classname}!")
        args.classnames.remove(classname)
        
print("Computing stats for:")
results = dict()
for classname, class_id in zip(args.classnames,class_ids):
    print(f" - '{classname}' (#{class_id})")
    # Stats for one class
    #objects = utils.objects.load()
    # How often is obj of class part of each other class?
    results[class_id] = dict(
        no_parent = 0,
        with_parent = 0,
        parents = {},
        imgs = 0
    )

for i in tqdm(range(utils.num_images),desc="Analyzing images"):
    if np.sum(data['objectPresence'][class_ids,i]) > 0:
        # Found image containing at least one class
        for class_id in class_ids:
            if data['objectPresence'][class_id,i] > 0:
                results[class_id]['imgs'] += 1
                
        imgdata = utils.ImgData.load(data['folder'][i],os.path.splitext(data['filename'][i])[0])
        for obj in imgdata['object']:
            class_id = obj['name_ndx']
            if not class_id in class_ids: continue
            if obj['parts']['ispartof'] == []:
                result[class_id]['no_parent'] += 1
            else:
                parent = utils.ImgData.find_obj_by_id(imgdata,obj['parts']['ispartof'])
                result[class_id]['with_parent'] += 1
                if not parent['name'] in parents:
                    result[class_id]['parents'][parent['name']] = 1
                else:
                    result[class_id]['parents'][parent['name']] += 1
                    
print("")

    
for classname, class_id in zip(args.classnames,class_ids):
    print(f"img {imgs:5} no_parent {no_parent:5} with_parent {with_parent:5}{'.'*80}\r",end="")
    d = results[classname]
    out_file = f"class_stats_{classname}.csv"
    with open(out_file,"w") as file:
        file.write("parent class; count\n")
        file.write(f"NONE; {d['no_parent']}\n")
        for name,count in d['parents'].items():
            file.write(f"{name}; {count};\n")
            
    
print("Encodings found:")
print(utils.imgdata_encs)
