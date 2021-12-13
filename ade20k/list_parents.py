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
parser.set_defaults(save=True,display=False)
parser.add_argument('classnames', type=str, nargs='+', help='names of the classes to list parents of, separated by colons (comma and whitespace can be part of an ADE20k classname).')
parser.add_argument('--out-dir', type=path_arg, default="", help='the folder to store the csv files in. (defaults to this folder)')
parser.add_argument('--no-save', dest="save", action="store_false", help='dont store the stats as csv file.')
parser.add_argument('--display', dest="display", action="store_true", help='print the stats to console.')
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
                results[class_id]['no_parent'] += 1
            else:
                parent = utils.ImgData.find_obj_by_id(imgdata,obj['parts']['ispartof'])
                results[class_id]['with_parent'] += 1
                if not parent['name'] in results[class_id]['parents']:
                    results[class_id]['parents'][parent['name']] = 1
                else:
                    results[class_id]['parents'][parent['name']] += 1
             

    
B  = '\033[34m' # blue
W  = '\033[0m'  # white (normal)
Gy  = '\033[90m' # Gray
underl = '\033[4;1m'
normal = '\033[0m'

if args.save and not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

for classname, class_id in zip(args.classnames,class_ids):       
    d = results[class_id]
    print("")
    print(underl + classname + normal + Gy + f" {d['imgs']:5} images,{d['with_parent']:5} with parent, {d['no_parent']:5} without" + normal)
    d['parents'] = dict(sorted(d['parents'].items(),key=lambda it:it[1],reverse=True))
    if args.display:
        print("parent class; count")
        print(f"{B}NONE{W}; {d['no_parent']}")
        for name,count in d['parents'].items():
            print(f"{name}; {count};")
    if args.save:
        out_file = f"class_stats_{classname}.csv"
        with open(os.path.join(args.out_dir,out_file),"w") as file:
            file.write("parent class; count\n")
            file.write(f"NONE; {d['no_parent']}\n")
            for name,count in d['parents'].items():
                file.write(f"{name}; {count};\n")
            
if len(utils.imgdata_encs) > 1:
    print("Multiple encodings found:")
    print(utils.imgdata_encs)
