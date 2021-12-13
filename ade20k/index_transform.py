"""Apply a Lookup-Table (LUT) to the indices of all images in the studienprojekt dataset. The used
palette will be the class colors from the configuration, with black appended at the beginning (only 
affects display of the training annotations, not the training itself.).

Example:
To join classes 0 and 1   : LUT=[0,0,1,2,3,...]
To leave the index 0 free : LUT=[1,2,3,...]"""
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
import shutil

from utils import * 


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--studienprojekt-dir', type=path_arg, default=conf.dataset_out_path, help='the folder containing the dataset (default from configuration).')
parser.add_argument('--out-dir', type=path_arg, default=conf.dataset_out_path, help='the folder to store the transformed dataset (defaults to overwriting the input folder!)')
parser.add_argument('--subsets', type=str, nargs='+', default=["indoor","outdoor","inout","outdoor_extended","inout_extended"], help='the subfolders of --studienprojekt-dir to use. (default: ["indoor","outdoor","inout","outdoor_extended","inout_extended"])')
parser.add_argument('--lut', type=int, nargs='+', default=None, help='the look-up table to use. If none is given, a selection is presented.')
args = parser.parse_args()

for subset in args.subsets:
    dirr = os.path.join(args.out_dir,subset)
    if os.path.exists(dirr) and len(os.listdir(dirr)) > 0:
        if input(f"Output folder {dirr} is not empty. Clear contents? [y/n]").lower() == 'y':
            shutil.rmtree(dirr)
            os.makedirs(dirr)
        else:
            print("Skipping this folder")
            args.subsets.remove(subset)
if len(args.subsets) == 0:
    print("No subfolders left")
    exit()
            

if args.lut is None:
    luts = [
        ("Leave out index 0 / right shift", np.arange(1,23)),
        ("Join index 0 and 1", np.concatenate([[0],np.arange(1,23)]))
    ]
    ch = choice(luts,displaylist=[f"{l[0]} : {l[1]}" for l in luts])
    if ch is None:
        exit()
    lutname, args.lut = ch
lut = np.array(args.lut)
# Train images palette starts at index one!
newpalette = np.concatenate([np.array([0,0,0],dtype=np.uint8),conf.palette.flatten()]).astype(np.uint8)

print(f"This will copy all annotations inside {args.studienprojekt_dir} to {args.out_dir} after applying the LUT")
print(f"LUT: {lut}.")
print(f"Picked subfolders: {' '.join(args.subsets)}")
print("Palette:",newpalette)
if not input(f"Okay? [y/n] ") in ["y","Y"]: exit()

for l1 in args.subsets:
    l2 = "annotations"
    for l3 in ["train","val"]:
        in_folder = os.path.join(args.studienprojekt_dir,l1,l2,l3)
        out_folder = os.path.join(args.out_dir,l1,l2,l3)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(in_folder,"->",out_folder,"(newly created)")
        else: print(in_folder,"->",out_folder)
        
        for name in tqdm(os.listdir(in_folder)):
            img = Image.open(os.path.join(in_folder,name))
            img = lut[img].astype(np.uint8)
            img = Image.fromarray(img,mode='P')
            img.putpalette(newpalette)
            img.save(os.path.join(out_folder,name))
            