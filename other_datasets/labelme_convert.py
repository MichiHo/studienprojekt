"""
Transform annotations from labelmefacade dataset to Studienprojekt Format.

Labelmefacade dataset uses the following color codes: 

various = 0:0:0
building = 128:0:0
car = 128:0:128
door = 128:128:0
pavement = 128:128:128
road = 128:64:0
sky = 0:128:128
vegetation = 0:128:0
window = 0:0:128
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import *


locations = ["outdoor_extended","inout_extended"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.set_defaults(overwrite=False)
parser.add_argument('--overwrite', dest="overwrite", action="store_true",  help=f'delete previous contents of the {" and ".join(locations)} folders without asking. If not set, a dialog is shown in case one of the folders is not empty.')
args = parser.parse_args()

prepare_dataset_extension_dirs(overwrite=args.overwrite)

labelme_to_studproj = {
    (0,0,0): 'background',
    (128,0,0): 'building',
    (128,0,128): 'background',
    (128,128,0): 'door',
    (128,128,128): 'background',
    (128,64,0): 'background',
    (0,128,128): 'background',
    (0,128,0): 'background',
    (0,0,128): 'window'
}

def myhash(t):
    return t[0]+1.1*t[1]+1.2*t[2],

labelme_to_studproj = {
    myhash(k) : conf.by_name[v].id for k,v in labelme_to_studproj.items()
}

ann_folder = os.path.join(conf.labelme_dir,"labels")
img_folder = os.path.join(conf.labelme_dir,"images")
filenames = [name[:-4] for name in os.listdir(img_folder)]
print(f"Converting {len(filenames)} files. Every {conf.extension_datasets_val_every}-th file is used as val data.")
skipped = 0
modes = {"train":0,"val":0}
iterr = tqdm(enumerate(filenames),total=len(filenames))
for i,filename in iterr:
    img_path = os.path.join(img_folder,filename + ".jpg")
    ann_path = os.path.join(ann_folder,filename + ".png")
    if not os.path.exists(ann_path):
        print(f"\rMissing annotation for {filename}. Skipping.")
        skipped += 1
        continue
    
    ann_labelme = Image.open(ann_path)
    imgdata = np.array(ann_labelme)[:,:,:-1] #skip alpha value
    newdata = np.apply_along_axis(lambda pix: labelme_to_studproj[myhash(pix)],2,imgdata)
    # +1 to skip the zero index
    newdata = newdata.astype(np.uint8) + 1
    ann_new = Image.fromarray(newdata,mode='P')
    ann_new.putpalette(conf.train_palette)
    
    mode = "val" if i % conf.extension_datasets_val_every == 0 else "train"
    modes[mode] += 1
    for scene in locations:
        ann_new.save(os.path.join(conf.dataset_out_path,scene,"annotations",mode,filename + ".png"))
        shutil.copy(img_path, os.path.join(conf.dataset_out_path,scene,"images",mode,filename + ".jpg"))
    iterr.set_description(f"{modes['train']:4} train, {modes['val']:4} val, {skipped:4} skipped")
