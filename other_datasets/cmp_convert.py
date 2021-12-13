"""
Transform annotations from CMP dataset to Studienprojekt Format.

CMP dataset has the following indices (zero is omitted):

1 background
2 facade
3 window
4 door
5 cornice
6 sill
7 balcony
8 blind
9 deco
10 molding
11 pillar
12 shop
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

cmp_to_studproj = [
    'background',
    'background',
    'building',
    'window',
    'door',
    'building',
    'building',
    'balcony',
    'building',
    'building',
    'building',
    'column',
    'building'
]
cmp_to_studproj = np.array([
    conf.by_name[v].id for v in cmp_to_studproj
])

cmp_folder = os.path.join(conf.cmp_dir,"all")
ann_folders = [os.path.join(conf.dataset_out_path,scene,"annotations","train") for scene in locations]
img_folders = [os.path.join(conf.dataset_out_path,scene,"images","train") for scene in locations]
filenames = [name[:-4] for name in os.listdir(cmp_folder) if name.endswith(".jpg")]
print(f"Converting {len(filenames)} files. Every {conf.extension_datasets_val_every}-th file is used as val instead of train. Using palette {conf.train_palette}")

skipped = 0
modes = {"train":0,"val":0}
iterr = tqdm(enumerate(filenames))
for i,filename in iterr:
    img_path = os.path.join(cmp_folder,filename + ".jpg")
    ann_path = os.path.join(cmp_folder,filename + ".png")
    if not os.path.exists(ann_path):
        print(f"Missing annotation for {filename}. Skipping.")
        skipped += 1
        continue
    ann = Image.open(ann_path)
    if ann.mode != 'P':
        print(f"Annotation for {filename} is not P-Mode. Skipping.")
        skipped += 1
        continue
    size = ann.size
    # +1 to skip the zero index
    newdata = cmp_to_studproj[np.array(ann)].reshape([size[1],size[0]]).astype(np.uint8) + 1
    ann_new = Image.fromarray(newdata)
    ann_new.putpalette(conf.train_palette)
    
    mode = "val" if i % conf.extension_datasets_val_every == 0 else "train"
    modes[mode] += 1
    for scene in ["outdoor","inout"]:
        ann_new.save(os.path.join(conf.dataset_out_path,scene+"_extended","annotations",mode,filename + ".png"))
        shutil.copy(img_path, os.path.join(conf.dataset_out_path,scene+"_extended","images",mode,filename + ".jpg"))
    iterr.set_description(f"{modes['train']:4} train, {modes['val']:4} val, {skipped:4} skipped")
