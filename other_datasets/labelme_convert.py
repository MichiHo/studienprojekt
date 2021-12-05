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

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil

from utils import *


labelme_to_studproj = {
    (0,0,0): 'environment',
    (128,0,0): 'building',
    (128,0,128): 'environment',
    (128,128,0): 'door',
    (128,128,128): 'environment',
    (128,64,0): 'environment',
    (0,128,128): 'environment',
    (0,128,0): 'environment',
    (0,0,128): 'window'
}
#cmp_to_studproj = [0] * 13
labelme_to_studproj = {
    k : conf.by_name[v].id for k,v in labelme_to_studproj.items()
}

ann_folder = os.path.join(conf.labelme_dir,"labels")
img_folder = os.path.join(conf.labelme_dir,"images")
filenames = [name[:-4] for name in os.listdir(img_folder)]
print(f"Converting {len(filenames)} files. Every {conf.extension_datasets_val_every}-th file is used as val data.")
skipped = 0
modes = {"train":0,"val":0}
iterr = tqdm(enumerate(filenames))
for i,filename in iterr:
    img_path = os.path.join(img_folder,filename + ".jpg")
    ann_path = os.path.join(ann_folder,filename + ".png")
    if not os.path.exists(ann_path):
        print(f"\rMissing annotation for {filename}. Skipping.")
        skipped += 1
        continue
    ann = Image.open(ann_path)
    size = ann.size
    #ann_new = Image.new('P', size)
    imgdata = np.array(ann)[:,:,:-1] #skip alpha value
    newdata = np.apply_along_axis(lambda pix: labelme_to_studproj[tuple(pix)],2,imgdata).astype(np.uint8)
    #newdata = cmp_to_studproj[np.array(ann)].reshape([size[1],size[0]]).astype(np.uint8)
    ann_new = Image.fromarray(newdata)
    ann_new.putpalette(conf.train_palette)
    
    mode = "val" if i % conf.extension_datasets_val_every == 0 else "train"
    modes[mode] += 1
    for scene in ["outdoor","inout"]:
        ann_new.save(os.path.join(conf.dataset_out_path,scene+"_extended","annotations",mode,filename + ".png"))
        shutil.copy(img_path, os.path.join(conf.dataset_out_path,scene+"_extended","images",mode,filename + ".jpg"))
    iterr.set_description(f"{modes['train']:4} train, {modes['val']:4} val, {skipped:4} skipped")