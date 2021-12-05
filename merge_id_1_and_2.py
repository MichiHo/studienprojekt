from PIL import Image
import numpy as np
from tqdm import tqdm
import os

import conf

# Train images palette starts at index one!
newpalette = np.concatenate([np.array([0,0,0],dtype=np.uint8),conf.palette.flatten()]).astype(np.uint8)
lut = np.concatenate([[0,1],np.arange(1,23)])

root_in = "data/studienprojekt"
root_out = "data/studienprojekt"
print(f"This will copy all annotations inside {root_in} to {root_out} after applying with LUT {lut}.")
if not input(f"Okay? [y/n] ") in ["y","Y"]: exit()

for l1 in ["indoor","outdoor","outdoor_extended"]:
    l2 = "annotations"
    for l3 in ["train","val"]:
        in_folder = os.path.join(root_in,l1,l2,l3)
        out_folder = os.path.join(root_out,l1,l2,l3)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(in_folder,"->",out_folder,"(newly created)")
        print(in_folder,"->",out_folder)
        
        for name in tqdm(os.listdir(in_folder)):
            img = Image.open(os.path.join(in_folder,name))
            img = lut[img].astype(np.uint8)
            img = Image.fromarray(img,mode='P')
            img.putpalette(newpalette)
            img.save(os.path.join(out_folder,name))
            