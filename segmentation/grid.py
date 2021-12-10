"""For the given folders, grid.py displays every image they all contain side by side for comparison.
The --rows argument specifies how many images to display at the same time."""
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from utils import *

parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(paths=False,maximize=True)
parser.add_argument('folders', type=path_arg, nargs='*', help='the folders to use. If omitted, an interactive dialog is shown to pick from folders inside --root-dir')
parser.add_argument('--root-dir', type=path_arg, default=conf.segmentation_out_path, help='the folder in which to look for the output folders. (default from configuration)')
parser.add_argument('--paths', dest="paths", action="store_true", help='interpret folders as paths instead of looking in --root-dir.')
parser.add_argument('--rows', type=int, default=4, help='the number of rows / different images to display at a time.')
parser.add_argument('--no-maximize', dest="maximize", action="store_false", help='dont maximize each new grid window.')
args = parser.parse_args()


if len(args.folders) == 0:
    folders = [os.path.join(args.root_dir,d) for d in os.listdir(args.root_dir)]
    folders = [d for d in folders if os.path.isdir(d)]
    print(f"Found {len(folders)} output folders.")
    if (len(folders) == 0): exit()
    picked_folders =  multichoice(folders,displaylist=[os.path.split(f)[1] for f in folders])
    if (len(picked_folders) == 0): exit()
else:
    if args.paths:
        picked_folders = [os.path.join(args.root_dir,f) for f in args.folders]
    else:
        picked_folders = args.folders

    for f in picked_folders:
        if not os.path.exists(f):
            print("Chosen dir does not exist:",f)
            exit()
foldernames = [os.path.split(f)[1] for f in picked_folders]
row = 0
imgsize=[900,600]
window_title = "  |  ".join(foldernames)
fig,ax = plt.subplots(args.rows,len(picked_folders),figsize=[10,10],num=window_title)
for imgname in [filename for filename in os.listdir(picked_folders[0]) if filename.lower().endswith(".jpg")]:
    missing = False
    for folder in picked_folders[1:]:
        if not os.path.exists(os.path.join(folder,imgname)):
            missing = True
            break
    if missing: 
        print("Skipping",imgname)
        continue
    if row==args.rows:
        plt.axis('off')
        plt.tight_layout()
        if args.maximize:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show()
        
        fig,ax = plt.subplots(args.rows,len(picked_folders),figsize=[10,10],num=window_title)
        row = 0
        
    for i,folder in enumerate(picked_folders):
        path = os.path.join(folder,imgname)
        img = Image.open(path)
        scaling = min(imgsize[0]/img.size[0],imgsize[1]/img.size[1])
        img = img.resize((int(img.size[0] * scaling),int(img.size[1]*scaling)))
        #img.thumbnail(imgsize,Image.ANTIALIAS)
        ax[row,i].imshow(img)
        if row==0:
            ax[row,i].set_title(foldernames[i])
        ax[row,i].set_yticks(ticks=[])
        ax[row,i].set_xticks(ticks=[])
    row += 1

plt.axis('off')
plt.tight_layout()
if args.maximize:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
plt.show()