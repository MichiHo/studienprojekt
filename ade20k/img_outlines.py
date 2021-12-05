""" For a folder containing images and imgdata for them, generate new images with the outlines
of certain classes added.
"""
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import ade_utils as utils

if len(sys.argv)<2:
    print("Specify folder name")
    exit()
foldername = sys.argv[1]

out_folder = os.path.join(foldername,"shapes")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

ade_index = utils.adeindex.load()

# classes_colors = [
#     ("windowpane, window",None,(0,255,0),None),
#     ("windowpane, window","building",(0,255,0),4),
#     ("window",None,(255,0,0),None),
#     ("window","building",(255,0,0),4),
#     ("door",None,(0,0,255),None),
#     ("door","building",(0,0,255),4),
#     ("wall",None,(255,255,0),None)
# ]

classes_colors = [
    ("folding door",None,(255,0,0),None),
    ("folding doors",None,(0,255,0),None)
]

if len(sys.argv)>=3:
    classes_colors = [
        (sys.argv[2],None,(255,0,0),None)
    ]

# convert to indices
classes_colors = [
    [utils.adeindex.class_index(ade_index,name),
    utils.adeindex.class_index(ade_index,parent) if parent is not None else None,
    c,t] for name,parent,c,t in classes_colors
]

files = os.listdir(foldername)

#fig,ax = plt.subplots(2,5,figsize=(10,10),sharex=True,sharey=True)
i = 0
for f in files:
    if f.endswith(".json"):
        filename = f[:-5]
        img = cv2.imread(os.path.join(foldername,filename+".jpg"))
        img_data = utils.imgdata.load(foldername,filename)
        img = utils.image.class_outlines(img,img_data,classes_colors)
        if img.shape[0] > img.shape[1]:
            print("shape 1 > 1 for",filename)
        #ax[i % 2][int(i/2)].imshow(img)
        i += 1
        cv2.imwrite(os.path.join(out_folder,filename+"_shapes.png"),img)

#fig.tight_layout()
#plt.show()
