"""Visualize images by adding a sidebar with all classes and drawing lines to all detected objects of that class. It takes either all images in a given folder or a single image.
"""

import argparse
import os
import sys
from numbers import Number

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

from utils import *


parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(save=True,resize=True,histogram=False)
parser.add_argument('--root-dir', type=path_arg, default=conf.segmentation_out_path,
    help='the folder to look for image folders in if --dir is not set. (default from configuration)')
parser.add_argument('--dir', type=path_arg, default=None,
    help='the folder to directly visualize images from. (defaults to None)')
parser.add_argument('--no-save', dest="save", action="store_false", 
    help='dont save the visualizations as png files.')
parser.add_argument('--save-dir', type=path_arg, default=None,
    help='the folder to save visualizations into instead of the input folder. (defaults to None)')
parser.add_argument('--mode',choices=['all','interactive'],default='interactive',
    help="whether to visualize 'all' images or ask for one  ")
parser.add_argument('--conf-path', type=path_arg, default=None, help='path of conf.json, if it deviates from the default.')
parser.add_argument('--display',choices=['window','maximized','none'],default='maximized',
    help="whether to display each image and how. (default: 'maximized')")
parser.add_argument('--no-resize', dest="resize", action="store_false", 
    help='dont resize big images to have a reasonable font size in comparison.')
parser.add_argument('--histogram', dest="histogram", action="store_true", 
    help='whether to also show a histogram (pixels of classes) in the live display.')
args = parser.parse_args()

if args.conf_path is not None:
    conf = GeneralConfig(args.conf_path)

def findclass(pixel):
    if hasattr(pixel, '__iter__'):
        for cl in conf.classes:
            if (cl.cv2color == pixel).all():
                return cl.id
        return None
    if isinstance(pixel,Number):
        if pixel < 0 or pixel >= len(conf.classes):
            print(f"Invalid index {pixel}")
            return None
        else: return conf.classes[pixel].id
    return None

if args.dir is None:
    while not os.path.exists(args.root_dir):
        print(f"Root dir {args.root_dir} does not exist. Enter new:")
        args.root_dir = input("")
    picked_folder = os.path.join(args.root_dir,choice(os.listdir(args.root_dir)))
else:
    while not os.path.exists(args.dir):
        print(f"Folder {args.dir} does not exist. Enter new:")
        args.dir = input("")
    picked_folder = args.dir

if args.save_dir is None:
    save_dir = picked_folder
else:
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

file_endings = {".png"}
def file_fit(f):
    name,ext = os.path.splitext(f)
    return ext.lower() in file_endings and not "_vis" in name and os.path.exists(os.path.join(picked_folder,name+".jpg"))
files = [f for f in os.listdir(picked_folder) if file_fit(f)]
all_i = -1
imgsize = [700,700]
picked_files = set()
while True:
    if args.mode == 'all':
        if all_i >= len(files): break
        all_i += 1
        ann_name = files[all_i]
        print(ann_name)
    else:
        ann_name = choice(files, text="File: ",grayout=picked_files)
        if ann_name is None: break
        picked_files.add(ann_name)
    ann_path = os.path.join(picked_folder,ann_name)
    img_path = os.path.join(picked_folder,ann_name[:-4]+".jpg")
    ############ LOAD IMAGES
    ann_pil = Image.open(ann_path)

    if ann_pil.mode != "P":
        print(f"- Annotation image {ann_path} is not P-Mode, but {ann_pil.mode} mode")
        
    img_rgb = cv2.imread(img_path)
    
    if args.resize:
        scaling = min(imgsize[0]/ann_pil.size[0],imgsize[1]/ann_pil.size[1])
        scaled_size = (int(ann_pil.size[0] * scaling),int(ann_pil.size[1]*scaling))
        img_rgb = cv2.resize(img_rgb,dsize=scaled_size)
        ann_pil = ann_pil.resize(scaled_size)
    
    ann = np.array(ann_pil)


    ############ PAINTING
    sidebar_width = 220
    vis_img = np.concatenate([img_rgb,np.ones([img_rgb.shape[0],sidebar_width,3])*255],axis=1)

    region_img, regioncount = label(ann,return_num=True)
    props = regionprops(region_img)
    print("- Found",regioncount,"objects")

    objects_per_class = {cl.id : [] for cl in conf.classes}
    for i,region in enumerate(props):
        x = region.centroid[0]
        y = region.centroid[1]
        cl = findclass(ann[int(x),int(y)])
        if cl is not None:
            objects_per_class[cl].append(region)
    print("- Found",len(objects_per_class),"different target classes.")

    margin = 10
    x = ann.shape[1] + margin
    fontsize = 10
    i = 0
    for class_i, region_list in objects_per_class.items():
        if len(region_list) == 0: continue
        i += 1
        cl = conf.classes[class_i]
        y = (margin+fontsize)*i
        cv2.putText(vis_img, f"[{class_i:2}] {cl.name} x{len(region_list)}", [x,y], cv2.FONT_HERSHEY_SIMPLEX,0.5,cl.cv2color)
        for j, region in enumerate(region_list):
            center = (int(region.centroid[1]),int(region.centroid[0]))
            cv2.circle(vis_img, center, 3, cl.cv2color,thickness=-1)
            cv2.line(vis_img,center,[x-3,y-5],cl.cv2color,1)
    if args.save:
        filename_no_ext = os.path.splitext(ann_name)[0]
        cv2.imwrite(os.path.join(save_dir,filename_no_ext+"_vis.png"), vis_img)
        
    if not args.display == 'none':
        if ann_pil.mode == 'P' and args.histogram:
            fig,[ax,hist_ax] = plt.subplots(1,2,gridspec_kw={'width_ratios': [3, 1]})
            bins = np.arange(len(conf.classes)+1)
            hist_ax.hist(ann.flatten(),bins=bins)
            hist_ax.set_xticks(bins[:-1]+0.5)
            hist_ax.set_xticklabels([cl.name for cl in conf.classes],rotation="vertical")
        else:
            fig,ax = plt.subplots()
        ax.imshow(vis_img.astype(int)[:,:,[2,1,0]])
        plt.tight_layout()
        if args.display == 'maximized':
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show()
