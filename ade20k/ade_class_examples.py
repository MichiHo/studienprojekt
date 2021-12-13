"""Find examples for a given list of classes and save the training images with outlines around the corresponding classes.
"""
import argparse
import os
import shutil

import cv2
from tqdm import tqdm

import ade_utils as utils
from utils import *

parser = argparse.ArgumentParser(description=__doc__, 
    usage="""usage: ade_class_examples.py [-h] [--out-dir OUT_DIR] [--num NUM]
                             [--no-outline]
                             classname : classname : ...""")
parser.set_defaults(outline=True)
parser.add_argument('classnames', type=str, nargs='+', help='names of the classes to show examples of, separated by colons (comma and whitespace can be part of an ADE20k classname).')
parser.add_argument('--out-dir', type=path_arg, default="examples", help='the folder in which a subfolder for each class will be filled with examples if --num > 1. If --num=1, all images are stored directly here. (default: "examples")')
parser.add_argument('--num', type=int, default=10, help='the number of examples to extract (default: 10)')
parser.add_argument('--no-outline', dest="outline", action="store_false", help='dont paint outlines around the classes on the image.')
args = parser.parse_args()

args.classnames = colon_separated(args.classnames)

ade_index = utils.AdeIndex.load()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
    
total_img_count = 0
for i,cname in enumerate(args.classnames):
    print(f"[{i:4d}/{len(args.classnames)}] ",end="")
    try:
        class_id = utils.AdeIndex.class_index(ade_index,cname)
    except ValueError:
        print("No class of name",cname,"skipping.")
        continue
    print(f"{cname} (#{class_id})")
    iterr = utils.AdeIndex.images_with_class(ade_index,class_id,count=args.num)
    if args.num > 1: 
        iterr = tqdm(iterr)
        class_folder_path = os.path.join(args.out_dir,cname)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)
    else:
        class_folder_path = args.out_dir
    
    for img_id in iterr: 
        img, img_data = utils.AdeIndex.load_img(ade_index,img_id,True)
        out_name = img_data['filename']
        if args.num == 1: out_name = f"{cname}_" + out_name
        total_img_count += 1
        if args.outline:
            img = utils.Images.class_outlines(img,img_data,[[class_id,None,(255,0,0),4]])
            cv2.imwrite(os.path.join(class_folder_path,out_name),img)
        else:
            img = utils.AdeIndex.load_img(ade_index,img_id,False)
            cv2.imwrite(os.path.join(class_folder_path,out_name),img)
print(f"Stored {total_img_count} images in {args.out_dir}")