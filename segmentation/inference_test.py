"""Choose a number of trained algorithms (their corresponding directories) to run a set of images through
each of them. Algorithms in the command line are interpreted as relative to --work-dir, unless --paths True
is set.

Images are loaded from the --images-dir, split into indoor/outdoor if both such named folders
exist in that directory. In that case, algorithms trained only indoor/outdoor will only process the 
corresponding images. 

Images are then saved to --output-dir, into one folder per model. Each image is stored as .jpg (original
with segmentation overlayed) and as .png (original segmentation mask, as P-Mode png)."""

import argparse
import os
import pickle
import sys

import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from PIL import Image
from tqdm import tqdm

from utils import *


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folders', type=path_arg, nargs='*', default=[], help='the algorithm folders to use. Interpreted as paths relative to --work-dir, or relative to this script if --paths True is set. if omitted, an interactive choice is shown.')
parser.add_argument('--paths', default=False, action=argparse.BooleanOptionalAction, help='whether to interpret the positional arguments as paths. If set to False (default), the arguments are interpreted as paths relative to --work-dir.')
parser.add_argument('--max-imgs', type=int, default=999999, help='the maximum number of images to run inference on.')
parser.add_argument('--work-dir', type=path_arg, default=conf.segmentation_model_path, help='the folder containing all trained algorithms folders. (default from configuration)')
parser.add_argument('--output-dir', type=path_arg, default=conf.segmentation_out_path, help='the folder to create one folder per algorithm in, which will contain the results. (default from configuration)')
parser.add_argument('--images-dir', type=path_arg, default=True, action=argparse.BooleanOptionalAction, help='whether or not to maximize each new grid window.')
parser.add_argument('--overlay-opacity', type=float, default=0.5, help='the opacity with which the segmentation map is painted over the original image for the jpg-output.')
parser.add_argument('--ignore-model-classcount', default=False, action=argparse.BooleanOptionalAction, help='whether to ignore if a model has more classes than the general configuration. If set to False, a dialog will show in this case, if set to True (or if the dialog is answered with Yes), the palette will be padded with blacks.')
args = parser.parse_args()


if not os.path.exists(args.images_dir):
    print("Images dir does not exist: ",args.images_dir)
    exit()
if not os.path.exists(args.work_dir):
    print("Work dir does not exist: ",args.work_dir)
    exit()

if len(args.folders) == 0:
    folders = [os.path.join(args.work_dir,d) for d in os.listdir(args.work_dir)]
    folders = [d for d in folders if os.path.isdir(d)]
    folders = [d for d in folders if "latest.pth" in os.listdir(d)]
    print(f"Found {len(folders)} configurations.")
    if (len(folders) == 0): exit()
    picked_folders =  multichoice(folders,displaylist=[os.path.split(p)[1] for p in folders])
    if (len(picked_folders) == 0): exit()
else:
    if args.paths:
        picked_folders = [os.path.join(args.work_dir,f) for f in args.folders]
    else:
        picked_folders = args.folders

    for f in picked_folders:
        if not os.path.exists(f):
            print("Algorithm dir does not exist:",f)
            exit()

print("Checking configurations...")
models = []
for f in picked_folders:
    conf_name = os.path.split(f)[1]
    print(conf_name)
    files = os.listdir(f)
    conf_file = [ff for ff in files if ff.endswith(".py")]
    if len(conf_file) > 1:
        print("Multiple possible conf files found")
        conf_file = choice(conf_file)
        if conf_file is None: continue
    elif len(conf_file) == 0:
        print("No conf file found for",f)
        continue
    else: conf_file = conf_file[0]
    conf_file = os.path.join(f,conf_file)
    print("\t- Conf:      ",conf_file)
    
    if "indoor" in conf_file:
        scenes = ["indoor"]
    elif "outdoor" in conf_file:
        scenes = ["outdoor"]
    elif "inout" in conf_file:
        scenes = ["indoor","outdoor"]
    else:
        print("No scenes detected")
        scenes = multichoice(["indoor","outdoor"])
    print("\t- Scenes:    ",scenes)
    checkpoint_file = os.path.join(f,"latest.pth")
    if os.path.exists(checkpoint_file):
        print("\t- Checkpoint:",checkpoint_file)
    models.append(dict(
        name = conf_name,
        input_folders = scenes,
        config_file = conf_file,
        checkpoint_file = checkpoint_file
    ))
if len(args.folders) == 0:
    if not input("Is this okay? [y/n]") in ["y","Y"]:
        exit()

print("Loading images ...")
in_folders = ["indoor","outdoor"]
in_out_split = True
for folder in in_folders:
    in_folder = os.path.join(args.images_dir,folder)
    if not os.path.exists(in_folder):
        print("Indoor/Outdoor folder missing, use all images from ",args.images_dir)
        in_out_split=False
    
if in_out_split:
    all_images = {}
    all_image_names = {}
    for folder in in_folders:
        folder_path = os.path.join(args.images_dir,folder)
        images = []
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            images.append(mmcv.imread(os.path.join(folder_path,img_name)))
        all_images[folder] = images
        all_image_names[folder] = img_names
    print(f" done. {len(all_images['indoor'])} indoor, {len(all_images['outdoor'])} outdoor. Type: {type(all_images['indoor'][0])}")
else:
    all_images = []
    all_image_names = os.listdir(custom_folder)
    for img_name in all_image_names:
        all_images.append(mmcv.imread(os.path.join(custom_folder,img_name)))
    print(f" done. {len(all_images)} images. Type: {type(all_images[0])}")

    


for model_desc in models:
    print("# Model",model_desc['name'],"#")
    print("- load model... ")
    # build the model from a config file and a checkpoint file
    model = init_segmentor(model_desc['config_file'], model_desc['checkpoint_file'], device='cuda:0')
    print(f"- ...done loading model.")
    
    if len(model.CLASSES) != len(conf.palette) and not args.ignore_model_classcount:
        if not input(f"{len(model.CLASSES)} classes in model, {len(conf.palette)} palette items. Okay? [y/n] ") in ["y","Y"]:
            continue
    palette = conf.padded_palette(len(model.CLASSES))
    
    out_folder = os.path.join(args.output_dir,"output_"+model_desc['name'])
    
    if not os.path.exists(args.out): os.makedirs(out_folder)
    
    if in_out_split:
        images = []
        #image_paths = []
        out_img_names = []
        for folder in model_desc['input_folders']:
            images = images + all_images[folder]
            #image_paths.extend([os.path.join(in_folder_root,folder,n) for n in all_image_names[folder]])
            out_img_names = out_img_names + [folder+"_"+n for n in all_image_names[folder]]
    else:
        images = all_images
        out_img_names = all_image_names
        
    num_imgs = min(args.max_imgs,len(out_img_names))
    print(f"- inference on {num_imgs} images... ")
    # test a single image and show the results
    results = [] 
    min_index = 9999
    max_index = -1
    for img in tqdm(images[:num_imgs]):
        result = inference_segmentor(model, img)
        #result = [res + 1 for res in result] # this would account for the shift. but maybe we don't need it.
        results.append(result)
        min_index = min(min_index,np.min(result[0]))
        max_index = max(max_index,np.max(result[0]))
    print("- ...done inference.")
    print(f"Min index {min_index} max index {max_index}")
    
    print("- store results... ",end="")
    for img_name,result,img in zip(out_img_names,results,images):
        # with open(os.path.join(out_folder,img_name[:-4]+".pkl"),"wb") as ff:
        #     pickle.dump(result, ff)
        ann_img = Image.fromarray(np.array(result[0],dtype=np.uint8),'P')
        ann_img.putpalette(palette.flatten())
        ann_img.save(os.path.join(out_folder,img_name[:-4]+".png"))
        model.show_result(img, result, palette=palette, out_file=os.path.join(out_folder,img_name[:-4]+".jpg"), opacity=args.overlay_opacity)
    print("done.")
