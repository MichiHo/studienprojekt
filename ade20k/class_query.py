"""Interactive script for querying examples or stats for individual classes
"""
import os
import pickle
import shutil

import ade_utils as utils

ade_index = utils.adeindex.load()

num_images = 27574
num_classes = 3688

mode = input("stats / examples")
if not mode:
    mode = "stats"

if mode == "examples":
    default_output_folder = "~/ssh_transfer"
    output_folder = input(f"Enter output folder (default {default_output_folder}): ")
    if not output_folder:
        output_folder = default_output_folder
        print(f"-> Pick default {default_output_folder}")
    output_folder = os.path.expanduser(output_folder)

    imgs_per_class = input("Images per class to search (default 1): ")
    if not imgs_per_class:
        imgs_per_class = 1
    else:
        imgs_per_class = int(imgs_per_class)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# examples
while True:
    c = input("Class (append ~ for text-matching): ")
    if not c:
        break

    if c[0] == "~":
        c = c[1:]
        classes = utils.adeindex.classes_containing(ade_index,c)
        print(f"-> Look for any of {[ade_index['objectnames'][cx] for cx in classes]}")
    else:
        try:
            classes = [utils.adeindex.class_index(ade_index,c)]
        except ValueError:
            print(f"-> No class named {c}!")
            continue
    if imgs_per_class > 1 and mode == "examples":
        class_folder = os.path.join(output_folder,c)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    imgs_found = 0
    for img_index in range(num_images):
        if utils.adeindex.matches_one_class(ade_index,classes,img_index):
            imgs_found += 1
            
            if mode == "examples":
                img_in = os.path.join(
                    ade_index['folder'][img_index],
                    ade_index['filename'][img_index])
                seg_in = os.path.join(
                    ade_index['folder'][img_index],
                    ade_index['filename'][img_index][:-4]+"_seg.png")
                par_in = os.path.join(
                    ade_index['folder'][img_index],
                    ade_index['filename'][img_index][:-4]+"_parts_1.png")
                dat_in = os.path.join(
                    ade_index['folder'][img_index],
                    ade_index['filename'][img_index][:-4]+".json")

                if imgs_per_class == 1:
                    img_out = os.path.join(output_folder,f"{c}.jpg")
                    seg_out = os.path.join(output_folder,f"{c}_seg.png")
                    par_out = os.path.join(output_folder,f"{c}_parts_1.png")
                    dat_out = os.path.join(output_folder,f"{c}.json")
                else:
                    img_out = class_folder
                    seg_out = class_folder
                    par_out = class_folder
                    dat_out = class_folder
                try:
                    shutil.copy(img_in,img_out)
                    shutil.copy(seg_in,seg_out)
                    shutil.copy(dat_in,dat_out)
                    if os.path.exists(par_in):
                        shutil.copy(par_in,par_out)
                except:
                    print("-> ERROR COPYING")
                if imgs_found >= imgs_per_class:
                    break
    if mode == "example":
        if imgs_found < imgs_per_class:
            print(f"-> Could only find {imgs_found}/{imgs_per_class} matches for {c}")
        else:
            print(f"-> Found {imgs_found} images.")
    elif mode == "stats":
        print(f"-> {imgs_found:5}/{num_images} ({float(100.0*imgs_found)/num_images:5.3}%)")
