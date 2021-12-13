"""Re-annotate ADE20k to contain new classes, derived from the original ones via filters defined in
a configuration json file (--ade-conf). Only images matching at least a certain number of classes are kept, which
is defined as detection_threshold in the configuration.

The re-annotated dataset is stored in a new directory (--out-dir), containing three subfolders, each of which 
present a folder structure ready for MMSegmentation:
-   indoor
-   outdoor
-   inout (containing both indoor and outdoor images)

Snippets from the generated dataset are also extracted into a separate snippet folder (--snippet-dir), 
to enable inspection of the results."""

import argparse
import os
import pickle
import shutil
import traceback

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from prettytable import PLAIN_COLUMNS, PrettyTable

import ade_utils as utils
from utils import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.set_defaults(overwrite=False,confirm=True,test_run=False)
parser.add_argument('--out-dir', type=path_arg, default=conf.dataset_out_path, help='the folder to store the result in. It will be filled with subfolders for indoor,outdoor,inout_extended etc. (default from configuration)')
parser.add_argument('--overwrite', dest="overwrite", action="store_true",  help='delete previous contents of --out-dir. If not set, a dialog is shown in case the folder is not empty.')
parser.add_argument('--ade-conf', type=path_arg, default=conf.annotate_filers_conf, help='the path to a json file with the ade-specific filters for re-annotation. (default from configuration)')
parser.add_argument('--snippet-dir', type=path_arg, default=conf.annotate_snippet_dir, help='the folder to store some snippet of the new dataset in. (default from configuration)')
parser.add_argument('--snippet-count', type=int, default=50, help='the number of images of the new dataset to also store in the snippet-dir. (default: 50)')
parser.add_argument('--snippet-every', type=int, default=200, help='the number of images to skip between each snippet. (default: 200)')
parser.add_argument('--no-confirm', dest="confirm", action="store_false",  help='dont prompt a confirmation from the user after showing the configuration and before starting the re-annotation.')
parser.add_argument('--test-run', dest="test_run", action="store_true",  help='just annotate a few random images and store the ')
args = parser.parse_args()

# Load configuration and index data
ade_index = utils.AdeIndex.load()
ade_conf = utils.AdeConfiguration.load(ade_index,args.ade_conf)

if not args.test_run:
    # Check output folder
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    elif len(os.listdir(args.out_dir)) > 0:
        if args.overwrite or input(f"Output folder {args.out_dir} is not empty. Clear it? [y/n]").lower() in ["y","Y"]:
            print("Deleting previous countents of",args.out_dir)
            shutil.rmtree(args.out_dir)
            os.makedirs(args.out_dir)
        else:
            print("Annotate works only for a cleared folder. Pick a different one.")
            exit()
    

stats = {
    'images' : [],
    'indoor_count' : 0,
    'outdoor_count' : 0,
    'total_count' : 0,
    'errors' : [],
    'skipped_synmatch' : 0,
    'skipped_scene' : 0,
    'skipped_fullmatch' : 0,
    'skipped_trainval' : 0
}

imgs_to_load = utils.num_images
palette = np.concatenate([[0,0,0],ade_conf.palette]).astype(np.uint8)

print()

if args.test_run:
    imgs_to_load = 20
    args.snippet_count = imgs_to_load
    args.snippet_every = 1
    

# Show configuration and ask confirmation
if args.test_run:
    print(f"TEST RUN: Threshold: {ade_conf.detection_thres}, Images to process: {imgs_to_load}")
    print("All images stored in", args.snippet_dir)
else:
    print(f"Threshold: {ade_conf.detection_thres}, Images to process: {imgs_to_load}, extract {args.snippet_count} snippets taken every {args.snippet_every} images")
    print("Dataset output folder:",args.out_dir)
    print("Snippet output folder",args.snippet_dir)
    
print()
t = PrettyTable()
t.field_names = ["id","class","scene","color (rgb)"]
i = 0
t.add_row([i,"SKIPPED","",f"{palette[i*3]:3d} {palette[i*3+1]:3d} {palette[i*3+2]:3d}"])
for ind, cl in ade_conf.all_classes.items():
    i = ind+1
    t.add_row([i,cl.name,"both" if cl.scene is None else cl.scene,f"{palette[i*3]:3d} {palette[i*3+1]:3d} {palette[i*3+2]:3d}"])
#t.set_style(PLAIN_COLUMNS)
print(t)

if args.confirm and input("Okay? [y/n] ") != "y": exit()

if not os.path.exists(args.snippet_dir): os.makedirs(args.snippet_dir)
if not args.test_run:
    # Create folder structure       
    for l0 in ["indoor","inout","outdoor"]:
        for l1 in ["annotations","images"]:
            for l2 in ["train","val"]:
                path = os.path.join(args.out_dir,l0,l1,l2)
                if not os.path.exists(path): os.makedirs(path)

    # Copy configuration to output folder
    shutil.copy(args.ade_conf,os.path.join(args.out_dir,"filters.json"))
else:
    shutil.copy(args.ade_conf,os.path.join(args.snippet_dir,"filters.json"))


imgs_found = 0
snippets_made = 0
imgs_skipped = 0
for cc,img_index in enumerate(utils.AdeIndex.any_images(random=True)):
    try:
        print(progress_bar(cc,utils.num_images,length=30,add_numbers=True),
            "outdoor:", stats['outdoor_count'], 
            "indoor:", stats['indoor_count'],
            "skipped:", imgs_skipped, 
            "errors:", len(stats['errors']), 
            "snippets:", snippets_made ,end="\r")
        
        det = ade_conf.syn_match(ade_index,img_index)
        if det < ade_conf.detection_thres: 
            # Just checking synonyms yielded too few matches
            stats['skipped_synmatch'] += 1
            imgs_skipped += 1
            continue
        
        filename = ade_index['filename'][img_index][:-4]
        folder = ade_index['folder'][img_index]
        img_data = utils.ImgData.loadi(ade_index,img_index)
        scene = img_data['scene'][0]
        
        if not scene in {"indoor","outdoor"} : 
            # Scene is not recognized
            stats['skipped_scene'] += 1
            imgs_skipped += 1
            continue
        
        if filename[4] == "t": mode = "train" 
        elif filename[4] == "v": mode = "val"
        else: 
            #print("Unrecognized mode:",filename)
            stats['skipped_trainval'] += 1
            imgs_skipped += 1
            continue
        
        result = utils.Images.annotate(
            ade_conf,filename,folder,img_data=img_data,
            detection_thres=ade_conf.detection_thres,stats=True)
        if result is None:
            # Also checking parent and scene constraints yielded too few matches
            stats['skipped_fullmatch'] += 1
            imgs_skipped += 1
            continue
        
        ann_indices, matches = result
        ann_img = Image.fromarray(ann_indices,mode='P')
        ann_img.putpalette(palette)
        stats[scene+'_count'] += 1
        stats['total_count'] += 1
        if not args.test_run:
            # Save png image (annotation):
            ann_img.save(os.path.join(args.out_dir,scene,"annotations",mode,filename+".png"),format="PNG")
            ann_img.save(os.path.join(args.out_dir,"inout","annotations",mode,filename+".png"),format="PNG")
            # Copy jpg image (image):
            shutil.copy(utils.AdeIndex.img_path(ade_index, img_index),
                        os.path.join(args.out_dir,scene,"images",mode,ade_index['filename'][img_index]))
            shutil.copy(utils.AdeIndex.img_path(ade_index, img_index),
                        os.path.join(args.out_dir,"inout","images",mode,ade_index['filename'][img_index]))
        
        match_list = {}
        for i,m in enumerate(matches):
            if len(m) > 0:
                match_list[ade_conf.content_classes[i].name] = len(m)
        
        stats['images'].append({
            'id': img_index,
            'syn_matches': det,
            'full_matches': len(match_list),
            'scene': scene,
            'matches': match_list
        })
        
        if args.test_run or snippets_made < args.snippet_count:
            # Save png image (annotation):
            ann_img.save(os.path.join(args.snippet_dir,filename+".png"),format="PNG")
            # Copy jpg image (image):
            shutil.copy(utils.AdeIndex.img_path(ade_index, img_index),
                        os.path.join(args.snippet_dir,ade_index['filename'][img_index]))
                        
            rgb_img = utils.AdeIndex.load_img(ade_index,img_index,pillow=True)
            overlay_img = Image.blend(rgb_img,ann_img.convert('RGB'),0.5)
            
            fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(8,10))
            ax[0].imshow(rgb_img)
            ax[0].axis('off')
            ax[1].imshow(overlay_img)
            
            legend_handles = []
            for i, match in enumerate(matches):
                if len(match) == 0: continue
                t_class = ade_conf.content_classes[i]
                legend_handles.append(mpatches.Patch(
                    color=t_class.color/255, label=f"{t_class.name} ({len(match)})"))
            t_class = ade_conf.remains_classes[img_data['scene'][0]]
            legend_handles.append(mpatches.Patch(
                color=t_class.color/255, label=f"{t_class.name} (remains)"))
            ax[1].legend(bbox_to_anchor=(1,1), loc="upper left",handles=legend_handles)
            
            ax[1].axis('off')
            plt.tight_layout()
            plt.subplots_adjust(right=0.8)
            
            plt.savefig(os.path.join(args.snippet_dir,filename + "_vis.png"))
            plt.close('all')
            
            snippets_made += 1
            
        imgs_found += 1
        if imgs_found >= imgs_to_load: break
        
    except KeyboardInterrupt:
        break
        
    except BaseException as e:
        traceback.print_exc()
        stats['errors'].append({
            'img_id': img_index,
            'error': e,
            'error_print': str(e)
        })
        
print()

stats_dir = args.snippet_dir if args.test_run else args.out_dir
with open(os.path.join(stats_dir,"stats.pkl"),"wb") as statsfile:
    pickle.dump(stats,statsfile)
