"""Re-annotate ADE20k to contain new classes, derived from the original ones via filters defined in
a configuration json file (--ade-conf). Only images matching at least a certain number of classes are kept, which
is defined as detection_threshold in the configuration.

The re-annotated dataset is stored in a new directory (--out-dir), containing three subfolders, each of which 
present a folder structure ready for MMSegmentation:
-   indoor
-   outdoor
-   inout (containing both indoor and outdoor images)

Snippets from the generated dataset are also extracted into a separate snippets folder (--snippets-dir), 
to enable inspection of the results."""

import argparse
import os
import pickle
import shutil

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import *
import ade_utils as utils


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=path_arg, default=conf.dataset_out_path,
    help='the folder to store the result in. It will be filled with subfolders for indoor,outdoor,inout_extended etc. (default from configuration)')
parser.add_argument('--overwrite', default=False, action=argparse.BooleanOptionalAction, 
    help='whether to delete previous contents of --out-dir. If set to False, a dialog is shown in case the folder is not empty. (default: False)')
parser.add_argument('--ade-conf', type=path_arg, default=conf.annotate_filers_conf,
    help='the path to a json file with the ade-specific filters for re-annotation. (default from configuration)')
parser.add_argument('--snippets-dir', type=path_arg, default=conf.annotate_snippets_dir,
    help='the folder to store some snippets of the new dataset in. (default from configuration)')
parser.add_argument('--snippet-count', type=int, default=50,
    help='the number of images of the new dataset to also store in the snippets-dir. (default: 50)')
parser.add_argument('--snippet-every', type=int, default=200,
    help='the number of images to skip between each snippet. (default: 200)')
parser.add_argument('--confirm', default=True, action=argparse.BooleanOptionalAction, 
    help='whether a confirmation should be prompted from the user after showing the configuration and before starting the re-annotation. (default: True)')
args = parser.parse_args()

# Load configuration and index data
ade_index = utils.adeindex.load()
ade_conf = utils.AdeConfiguration.load(ade_index,args.ade_conf)

# Check output folder
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
elif len(os.listdir(args.out_dir)) > 0:
    if args.overwrite or input("Output folder",args.out_dir,"is not empty. Clear it? [y/n]").lower() in ["y","Y"]:
        shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir)
    else:
        exit()
 
# Create folder structure       
if not os.path.exists(args.snippets_dir): os.makedirs(args.snippets_dir)
for l0 in ["indoor","inout","outdoor"]:
    for l1 in ["annotations","images"]:
        for l2 in ["train","val"]:
            path = os.path.join(args.out_dir,l0,l1,l2)
            if not os.path.exists(path): os.makedirs(path)

# Copy configuration to output folder
shutil.copy(args.ade_conf,os.path.join(args.out_dir,"filters.json"))

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
imgs_loaded = 0
palette = np.concatenate([[0,0,0],ade_conf.palette]).astype(np.uint8)

# Show configuration and ask confirmation
print(palette)
print(f"Threshold: {ade_conf.detection_thres}, Images to process: {imgs_to_load}, extract {args.snippet_count} snippets taken every {args.snippets_every} images")
print(f"Palette: {palette}")
for ind, cl in ade_conf.all_classes.items():
    print(f"{ind:2d} | {cl.name}")
if not args.confirm and input("Okay?") != "y": exit()


for img_index in range(utils.num_images):
    try:
        print(progress_bar(imgs_loaded,imgs_to_load,length=20,add_numbers=True),f"o: {stats['outdoor_count']:5} i: {stats['indoor_count']:5}" ,end="\r")
        
        det = ade_conf.syn_match(ade_index,img_index)
        if det < ade_conf.detection_thres: 
            # Just checking synonyms yielded too few matches
            stats['skipped_synmatch'] += 1
            continue
        
        filename = ade_index['filename'][img_index][:-4]
        folder = ade_index['folder'][img_index]
        img_data = utils.imgdata.load(folder,filename)
        scene = img_data['scene'][0]
        
        if not scene in {"indoor","outdoor"} : 
            # Scene is not recognized
            stats['skipped_scene'] += 1
            continue
        
        if filename[4] == "t": mode = "train" 
        elif filename[4] == "v": mode = "val"
        else: 
            print("Unrecognized mode:",filename)
            stats['skipped_trainval'] += 1
            continue
        
        result = utils.image.annotate(
            ade_conf,filename,folder,img_data=img_data,
            detection_thres=ade_conf.detection_thres,stats=True)
        if result is None:
            # Also checking parent constraints yielded too few matches
            stats['skipped_fullmatch'] += 1
            continue
        
        img, matches = result
        img_png = Image.fromarray(img,mode='P')
        img_png.putpalette(palette)
        stats[scene+'_count'] += 1
        stats['total_count'] += 1
        
        img_png.save(os.path.join(args.out_dir,scene,"annotations",mode,filename+".png"),format="PNG")
        img_png.save(os.path.join(args.out_dir,"inout","annotations",mode,filename+".png"),format="PNG")
        shutil.copy(os.path.join(folder,ade_index['filename'][img_index]),
                    os.path.join(args.out_dir,scene,"images",mode,ade_index['filename'][img_index]))
        shutil.copy(os.path.join(folder,ade_index['filename'][img_index]),
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
        
        if imgs_loaded - 1 % args.snippet_every == 0 and imgs_loaded/args.snippet_every < args.snippet_count:
            rgb_img = utils.adeindex.load_img(ade_index,img_index)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(8,10))
            ax[0].imshow(img)

            # LEGEND
            legend_handles = []
            for i, match in enumerate(matches):
                if len(match) == 0: continue
                t_class = ade_conf.content_classes[i]
                legend_handles.append(mpatches.Patch(
                    color=t_class.color/255, label=f"{t_class.name} ({len(match)})"))
                    
            t_class = ade_conf.remains_classes[img_data['scene'][0]]
            legend_handles.append(mpatches.Patch(
                color=t_class.color/255, label=f"{t_class.name} (remains)"))
            ax[0].legend(bbox_to_anchor=(1,1), loc="upper left",handles=legend_handles)
            ax[1].imshow(cv2.addWeighted(img, 0.5, rgb_img, 0.5, 0.0))
            plt.tight_layout()
            plt.savefig(os.path.join(args.snippets_dir,filename + "_snippet.png"))
            plt.close('all')
        imgs_loaded += 1
        if imgs_loaded >= imgs_to_load: break
    except KeyboardInterrupt:
        exit()
    except BaseException as e:
        print("\n",e)
        stats['errors'].append({
            'img_id': img_index,
            'error': e,
            'error_print': str(e)
        })
        
print()

with open(os.path.join(args.out_dir,"stats.pkl"),"wb") as statsfile:
    pickle.dump(stats,statsfile)
