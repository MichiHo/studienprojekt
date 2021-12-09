"""Create a summary html document for a given list of classes and class-combinations, containing a
few example images with outlines and stats."""

import json
import os
import random
import sys
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import *
import ade_utils as utils



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=path_arg, default="ade_summary", help='the folder to create the folder structure and store the html and image files in . (default: "ade_summary")')
parser.add_argument('-n','--classnames',nargs='*',default=[],help='single classes to look for, separated by colon (:).')
parser.add_argument('-c','--combinations',action='append',default=[],nargs='*',help='combinations of classes to look for in single images. Takes a list of ADE-class names, separated by colon (:).')
parser.add_argument('-p','--parents',action='append',nargs='*',default=[],help='combinations of parent-class and child-class to look for in single images. Expects a list of ADE-class names, separated by colon (:), where the first item is the child and all following items are allowed parents. Example: window: house: building:')
parser.add_argument('--count',type=int,help="number of examples to extract for each class / combination / parent-child pair.")
# parser.add_argument('logfiles', type=path_arg, nargs='*', default=None,  help='the training log json to visualize.')
# parser.add_argument('--plots-dir', type=path_arg, default=conf.segmentation_plots_path, help='the folder to store generated plots into. Will be created if not existent. No files are overwritten. (default from configuration)')
# parser.add_argument('--save-plots', default=True, action=argparse.BooleanOptionalAction,  help='whether to save plots as svg into a folder.')
# parser.add_argument('--epochs', default=False, action=argparse.BooleanOptionalAction,  help='whether to use epochs instead of iterations as time value.')
# parser.add_argument('--mode', default='joint', choices=['joint','separate'], help='whether to display multiple training logs in one joint plot or in separate plots. (default: joint)')
# parser.add_argument('--class-property', default="IoU", choices=['IoU','Acc'],  help='the property displayed in the class-wise plot. (default: IoU)')
# parser.add_argument('--global-property', default="mIoU", choices=['mIoU','aAcc','mAcc','lr'],  help='the property displayed over time in the total plot. (default: mIoU)')
# parser.add_argument('--second-global-property', default=None, choices=['mIoU','aAcc','mAcc','lr'],  help='which other property to plot along the global-property on the left plot. (default: None)')
# parser.add_argument('--colors',type=str,default=["red","blue","green","orange","purple","pink"],nargs='+', help='colors to assign to the training logs in that order.')
# parser.add_argument('--names',type=str,nargs='+',default=[], help='names to assign to the training logs in that order. (defaults to the main part of the filename or the name specified in the log)')
# parser.add_argument('--time-range',choices=['min','max','ask'],default='ask', help="how to behave in case of different time spans of the training logs. 'min' crops all logs by the shortest length, 'max' displays all logs completely and 'ask' asks interactively. (default: ask)")
args = parser.parse_args()

if (len(args.classnames) + len(args.parents) + len(args.combinations)) == 0:
    print("Specify at least one classname / combination / parent-child set")
    exit()

print("Loading data...")
ade_index = utils.AdeIndex.load()
ade_stats = utils.AdeStats.load()
print("-> done")

class_color_dict = set()
args.classnames = colon_separated(args.classnames)
for i,cname in enumerate(args.classnames):
    try:
        ind = utils.AdeIndex.class_index(ade_index,cname)
        class_color_dict[ind] = None
        args.classnames[i] = (ind,cname)
    except ValueError:
        print("Class",cname,"does not exist in ADE20k and is skipped.")
        args.classnames.pop(i)
    
args.combinations = [colon_separated(comb) for comb in args.combinations]
for i,combi in enumerate(args.combinations):
    for j,cname in enumerate(combi):
        try:
            ind = utils.AdeIndex.class_index(ade_index,cname)
            class_color_dict[ind] = None
            args.combinations[i][j] = (ind,cname)
        except ValueError:
            print("Class",cname,"does not exist in ADE20k and the combination containing it is skipped.")
            args.combinations.pop(i)
            break
args.parents = [colon_separated(comb) for comb in args.parents]
for i,combi in enumerate(args.parents):
    for j,cname in enumerate(combi):
        try:
            if cname.upper() == "NONE" and j > 0:
                args.parents[i][j] = (-1,"NONE")
            else:
                ind = utils.AdeIndex.class_index(ade_index,cname)
                class_color_dict[ind] = None
                args.parents[i][j] = (ind,cname)
        except ValueError:
            print("Class",cname,"does not exist in ADE20k and the parents-child set containing it is skipped.")
            args.parents.pop(i)
            break


# Create color scheme and legend for all classes present in the arguments
cmap = plt.get_cmap("Dark2")
classes_colors = []
legend = ""
for i,name in enumerate(class_color_dict.keys()):
    n_id = utils.AdeIndex.class_index(ade_index,name)
    color = cmap(i)
    class_color_dict[name] = color
    classes_colors.append([n_id,None,color,None])
    legend = legend + f"<span style='text-decoration: underline solid 4px {html.color(color)};' class='inline-legend-item'>{name}</span>"


img_folder = os.path.join(args.out_dir,"imgs")
if not os.path.exists(img_folder):
    os.makedirs(img_folder)
    
parent_color = (45, 139, 209)
child_color = (209, 115, 45)
outline_color = [255,0,0]




with HtmlContext(args.out_dir,"ADE20k Summary") as w:
    # Single Classes
    if len(args.classnames) > 0:
        w("<details class='l0'><summary>Single classes</summary>")
        for class_index,classname in args.classnames:
            print(f"Single class '{classname}'")
            
            # Create barplot
            barplot_name = f"{classname}_barplot.png"
            barplot_path = os.path.join(img_folder,barplot_name)
            utils.Plots.parent_stats(ade_index,ade_stats,class_index,barplot_path)

            # Some infos
            w(f"<div class='section'><h1><span class='box' style='background-color:{html.color(outline_color)}'></span>{classname}</h1>")
            w(f'''<div class="summary">
            {html.item("image count",               ade_stats[classname]["image_count"])}
            {html.item("instance count (calculated)", ade_stats[class_index]['object_count'])}
            {html.item("wordnet_gloss",     index['wordnet_gloss'][class_index])}
            {html.item("wordnet_frequency", index['wordnet_frequency'][class_index])}
            {html.item("wordnet_hypernym",  index['wordnet_hypernym'][class_index])}
            {html.item("wordnet_synset",    index['wordnet_synset'][class_index])}
            </div>''')
            
            # Outlines
            w(f'''<div class='examples'>
            <h2>Examples{legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            for img_index in utils.AdeIndex.images_with_class(ade_index,class_index,args.count,random=True):
                filename = ade_index['filename'][img_index][:-4]
                foldername = ade_index['folder'][img_index]
                out_name = f"{filename}_{classname}_outlines.jpg"
                out_path = os.path.join(img_folder,out_name)
                img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                img_data = utils.ImgData.load(foldername,filename)
                # Highlight this class
                classes_colors.append((class_index,None,class_color_dict[class_index],8 ))
                img,obj_counts = utils.Images.class_outlines(img,img_data,classes_colors,False,True)
                classes_colors.pop()
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{os.path.join('imgs',out_name)}'></div>''')
            w("</div></div></div>")
        w("</details>")
    
    # Combinations
    if len(args.combinations) > 0:
        w("<div class='part'>Combinations</div>")
        w("<div class='section'>")
        for i,combi in enumerate(args.combinations):
            combi_indices = [i for i,n in combi]
            title = " &amp; ".join([n for i,n in combi])
        
            w(f'''<div class='examples'>
            <h2>{title} {legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            for img_index in utils.AdeIndex.images_with_classes(ade_index,combi_indices,12,0):
                filename = ade_index['filename'][img_index][:-4]
                foldername = ade_index['folder'][img_index]
                out_name = f"{filename}_combi{i}_outlines.jpg"
                out_path = os.path.join(img_folder,out_name)
                img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                img_data = utils.ImgData.load(foldername,filename)
                img,obj_counts = utils.Images.class_outlines(img,img_data,classes_colors,False,True)
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{os.path.join('imgs',out_name)}'></div>''')
            w("</div></div>")
        w("</div>")


    # Parents
    if len(args.parents) > 0:
        w(f'''<div class='part'>
        Parents
        <p class="subtext">
        Thick outlines for <span style="color:{html.color(parent_color)}">parent</span> and 
        <span style="color:{html.color(child_color)}">child</span> object.
        </p>
        </div>''')
        w("<div class='section'>")

        for child,*parents in args.parents:
            child_class_id, child_class_name = child
            print("Child",child_class_name)
            w(f"<div><h2>{child_class_name}</h2>")
            
            parents_left = { i for i,n in parents}
            
            print("-> Parents =",parents_left,"                                             ")
            
            for img_index in utils.AdeIndex.images_with_class(ade_index,child_class_id):
                #print(f"   IMG {img_index}",end="\r")
                filename = ade_index['filename'][img_index][:-4]
                foldername = ade_index['folder'][img_index]
                img_data = utils.ImgData.load(foldername,filename)
                for child_instance in utils.ImgData.objects_of_class(img_data,child_class_id):
                    parent_id = child_instance['parts']['ispartof']
                    if type(parent_id) == int:
                        parent_instance = utils.ImgData.find_obj_by_id(img_data,parent_id)
                        parent_class_id = parent_instance['name_ndx']
                    else:
                        parent_class_id = -1
                    #print(f"   {child_instance['name']} of {parent_class_id}", end="\r")
                    if parent_class_id in parents_left:
                        parents_left.remove(parent_class_id)
                        if parent_class_id >= 0:
                            parent_class = ade_index['objectnames'][parent_class_id]
                        else:
                            parent_class = "NONE"
                        print(f"   -> Found {parent_instance['name']} ({parent_class_id})      ")
                        out_name = f"{filename}_{child_class_name}_of_{parent_class}_outlines.jpg"
                        out_path = os.path.join(img_folder,out_name)
                        img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                        img = utils.Images.class_outlines(img,img_data,classes_colors,legend=False,highlight_instances=[
                            { 
                                "id": parent_id,
                                "color": parent_color,
                                "thickness": 6
                            },
                            {
                                "id": child_instance['id'],
                                "color": child_color,
                                "thickness": 6
                            }
                        ])
                        cv2.imwrite(out_path,img)
                        w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="capt">{parent_class}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{os.path.join('imgs',out_name)}'></div>''')
                        
                        if len(parents_left) == 0:
                            break
                if len(parents_left) == 0:
                    break

            w("</div>")
        w("</div>")
    

    #w(f'''</body>''')
