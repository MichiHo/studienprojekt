"""Create a summary html document for a given list of classes and class-combinations, containing a
few example images with outlines and stats."""

import argparse
import json
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ade_utils as utils
from utils import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=path_arg, default="ade_summary", help='the folder to create the folder structure and store the html and image files in . (default: "ade_summary")')
parser.add_argument('-n','--classnames',nargs='*',default=[],help='single classes to look for, separated by colon (:).')
parser.add_argument('-c','--combinations',action='append',default=[],nargs='*',help='combinations of classes to look for in single images. Takes a list of ADE-class names, separated by colon (:).')
parser.add_argument('-p','--parents',action='append',nargs='*',default=[],help='combinations of parent-class and child-class to look for in single images. Expects a list of ADE-class names, separated by colon (:), where the first item is the child and all following items are allowed parents. Example: window: house: building:')
parser.add_argument('--count',type=int,default=10,help="number of examples to extract for each class / combination / parent-child pair.")
args = parser.parse_args()

if (len(args.classnames) + len(args.parents) + len(args.combinations)) == 0:
    print("Specify at least one classname / combination / parent-child set")
    exit()

print("Loading data...")
ade_index = utils.AdeIndex.load()
ade_stats = utils.AdeStats.load()
print("-> done")

##########################################################################################
## Check all inputs, show guesse for each nonexistent class and
## otherwise remove these classes

class_color_dict = dict()
args.classnames = colon_separated(args.classnames)
def guess(query):
    guesses = utils.AdeIndex.class_guess(ade_index, query)
    if len(guesses) == 0: return None
    return choice(guesses)
classnames = []
for i,cname in enumerate(args.classnames):
    if not cname[0] == '~':
        try:
            ind = utils.AdeIndex.class_index(ade_index,cname)
            class_color_dict[ind] = None
            classnames.append((ind,cname))
            continue
        except ValueError:
            pass
    else: cname = cname[1:]
    cname = guess(cname)
    if cname == None:
        print("Class",cname,"does not exist in ADE20k and is skipped.")
    else:
        ind = utils.AdeIndex.class_index(ade_index,cname)
        class_color_dict[ind] = None
        classnames.append((ind,cname))
        
    
args.combinations = [colon_separated(comb) for comb in args.combinations]
combinations = []
for i,combi in enumerate(args.combinations):
    res_combination = []
    for j,cname in enumerate(combi):
        if not cname[0] == '~':
            try:
                ind = utils.AdeIndex.class_index(ade_index,cname)
                class_color_dict[ind] = None
                res_combination((ind,cname))
                continue
            except ValueError: pass
        else: cname = cname[1:]
        cname = guess(cname)
        if cname == None:
            print("Class",cname,"does not exist in ADE20k and the combination containing it is skipped.")
            res_combination = None
            break
        else:
            ind = utils.AdeIndex.class_index(ade_index,cname)
            class_color_dict[ind] = None
            res_combination((ind,cname))
    if res_combination is not None:
        combinations.append(res_combination)
        
args.parents = [colon_separated(comb) for comb in args.parents]
child_parents = []
for i,combi in enumerate(args.parents):
    res_child_parent = []
    for j,cname in enumerate(combi):
        if cname.upper() == "NONE" and j > 0:
            res_child_parent.append((-1,"NONE"))
        elif not cname[0] == '~':
            try:
                ind = utils.AdeIndex.class_index(ade_index,cname)
                class_color_dict[ind] = None
                res_child_parent.append((ind,cname))
                continue
            except ValueError: pass
        else: cname = cname[1:]
        cname = guess(cname)
        if cname == None:
            if j == 0:
                print("Class",cname,"does not exist in ADE20k and the parents-child set containing it is skipped.")
                args.parents.pop(i)
                res_child_parent = None
                break
            else:
                print("Class",cname,"does not exist in ADE20k and is not looked for as parent.")
                continue
        else:
            ind = utils.AdeIndex.class_index(ade_index,cname)
            class_color_dict[ind] = None
            res_child_parent.append((ind,cname))
    if res_child_parent is not None:
        child_parents.append(res_child_parent)
        
num_total_items = len(args.classnames) + len(args.parents) + len(args.combinations)
if num_total_items == 0:
    print("Specify at least one classname / combination / parent-child set")
    exit()

##########################################################################################
## Create color scheme and legend for all classes present in the arguments

cmap = plt.get_cmap("Dark2")
classes_colors = []
legend = ""
for i,class_id in enumerate(class_color_dict.keys()):
    c = cmap(i)
    color = (c[0]*255,c[1]*255,c[2]*255)
    class_color_dict[class_id] = color
    # class, parent, rgb color, thickness
    classes_colors.append([class_id,None,color,2])
    legend = legend + f"<span style='text-decoration: underline solid 4px {HtmlContext.color(color)};' class='inline-legend-item'>{utils.AdeIndex.classname(ade_index, class_id)}</span>"

parent_color = (45, 139, 209)
child_color = (209, 115, 45)
outline_color = [255,0,0]

current_item = 0
goal_width = 300
def scalefac(img):
    return goal_width/img.shape[0]  


with HtmlContext(args.out_dir,"ADE20k Summary") as w:
    
    ##########################################################################################
    ## Single Classes
    
    if len(classnames) > 0:
        w("<details open class='l0'><summary class='part'>Single classes</summary>")
        for class_index,classname in classnames:
            current_item += 1
            print(f"({current_item}/{num_total_items})Single class '{classname}' (#{class_index})")
            
            # Create barplot
            barplot_name = f"{classname}_barplot.png"
            barplot_path = w.imgpath(barplot_name)
            utils.Plots.parent_stats(ade_index,ade_stats,class_index,os.path.join(args.out_dir,barplot_path))

            # Some infos
            w(f"<div class='section'><h1><span class='box' style='background-color:{HtmlContext.color(outline_color)}'></span>{classname}</h1>")
            w(f'''<div class="summary">
            {HtmlContext.item("image count",               ade_stats['classes'][class_index]["image_count"])}
            {HtmlContext.item("instance count (calculated)", ade_stats['classes'][class_index]['object_count'])}
            {HtmlContext.item("wordnet_gloss",     ade_index['wordnet_gloss'][class_index])}
            {HtmlContext.item("wordnet_frequency", ade_index['wordnet_frequency'][class_index])}
            {HtmlContext.item("wordnet_hypernym",  ade_index['wordnet_hypernym'][class_index])}
            {HtmlContext.item("wordnet_synset",    ade_index['wordnet_synset'][class_index])}
            <img src='{barplot_path}'>
            </div>''')
            
            # Outlines
            w(f'''<div class='examples'>
            <h2>Examples{legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            for img_index in tqdm(utils.AdeIndex.images_with_class(ade_index,class_index,args.count,random=True),total=args.count,desc="Search for examples"):
                filename = ade_index['filename'][img_index][:-4]
                #foldername = ade_index['folder'][img_index]
                out_name = f"{filename}_{classname}_outlines.jpg"
                out_path = os.path.join(w.img_folder,out_name)
                img, img_data = utils.AdeIndex.load_img(ade_index, img_index,load_imgdata=True)
                # img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                # img_data = utils.ImgData.load(foldername,filename)
                # Highlight this class
                classes_colors.append((class_index,None,class_color_dict[class_index],8 ))
                img,obj_counts = utils.Images.class_outlines(img,img_data,classes_colors,False,True,scaling=scalefac(img))
                classes_colors.pop()
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{w.imgpath(out_name)}'></div>''')
            w("</div></div></div>")
        w("</details>")
    
    ##########################################################################################
    ## Class combinations
    
    if len(combinations) > 0:
        w("<div class='part'>Combinations</div>")
        w("<div class='section'>")
        for i,combi in enumerate(combinations):
            combi_indices = [i for i,n in combi]
            title = " &amp; ".join([n for i,n in combi])
            current_item += 1
            print(f"({current_item}/{num_total_items}) Combination '{' & '.join([f'{n}(#{i})' for i,n in combi])}'")
        
            all_matches = list(utils.AdeIndex.images_with_classes(ade_index,combi_indices,random=True))
            scenes = dict()
            for item in all_matches:
                scene = ade_index['scene'][item].strip(" /")
                if not scene in scenes: scenes[scene] = 1
                else: scenes[scene] += 1
            
            w(f'''<div class='examples'>
            <div class="summary">
            {HtmlContext.item("total matches", len(all_matches))}
            {' '.join([HtmlContext.item(scene+" matches", count) for scene,count in scenes.items()])}
            </div>
            <h2>{title} {legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            
            for img_index in tqdm(all_matches[:args.count],desc="Search for examples"):
                filename = ade_index['filename'][img_index][:-4]
                #foldername = ade_index['folder'][img_index]
                out_name = f"{filename}_combi{i}_outlines.jpg"
                out_path = os.path.join(w.img_folder,out_name)
                img, img_data = utils.AdeIndex.load_img(ade_index, img_index,load_imgdata=True)
                # img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                # img_data = utils.ImgData.load(foldername,filename)
                img,obj_counts = utils.Images.class_outlines(img,img_data,classes_colors,False,True,scaling=scalefac(img))
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{w.imgpath(out_name)}'></div>''')
            w("</div></div>")
        w("</div>")


    ##########################################################################################
    ## Parent-child combinations
    
    if len(child_parents) > 0:
        w(f'''<div class='part'>
        Parents
        <p class="subtext">
        Thick outlines for <span style="color:{HtmlContext.color(parent_color)}">parent</span> and 
        <span style="color:{HtmlContext.color(child_color)}">child</span> object.
        </p>
        </div>''')
        w("<div class='section'>")

        for child,*parents in child_parents:
            child_class_id, child_class_name = child
            w(f"<div><h2>{child_class_name}</h2>")
            
            parents_left = { i for i,n in parents}
            
            current_item += 1
            print(f"({current_item}/{num_total_items}) Child: {child_class_name} Parents: {' & '.join([f'{n}(#{i})' for i,n in parents])}")
            t = tqdm(total=len(parents),desc="Search for one example per parent")
            for img_index in utils.AdeIndex.images_with_class(ade_index,child_class_id,random=True):
                #print(f"   IMG {img_index}",end="\r")
                filename = ade_index['filename'][img_index][:-4]
                #foldername = ade_index['folder'][img_index]
                img_data = utils.AdeIndex.load_img(ade_index, img_index,
                                load_imgdata=True,load_training_image=False)
                #img_data = utils.ImgData.load(foldername,filename)
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
                        t.update(len(parents) - len(parents_left))
                        if parent_class_id >= 0:
                            parent_class = ade_index['objectnames'][parent_class_id]
                        else:
                            parent_class = "NONE"
                        out_name = f"{filename}_{child_class_name}_of_{parent_class}_outlines.jpg"
                        out_path = os.path.join(w.img_folder,out_name)
                        img = utils.AdeIndex.load_img(ade_index, img_index)
                        #img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                        img = utils.Images.class_outlines(img,img_data,classes_colors,legend=False,highlight_instances=[
                            { 
                                "id": parent_id,
                                "color": parent_color,
                                "thickness": 4
                            },
                            {
                                "id": child_instance['id'],
                                "color": child_color,
                                "thickness": 4
                            }
                        ],scaling=scalefac(img))
                        cv2.imwrite(out_path,img)
                        w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="capt">{parent_class}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{w.imgpath(out_name)}'></div>''')
                        
                        if len(parents_left) == 0:
                            break
                if len(parents_left) == 0:
                    break
            t.close()
            w("</div>")
        w("</div>")
    