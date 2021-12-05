import json
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ade_utils as utils


class html(object):
    @staticmethod
    def color(cl):
        return f"rgb({cl[0]},{cl[1]},{cl[2]})"
    
    def item(title,val):
        return f"<p class='item{' numeric-item' if isinstance(val,(int,float,complex)) else ''}'><span class='title'>{title}</span><span class='val'>{val}</span></p>"

print("Loading data...")
ade_index = utils.adeindex.load()
class_stats = pd.read_csv("data_stats.csv",delimiter=";",index_col=False,header=0,names=[
        "name",
        "objectcounts", 
        "proportionClassIsPart", 
        "imgcount", 
        "instcount"]).fillna("")
class_stats = class_stats.set_index('name')
class_stats_2 = utils.classes_new.load()
print("-> done")

out_folder = "~/ssh_transfer/classes"
out_folder = os.path.expanduser(out_folder)
img_folder = os.path.join(out_folder,"imgs")
if not os.path.exists(img_folder):
    os.makedirs(img_folder)
    
show_overviews = False
show_combinations = False
show_parents = True

parent_color = (45, 139, 209)
child_color = (209, 115, 45)


classes = [
    ["windowpane, window",(150,0,0)],
    ["window",(255,0,0)],
    ["door",(0,0,255)],
    ["wall",(0,255,0)],
    ["ceiling",(255,255,0)]
]

target_classes = [
    {
        'name': "window",
        'synonyms': [
            {
                'name': ""
            }
        ]
    }
]

classes_colors = []
legend = ""
for name, col in classes:
    n_id = utils.adeindex.class_index(ade_index,name)
    classes_colors.append([n_id,None,col,None])
    classes_colors.append([n_id,utils.adeindex.class_index(ade_index,"building"),col,2])
    legend = legend + f"<span style='text-decoration: underline solid 4px {html.color(col)};' class='inline-legend-item'>{name}</span>"

combinations = [
    ["window","wall","ceiling"],
    ["window","door","wall","ceiling"],
    ["window","wall","building"],
    ["windowpane, window", "window"]
]

parents_choice = [
    {
        "child": "door",
        "parents": [
            "double door",
            "table",
            "windowpane, window",
            "pantry",
            "door",
            "doble door"
        ] 
    },
    {
        "child": "window",
        "parents": [
            "window"
        ]
    },
    {
        "child": "ceiling",
        "parents": [
            "NONE",
            "double door",
            "car, auto, automobile, machine, motorcar"
        ]
    }
]




with utils.html(os.path.join(out_folder,"index.htm"),"ADE Summary") as w:

# with open(os.path.join(out_folder,"index.htm"),"w") as filehandle:
#     w = utils.html(filehandle,"ADE Summary")
    w("hans")

    # Target Classes
    if show_overviews:
        w("<details class='l0'><summary>Target classes</summary>")
        classes_index_stuff = 0
        for classname, color in classes:
            print(f"Class {classname}")
            class_index = utils.adeindex.class_index(ade_index,classname)
            
            # Create barplot
            barplot_name = f"{classname}_barplot.png"
            barplot_path = os.path.join(img_folder,barplot_name)
            utils.plots.parent_stats(ade_index,class_stats_2,class_index,barplot_path)

            # Some infos
            w(f"<div class='section'><h1><span class='box' style='background-color:{html.color(color)}'></span>{classname}</h1>")
            w(f'''<div class="summary">
            {html.item("image count",               class_stats.loc[classname,"imgcount"])}
            {html.item("instance count",            class_stats.loc[classname,"instcount"])}
            {html.item("image count (objects.txt)", class_stats.loc[classname,"objectcounts"])}
            {html.item("proportion class is part",  class_stats.loc[classname,"proportionClassIsPart"])}
            {html.item("instance count (calculated)", class_stats_2[class_index]['object_count'])}
            {html.item("wordnet_gloss",     index['wordnet_gloss'][class_index])}
            {html.item("wordnet_frequency", index['wordnet_frequency'][class_index])}
            {html.item("wordnet_hypernym",  index['wordnet_hypernym'][class_index])}
            {html.item("wordnet_synset",    index['wordnet_synset'][class_index])}
            <img src='{os.path.join('imgs',out_name)}'>
            </div>''')
            # {html.item("wordnet_synonyms",index['wordnet_synonyms'][class_index])}
            # Outlines
            w(f'''<div class='examples'>
            <h2>Examples{legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            for img_index in utils.iadeindexndex.images_with_class(iade_indexndex,class_index,6,random.randint(0,int(0.5*utils.num_images))):
                filename = iade_indexndex['filename'][img_index][:-4]
                foldername = iade_indexndex['folder'][img_index]
                out_name = f"{filename}_{classname}_outlines.jpg"
                out_path = os.path.join(img_folder,out_name)
                img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                img_data = utils.imgdata.load(foldername,filename)
                # Highlight this class
                classes_colors[classes_index_stuff*2][3] = 8 
                img,obj_counts = utils.image.class_outlines(img,img_data,classes_colors,False,True)
                classes_colors[classes_index_stuff*2][3] = None
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{os.path.join('imgs',out_name)}'></div>''')
            w("</div></div></div>")
            classes_index_stuff += 1
        w("</details>")
    
    # Combinations
    if show_combinations:
        w("<div class='part'>Combinations</div>")
        w("<div class='section'>")
        for i,combi in enumerate(combinations):
            combi_indices = [utils.iadeindexndex.class_index(iade_indexndex,cl) for cl in combi]
        
            w(f'''<div class='examples'>
            <h2>{combi} {legend}</h2>
            <div class='img-grid grid' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>''')
            for img_index in utils.iadeindexndex.images_with_classes(iade_indexndex,combi_indices,12,0):
                filename = iade_indexndex['filename'][img_index][:-4]
                foldername = iade_indexndex['folder'][img_index]
                out_name = f"{filename}_combi{i}_outlines.jpg"
                out_path = os.path.join(img_folder,out_name)
                img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                img_data = utils.imgdata.load(foldername,filename)
                img,obj_counts = utils.image.class_outlines(img,img_data,classes_colors,False,True)
                cv2.imwrite(out_path,img)
                w(f'''<div title="{img_data['scene']}" class='grid-item'><p class="scene">{img_data['scene']}</p><img {'class="portrait" ' if img.shape[0] > img.shape[1] else ""} src='{os.path.join('imgs',out_name)}'></div>''')
            w("</div></div>")
        w("</div>")


    # Parents
    if show_parents:
        w(f'''<div class='part'>
        Parents
        <p class="subtext">
        Thick outlines for <span style="color:{html.color(parent_color)}">parent</span> and 
        <span style="color:{html.color(child_color)}">child</span> object.
        </p>
        </div>''')
        w("<div class='section'>")

        for data in parents_choice:
            print("Child =",data['child'])
            w(f"<div><h2>{data['child']}</h2>")
            child_class_id = utils.iadeindexndex.class_index(iade_indexndex,data['child'])
            parents_left = [ 
                -1 if cname == "NONE" else utils.iadeindexndex.class_index(iade_indexndex,cname) 
                for cname in data['parents']]
            
            print("-> Parents =",parents_left,"                                             ")
            
            for img_index in utils.iadeindexndex.images_with_class(iade_indexndex,child_class_id):
                #print(f"   IMG {img_index}",end="\r")
                filename = iade_indexndex['filename'][img_index][:-4]
                foldername = iade_indexndex['folder'][img_index]
                img_data = utils.imgdata.load(foldername,filename)
                for child_instance in utils.imgdata.objects_of_class(img_data,child_class_id):
                    parent_id = child_instance['parts']['ispartof']
                    if type(parent_id) == int:
                        parent_instance = utils.imgdata.find_obj_by_id(img_data,parent_id)
                        parent_class_id = parent_instance['name_ndx']
                    else:
                        parent_class_id = -1
                    #print(f"   {child_instance['name']} of {parent_class_id}", end="\r")
                    if parent_class_id in parents_left:
                        parents_left.remove(parent_class_id)
                        if parent_class_id >= 0:
                            parent_class = iade_indexndex['objectnames'][parent_class_id]
                        else:
                            parent_class = "NONE"
                        print(f"   -> Found {parent_instance['name']} ({parent_class_id})      ")
                        out_name = f"{filename}_{data['child']}_of_{parent_class}_outlines.jpg"
                        out_path = os.path.join(img_folder,out_name)
                        img = cv2.imread(os.path.join(foldername,filename+".jpg"))
                        img = utils.image.class_outlines(img,img_data,classes_colors,legend=False,highlight_instances=[
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
