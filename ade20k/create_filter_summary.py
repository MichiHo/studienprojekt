import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import *
import ade_utils as utils

ade_index = utils.adeindex.load()

conf = utils.AdeConfiguration.load(ade_index,"check_out_filters.json")
#conf = utils.configuration.load(ade_index,"filters.json")

#out_folder = os.path.expanduser("~/ssh_transfer/filtertest")
out_folder = os.path.expanduser("~/ssh_transfer/filter_others")
#out_folder = "output/filter_test"
img_folder = os.path.join(out_folder,"img")
if not os.path.exists(img_folder): os.makedirs(img_folder)

with HtmlContext(out_folder,"Filter Test") as w:
    w(f"""<script>
    var masonries = {{}};
    function grid(id){{
        masonries[id] = new Masonry( '#grid-'+id, {{ 
            "itemSelector": ".grid-item", 
            "columnWidth": 100, 
            "gutter": 3 }});
        
    }}
    function relayout(id){{
        masonries[id].layout()
    }}
    </script>""")
    
    for tclass in conf.content_classes:
        print("CLASS",tclass.name)
        scene_str = "both" if tclass.scene is None else tclass.scene
        w(f"""<div class='section'>
        <h2><span class='box' style='background-color:{utils.html.color(tclass.color)}'></span>
        {tclass.name}<span class="header-info scene-{scene_str}">
        {scene_str}</span>
        <span class="header-info">z: {tclass.z_index}</span></h2>
        <div class="summary synonyms img-grid grid" id="grid-{tclass.name}">""")
        
        if isinstance(tclass.synonyms,str):
            w(tclass.synonyms)
        else:
            for syn_class, parents in tclass.synonyms.items():
                w(f"""<div class="synonym grid-item">
                <h2>{utils.adeindex.classname(ade_index,syn_class)}</h2>""")
                if len(parents) > 0:
                    w(f"""<details{" open" if len(parents) < 6 else ""} class='parents' ontoggle='relayout("{tclass.name}")'>
                    <summary>One of {len(parents)} parents required</summary>
                    <p> 
                    {"</p><p>".join(utils.adeindex.classnames(ade_index,parents))}
                    </p></details>""")
                w("""</div>""") 
        w("</div>")
        w(f"""<script>
        grid("{tclass.name}")
        </script>""")
        w(f"""<div class='examples'>
        <p>Instances of synonyms in 
        <span style='color:{utils.html.color((255,0,0))}'>thin outline</span> 
        and those with matched parents with 
        <span style='color:{utils.html.color((255,255,0))}'>thick outline</span>
        </p><div class='img-grid grid' id='grid-{tclass.name}-examples'>""")
        # Find example
        synmatch = 0
        scenematch = 0
        found = 0
        grid_id = f"{tclass.name}-examples"
        for i in range(utils.num_images):
            if tclass.syn_match(ade_index,i):
                synmatch += 1
                img_data = utils.imgdata.loadi(ade_index,i)
                if not tclass.scene_match(img_data): continue
                scenematch += 1
                instances = tclass.full_match(img_data)
                if len(instances) > 0:
                    print(" - MATCH",found,":",len(instances),"instances")
                    out_name = f"{tclass.name}_example_{found}.jpg"
                    out_path = os.path.join(img_folder,out_name)
                    img = utils.adeindex.load_img(ade_index,i,False)
                    img = utils.image.class_outlines(img,img_data,
                        [[synname,None,(255,0,0),3] for synname in tclass.synonyms],
                        legend=False,add_info=False,
                        highlight_instances=[{
                            "id": o['id'],
                            "color": (255,255,0),
                            "thickness": 5
                        } for o in instances])
                    cv2.imwrite(out_path,img)
                    parents = {
                        utils.imgdata.find_obj_by_id(img_data,o['parts']['ispartof'])['name'] if o['parts']['ispartof'] != [] else "NONE"
                        for o in instances
                    }
                    w(f"<img class='grid-item' title={parents} src='img/{out_name}'>")
                    found += 1
                    if found > 20: break
        
        w(f"""</div><script>
        grid("{tclass.name}-examples")
        </script>""")
        w(f"<p>Searched {i} images, skipping {synmatch-1} synonym-matches, {scenematch-1} scene-matches and finding {len(instances)} parent-matching instances.</p>")
        w("</div>")
        w("</div>")
