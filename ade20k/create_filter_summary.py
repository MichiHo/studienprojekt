"""Creates a summary html document for a given filter specification, containing the definition of each target class, random image examples with outlines around matched instances, and statistics. Use --full-count to count matches in all images (will still only extract --count many examples)."""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ade_utils as utils
from utils import *

parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(full_count=False)
parser.add_argument('--conf-path', type=path_arg, default=conf.annotate_filers_conf, help='the path of the filter configuration. (default from configuration)')
parser.add_argument('--out-dir', type=path_arg, default="filter_summary", help='the folder to create the folder structure and store the html and image files in . (default: "filter_summary")')
parser.add_argument('--count',type=int,default=10,help="number of examples to extract for each target class.")
parser.add_argument('--full-count', action="store_true",dest="full_count", help='whether to process the whole dataset and show how many matches were made per class. Otherwise only --count examples are processed. (default: False)')
args = parser.parse_args()

ade_index = utils.AdeIndex.load()
conf = utils.AdeConfiguration.load(ade_index,args.conf_path)

goal_width = 300
def scalefac(img):
    return goal_width/img.shape[0]  
    
with HtmlContext(args.out_dir,"Filter Summary") as w:
    # w(f"""<script>
    # var masonries = {{}};
    # function grid(id){{
    #     masonries[id] = new Masonry( '#grid-'+id, {{ 
    #         "itemSelector": ".grid-item", 
    #         "columnWidth": 100, 
    #         "gutter": 3 }});
        
    # }}
    # function relayout(id){{
    #     masonries[id].layout()
    # }}
    # </script>""")
    
    for tclass in conf.content_classes:
        print("CLASS",tclass.name)
        scene_str = "both" if tclass.scene is None else tclass.scene
        w(f"""<div class='section'>
        <h2><span class='box' style='background-color:{w.color(tclass.color)}'></span>
        {tclass.name}<span class="header-info scene-{scene_str}">
        {scene_str}</span>
        <span class="header-info">z: {tclass.z_index}</span></h2>
        <div class="summary synonyms img-grid grid" id="grid-{tclass.name}">""")
        
        if isinstance(tclass.synonyms,str):
            w(tclass.synonyms)
        else:
            for syn_class, parents in tclass.synonyms.items():
                w(f"""<div class="synonym grid-item">
                <h2>{utils.AdeIndex.classname(ade_index,syn_class)}</h2>""")
                if len(parents) > 0:
                    w(f"""<details{" open" if len(parents) < 6 else ""} class='parents' ontoggle='relayout("{tclass.name}")'>
                    <summary>One of {len(parents)} parents required</summary>
                    <p> 
                    {"</p><p>".join(utils.AdeIndex.classnames(ade_index,parents))}
                    </p></details>""")
                w("""</div>""") 
        w("</div>")
        w(f"""<script>
        grid("{tclass.name}")
        </script>""")
        w(f"""<div class='examples'>
        <p>Instances of synonyms in 
        <span style='color:{w.color((255,0,0))}'>thin outline</span> 
        and those with matched parents with 
        <span style='color:{w.color((255,255,0))}'>thick outline</span>
        </p>""")
        
        ##########################################################################################
        ## Find examples (and process all remaining images if args.full_count)

        synmatch = 0
        scenematch = 0
        fullmatch = 0
        instancesum = 0
        if args.full_count: progressbar = tqdm(total=utils.num_images,desc="Process all images")
        else: progressbar = tqdm(total=args.count,desc=f"Look for {args.count} examples only")
        html_buffer = f"""<div class='img-grid grid' id='grid-{tclass.name}-examples' data-masonry='{{ "itemSelector": ".grid-item", "columnWidth": 100, "gutter": 3 }}'>"""
        for img_count,img_id in enumerate(utils.AdeIndex.any_images(random=True)):
            if tclass.syn_match(ade_index,img_id):
                synmatch += 1
                img_data = utils.ImgData.loadi(ade_index,img_id)
                if not tclass.scene_match(img_data): continue
                scenematch += 1
                instances = tclass.full_match(img_data)
                if len(instances) <= 0: continue
                fullmatch += 1
                instancesum += len(instances)
                if fullmatch <= args.count:
                    out_name = f"{tclass.name}_example_{fullmatch}.jpg"
                    out_path = w.imgpath(out_name)
                    img = utils.AdeIndex.load_img(ade_index,img_id,False)
                    img = utils.Images.class_outlines(img,img_data,
                        [[synname,None,(255,0,0),3] for synname in tclass.synonyms],
                        legend=False,add_info=False,
                        highlight_instances=[{
                            "id": o['id'],
                            "color": (255,255,0),
                            "thickness": 5
                        } for o in instances],scaling=scalefac(img))
                    cv2.imwrite(os.path.join(args.out_dir,out_path),img)
                    parents = {
                        utils.ImgData.find_obj_by_id(img_data,o['parts']['ispartof'])['name'] if o['parts']['ispartof'] != [] else "NONE"
                        for o in instances
                    }
                    html_buffer += f"<img class='grid-item' title={parents} src='{out_path}'>"
                        
            if args.full_count:
                if fullmatch >= args.count:
                    break
                progressbar.update(n=img_count)
            else:
                progressbar.update(n=fullmatch)
        progressbar.close()   
        html_buffer += "</div>"
        w(f'''<div class="summary">
        {HtmlContext.item("Images searched",img_count)}
        {HtmlContext.item("Synonym matches",synmatch-1)}
        {HtmlContext.item("Synonym+Scene matches",scenematch-1)}
        {HtmlContext.item("Full matches",fullmatch)}
        {HtmlContext.item("Avg instances per image",instancesum / fullmatch)}
        </div>''')
        w(html_buffer)
        # w(f"""<script>
        # grid("{tclass.name}-examples")
        # </script>""")
        w("</div>")
        w("</div>")
