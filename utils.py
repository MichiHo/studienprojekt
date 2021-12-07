"""Single import file for general utilities and the configuration, which is directly loaded by default from 'conf.json'.
If a different configuration file is required, just use conf = GeneralConfig(filename)."""
import json
import os
import shutil
from typing import List, Set

import numpy as np

project_root_folder = os.path.split(os.path.realpath(__file__))[0]

#######################################################
# Configuration

class TargetClass(object):
    """Description of a target class.
    
    -   name
    -   color: np.array of size 3 with RGB colors
    -   cv2color: 3-tuple with BGR colors for opencv
    -   id: integer id of the class (to be stored pixel-wise by the segmentor)
    -   scene: one of "indoor" "outdoor" "both"
    -   scenes: a set containing "indoor" and/or "outdoor"
    """
    def __init__(self,json_dict):
        self.name = json_dict['name']
        self.color = np.array(json_dict['color'],dtype=np.uint8)
        self.cv2color = (int(self.color[2]),int(self.color[1]),int(self.color[0]))
        self.id = json_dict['id']
        self.scene = json_dict['scene']
        self.scenes = {self.scene}
        if self.scene == "both":
            self.scenes = {"indoor","outdoor"}

class GeneralConfig(object):
    """Class for the general configuration, including classes (name, id, color, scenes) and 
    default folders (as absolute paths)."""
    classes : List[TargetClass] = []
    by_name = dict()
    
    def __init__(self,conf_path="conf.json"):
        
        
        with open(conf_path,"r") as ffile:
            data = json.load(ffile)
        self.classes = [TargetClass(c) for c in data['classes']]
        self.by_name = {cl.name: cl for cl in self.classes}
        self.palette = np.array([c.color for c in self.classes],dtype=np.uint8)
        self.train_palette = np.concatenate([
            np.array([[0,0,0]],dtype=np.uint8),
            self.palette],dtype=np.uint8).flatten()
        def path(p):
            return os.path.expanduser(os.path.join(project_root_folder,data[p]))
            
        self.annotate_filers_conf =    path('annotate_filers_conf')
        self.annotate_snippets_dir =   path('annotate_snippets_dir')
        self.dataset_out_path =        path('dataset_out_dir')
        
        self.ade_index_path =          path('ade_index_path')
        self.ade_dir =                 path('ade_dir')
        self.ade_classes_new_path =    path('ade_classes_new_path')
        
        self.labelme_dir =             path('labelme_dir')
        self.cmp_dir =                 path('cmp_dir')
        self.extension_datasets_val_every = data['extension_datasets_val_every']
            
        self.segmentation_logs_path =  path('segmentation_logs_dir')
        self.segmentation_plots_path = path('segmentation_plots_dir')
        self.segmentation_stats_path = path('segmentation_stats_dir')
        self.segmentation_model_path = path('segmentation_model_dir')
        self.segmentation_out_path =   path('segmentation_out_dir')

    def padded_palette(padding_length):
        """Return the color palette, padded with [0,0,0] entries to the given length. If padding_length
        is smaller than the palette, it will be cropped"""
        if padding_length < len(palette):
            print(f"Padded palette will be shorter than original palette! {padding_length} < {len(palette)}")
            return palette[:padding_length]
        
        return np.concatenate([palette,np.zeros((len(padding_length)-len(palette),3),dtype=np.uint8)])
try:   
    conf = GeneralConfig()
except FileNotFoundError:
    print ("Default conf file 'conf.json' missing. No configuration loaded.")
    conf = None


#######################################################
# Command line utils

console_colors = {
    "white" : '\033[0m',  # white (normal)
    "red" : '\033[31m', # red
    "green" : '\033[32m', # green
    "orange" : '\033[33m', # orange
    "blue" : '\033[34m', # blue
    "purple" : '\033[35m', # purple
    "gray" : '\033[90m' # Gray
}

def choice(l:List,indices:bool=False, text:str="Choice: ",displaylist:List[str]=None,grayout:Set=None):
    """Displays a given list in the command-line, each element with its index in the list,
    and awaits a choice by the user (an index). Returns the element, or its index if indices=True

    Args:
        l (list): The list to pick from
        indices (bool, optional): Whether to return the index in the list instead of the item itself. Defaults to False.
        text (str, optional): Text to show directly before the input. Defaults to "Choice: ".
        displaylist (list, optional): List of strings of same length as l. Defaults to None.

    Returns:
        An item from l or its index in l
    """
    if grayout is None:
        for i,f in enumerate(l):
            print(f"[{i:2d}] {f}")
    else:
        for i,f in enumerate(l):
            s=""
            if f in grayout: s = console_colors['gray']
            print(f"{s}[{i:2d}] {f}{console_colors['white']}")
    try:
        ind = int(input(text))
        if indices: return ind
        else: return l[ind]
    except BaseException:
        return None

def multichoice(l:List,indices:bool=False,text:str="Choice [leave empty to stop]: ",displaylist:List[str]=None):
    """Displays a given list in the command-line, each element with its index in the list,
    and awaits multiple choices by the user (given as indices). The process is stopped when the user 
    hits enter without anything typed. Returns a list of elements, or a list of indices if indices=True

    Args:
        l (List): The list to pick from
        indices (bool, optional):  Whether to return the index in the list instead of the item itself. Defaults to False.
        text (str, optional): Text to show directly before the input. Defaults to "Choice [leave empty to stop]: ".
        displaylist (List[str], optional): List of strings of same length as l. Defaults to None.

    Returns:
        A list of items from l or a list of their indices
    """
    if displaylist is None or len(displaylist) != len(l):
        displaylist = l
    for i,f in enumerate(displaylist):
        print(f"[{i:2d}] {f}")
    result = []
    while True:
        try:
            ind = int(input(text))
            if indices: result.append(ind)
            else: result.append(l[ind])
        except BaseException:
            break
    return result
    
def progress_bar(val, maxVal, minVal=0, length=10, full='#', empty='-', add_numbers = False):
    """Returns a simple progress-bar string

    Args:
        val (number): Current value
        maxVal (number): Max value
        minVal (number, optional): Current value. Defaults to 0.
        length (int, optional): Length of progress bar in characters. Defaults to 10.
        full (str, optional): Character for full bar. Defaults to '#'.
        empty (str, optional): Character for empty bar. Defaults to '-'.
        add_numbers (bool, optional): Whether to show {val} / {maxVal} before the progress bar. Defaults to False.

    Returns:
        str: Generated string
    """
    n = min(int(length*(val-minVal)/(maxVal-minVal)), length)
    if add_numbers:
        m = str(maxVal)
        return f"{val:^{len(m)}} / {m} " + full*n + empty*(length-n)
    else:
        return full*n + empty*(length-n)


#######################################################
# Data loading utils

def multi_root_json_parse(string:str):
    """Parse a json string with multiple root elements into a list of dicts. Accepts only
    root elements enclosed in curly brackets. If a json-file had multiple lists as root-elements
    it would not work.

    Args:
        string (str): JSON-string as loaded from a file.

    Raises:
        ValueError: If there are more closing brackets than opening brackets

    Returns:
        List[dict]: List of dicts created with json.loads
    """
    import re
    regex = re.compile("\{|\}")
    level = 0
    start = 0
    jsons = []
    for match in regex.finditer(string):
        if match.group() == "{": level+=1
        elif match.group() == "}":
            if level <= 0:
                raise ValueError("JSON bracket level conflict!! Too many }'s")
            level -= 1
            if level == 0:
                end = match.span()[1]
                jsons.append(json.loads(string[start:end]))
                start = end+1
    return jsons

def path_arg(path):
    """Runs expanduser on the path"""
    return os.path.expanduser(path)


#######################################################
# Specific other things

# for converting the first ADE20k-classnames to the old own class names
# used for old logfiles, before the own class names were added to mmsegmentation
ade_to_mydataset = {"environment" : "wall",
"wall_indoor" : "building",
"window" : "sky",
"door" : "floor",
"ceiling" : "tree",
"floor" : "ceiling",
"building" : "road",
"stairs" : "bed ",
"roof" : "windowpane",
"balcony" : "grass",
"air_conditioner" : "cabinet",
"chimney" : "sidewalk",
"column" : "person",
"sink" : "earth",
"toilet" : "door",
"bathtub" : "table",
"shower" : "mountain",
"outlet" : "plant",
"vents" : "curtain",
"fire_extinguisher" : "chair",
"oven" : "car",
"radiator" : "water",
"railing" : "painting",
"fire_escape" : "sofa"}

def nice_config_title(this_info):
    """Takes the first element of a MMSegmentation training log and derives a hopefully nice
    descriptive name for it.  It first looks, whether the field "studienprojekt_mode" has been
    set, which would contain a custom title. If not, it tries to crop common parts of the title.

    Args:
        info_json (dict): The first element from a training log, as parsed json
    """
    title_prefix = "upernet_swin_base_patch4_window12_512x512_"
    title_suffix = "_pretrain_384x384_22K.py"
    
    if "studienprojekt_mode" in this_info:
        return this_info['studienprojekt_mode']
    elif this_info['exp_name'].startswith(title_prefix) and this_info['exp_name'].endswith(title_suffix):
        return this_info['exp_name'][len(title_prefix):-1*len(title_suffix)]
    
    return this_info['exp_name']

def prepare_dataset_extension_dirs():
    """Creates the folders for the extended dataset by copying the outdoor and inout folders into
    outdoor_extended and inout_extended, if they do not exist already."""
    for loc in ["inout","outdoor"]:
        in_path = os.path.join(conf.dataset_out_path,loc)
        out_path = os.path.join(conf.dataset_out_path,loc+"_extended")
        if not os.path.exist(out_path):
            print(f"Generate {out_path} by copying {in_path}")
            shutil.copytree(in_path, out_path)