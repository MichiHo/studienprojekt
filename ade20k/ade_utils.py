"""Single-import file containing utilities for exploration and re-annotation of the ADE20k dataset,
including the configuration of filters. The contents are:

-   AdeTargetClass : Description of a target class in terms of how it is derived from ADE20k
-   configuration : Description of the full conf, including a list of AdeTargetClasses
-   image (statis): Utilities for image manipulation
-   adeindex  (static): Utilities for querying the index pkl file
-   imgdata (static): Utilities for querying the json of a single image
-   classes_new (static): Utilities for querying the new stats file generated from ADE20k
-   plots (static): Plotting utils
-   html : A ContextManager class to open and write a html summary
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

import chardet
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import conf as general_conf, project_root_folder

num_images = 27574
num_classes = 3688
     
class AdeTargetClass(object):
    """Class for segmentation target classes, describing how to derive them from ADE20k classes

    The specification can be loaded from json with the following schema:
    - name : str naming the target class
    - synonyms : list of synonym definitions or str "remains"
                 "remains" defines the class for all pixels not covered by 
                 all other classes, optionally with scene constraint
    - scene (optional) : str in ["indoor", "outdoor"] defining a constraint
    - z_index (optional) : in case of overlapping regions in ADE20k, 
                           they are sorted accordingly. Default = 0
    - color (optional) : list of three ints 0-255 RGB

    A synonym definition can be either just a string with the ADE20k-name or
    - name : str naming the ADE20k class
    - parents : list of str naming the ADE20k class of allowed parent classes, can also include "NONE"

    All ADE20k class names are converted to the corresponding int indices on load.
    """
    # further ideas: attributes from the ann-jsons, scenes per synonym

    @staticmethod
    def from_json(ade_index : dict, json : dict):
        """Generate a single target_class from the corresponding dict loaded from json

        Args:
            ade_index (dict): ADE20k pkl index used for finding ADE-class IDs
            json (dict): dict corresponding to a single class, as loaded from json

        Returns:
            instance of target_class
        """
        name = json['name'].replace(" ","_")
        # Find corresponding global target_class
        global_class = [cl for cl in general_conf.classes if cl.name == name]
        if len(global_class) != 1: 
            print(f"ADE-target_class {name} has no match in global classes and will be ignored")
            return
        global_class = global_class[0]
        
        synonyms = {}
        # convert class names to indices and sets where possible
        if type(json['synonyms']) == list:
            for syn in json['synonyms']:
                if type(syn) == str:
                    synonyms[adeindex.class_index(ade_index, syn)] = {}
                elif type(syn) == dict:
                    synonyms[adeindex.class_index(ade_index, syn['name'])] = {adeindex.class_index(
                        ade_index, par) for par in syn['parents']}
                else:
                    raise AttributeError()
        elif json['synonyms'] == "remains":
            synonyms = "remains"
        else:
            raise AttributeError()
        for prop in ['color','scene']:
            if prop in json: print(f"{prop} of ADE-target_class {name} will be ignored.")
        # color = None
        # if 'color' in json:
        #     if type(json['color']) == str:
        #         color = [int(json['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
        #     else:
        #         color = [int(c) for c in json['color']]

        return AdeTargetClass(
            name,
            synonyms,
            global_class.color,
            None if global_class.scene=="both" else global_class.scene,
            int(json['z_index']) if 'z_index' in json else 0,
            global_class.id
        )

    # class synonym(object):
    #     def __init__(self, class_id: int, parent_ids: List[int] = None):
    #         self.class_id = class_id
    #         self.parent_ids = parent_ids
    
    def __init__(self,
                 name: str,
                 synonyms: dict,
                 color: List[int] = [0,0,0],
                 scene: str = None,
                 z_index: int = 0,
                 class_id: int = -1):

        self.name = name
        self.synonyms = synonyms
        self.color = np.array(color if color is not None else [0,0,0])
        self.cv2color = np.array([self.color[2],self.color[1],self.color[0]])
        self.scene = scene
        self.z_index = z_index
        self.id = class_id
        self.mask = np.zeros((num_classes))
        if self.synonyms != "remains": 
            for sy in self.synonyms.keys():
                self.mask[sy] = 1
                
    def syn_match(self,ade_index,img_index):
        """Return the number of matched synonyms in the given image. 
        Use as boolean to find matching images.

        Args:
            ade_index (dict): ADE20k index
            img_index (int): Index of image

        Returns:
            int: Count of matched synonyms. 
        """
        return np.sum(np.logical_and(self.mask,ade_index['objectPresence'][:,img_index]))
    
    def scene_match(self,img_data : dict) -> bool:
        return not self.scene or img_data['scene'][0] == self.scene
        
    def full_match(self,img_data : dict) -> List[dict]:
        """Find all object instances matching the target class fully.

        Args:
            img_data (dict): Data loaded from annotations json of one image

        Returns:
            List[dict]: List of object descriptions from img_data. Empty list if none found.
        """
        
        result = []
        for obj in img_data['object']:
            parents = self.synonyms.get(obj['name_ndx'])
            # if the obj class is not one of the synonyms, parents is None
            if parents is None: 
                continue
            
            #any parent accepted for this synonym
            if len(parents) == 0: 
                result.append(obj)
            #has no parent and this is accepted
            elif obj['parts']['ispartof'] == []:
                if -1 in parents: 
                    result.append(obj)
            #check parent class
            else: 
               #print(" has parent",end="")
                parent = imgdata.find_obj_by_id(img_data,obj['parts']['ispartof'])
                if parent['name_ndx'] in parents: 
                    #print(" and its good!")
                    result.append(obj)
        
        return result
            

class AdeConfiguration(object):
    """Configuration class to manage a set of target_class'es. Splits all input target_class objects
    into two lists:
    
        target_classes : contains all except the 'remains' classes, sorted by z-index.
        remains_classes : one 'remains' class per scene, indexed by scene name (str)
    
    Also creates a global class-mask of length num_classes with a 1 at each ADE20k-class present in 
    at least one target_class.
    """
    @staticmethod
    def load(ade_index : dict, filepath : str = general_conf.annotate_filers_conf) -> AdeConfiguration:
        with open(filepath, "r") as f:
            json = json.load(f)
            
        target_classes = [
            AdeTargetClass.from_json(ade_index, class_json)
            for class_json in json['classes']]
        target_classes = [cl for cl in target_classes if cl is not None]
            
        target_classes = {cl.id : cl for cl in target_classes}
        for conf_class in general_conf.classes:
            if not conf_class.id in target_classes: 
                raise ValueError(f"Target class {conf_class.name} missing in ade-specific json conf!")
            
        return AdeConfiguration(target_classes, json['detection_threshold'])

    def __init__(self, target_classes: Dict[int,AdeTargetClass], detection_thres: int):
        # create mask
        self.content_classes : List[AdeTargetClass] = []
        self.remains_classes : Dict(str,AdeTargetClass) = {} 
        self.all_classes : Dict(int,AdeTargetClass) = target_classes
        self.detection_thres = detection_thres
        
        max_index = max(target_classes.keys())
        self.palette = np.zeros((max_index+1)*3)
        mask = np.zeros((num_classes))
        for i,cl in target_classes.items():
            self.palette[i*3:(i+1)*3] = cl.color
            if cl.synonyms == "remains": 
                if cl.scene is None:
                    if len(self.remains_classes) > 0:
                        print("!!! Only one 'remains' class per scene possible. Found global 'remains' class after already registering one for",self.remains_classes.keys())
                    else:
                        self.remains_classes['indoor'] = cl
                        self.remains_classes['outdoor'] = cl
                else:
                    if not cl.scene in self.remains_classes:
                        self.remains_classes[cl.scene] = cl
                    else: print("!!! Only one 'remains' class per scene possible. Duplicate for",cl.scene,"found !!!")
                
                continue
            mask = np.logical_and(mask,cl.mask)
            self.content_classes.append(cl)
        self.content_classes.sort(key=lambda cl: cl.z_index)
        self.class_mask = mask
        
    def syn_match(self,ade_index,img_index,classes=False):
        """Return the number of matched target classes in the given image

        Args:
            ade_index (dict): ADE20k index
            img_index (int): Index of image

        Returns:
            int: Count of matched / 'activated' target classes. 
        """
        if classes:
            cll = []
            for cl in self.content_classes:
                if cl.syn_match(ade_index,img_index) > 0:
                    cll.append(cl)
            return cll
        else:
            return np.sum([
                cl.syn_match(ade_index,img_index) > 0 
                for cl in self.content_classes])


class image(object):
    """Utils for dealing with image data."""
    @staticmethod
    def instance_outline(img, instance_data, color, thickness=None, lineType=None, shift=None):
        points = np.array([
            [instance_data['polygon']['x'][i], instance_data['polygon']['y'][i]]
            for i in range(len(instance_data['polygon']['x']))], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], True, color,
                      thickness=thickness, lineType=lineType, shift=shift)
        return img

    @staticmethod
    def class_outlines(img, img_data, classes_colors, legend=True, add_info=False, highlight_instances=[]):
        """Create image with outlines of class instances of given classes

        Args:
            in_folder (str): Folder containing the jpg and json file
            in_name (str): Name (without extension)
            classes_colors (list): (class: int, parent: int, color: 3-tuple, thickness: int) uses class indices
            legend (bool): Whether to paint a legend of classes_colors on the image
            add_info (bool): Whether to return additional info
            highlight_instances (List[dict]): List of instances to highlight. Each item needs attributes
            'id' 'color' and 'thickness'.

        Returns (add_info=False):
            cv2 image: Photograph with outlines

        Returns (add_info=True):
            cv2 image: Photograph with outlines
            dict: Image annotation data
            dict: Counts of each object class
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        legend_y = img.shape[0]-2
        counts = {}

        for inst in highlight_instances:
            col = (inst['color'][2], inst['color'][1], inst['color'][0])
            image.instance_outline(img, imgdata.find_obj_by_id(
                img_data, inst['id']), col, inst['thickness'])

        for classname, partof, col, thickness in classes_colors:
            # bgr <-> rgb
            col = (col[2], col[1], col[0])
            count = 0
            # Find instances matching classname and optionally the parent's class
            for o in img_data['object']:
                if o['name_ndx'] != classname:
                    continue
                if partof and type(o['parts']['ispartof']) == int:
                    parent_id = o['parts']['ispartof']
                    parent = imgdata.find_obj_by_id(img_data, parent_id)
                    if parent['name'] != partof:
                        continue
                count += 1
                image.instance_outline(img, o, col, thickness)

            if not classname in counts:
                counts[classname] = count
            else:
                counts[classname] = max(count, counts[classname])

            if legend:
                text = f"{count:2}x #{classname}"
                if partof:
                    text = text + " from " + partof
                pos = (0, legend_y)
                cv2.putText(img, text, pos, font, 0.5,
                            (255, 255, 255, 0.5), 6, cv2.LINE_AA)
                cv2.putText(img, text, pos, font, 0.5, col, 1, cv2.LINE_AA)
                legend_y -= 17
        if add_info:
            return (img, counts)
        else:
            return img
    
    @staticmethod
    def annotate(conf : AdeConfiguration, filename : str, folder : str, img_data : dict = None,
     detection_thres : int = 2, stats = False,color = False, skip_zero_index = True):
        """Find matches of target classes for the given image and return a new segmentation image
        with the colors from conf. Returns None, if no matches were found (can be used to do the 
        matching as well). If color = False, the pixels are the class indices, incremented by one if 
        skip_zero_index=True, for omitting the zero index for the reduce_zero_label setting of 
        MMSegmentation.

        Args:
            conf (configuration): Configuration
            filename (str): Filename (no extension)
            folder (str): Folder path
            img_data (dict, optional): Img data. If None, it is loaded.
            detection_thres (int): Number of minimum required matched classes to not return None
            stats (bool): Whether to also return the list of matches
            color (bool): Whether to fill the pixels with the colors instead of the class indices
            skip_zero_index (bool): Whether to increment class indices by one for color=False

        Returns:
            None, if the number of found matches is below the detection threshold
            cv2 image if matches were found and stats=False
            image, matches if stats=True
        """
        
        if img_data is None:
            img_data = imgdata.load(folder,filename)
        
        # Look for all matches first. If none, abort.
        matches = []
        classes = 0
        for t_class in conf.content_classes:
            if not t_class.scene_match(img_data): 
                matches.append([])
            else:
                match = t_class.full_match(img_data)
                matches.append(match)
                if len(match) > 0: classes += 1
        
        if classes < detection_thres: return None
        
        img_size = img_data['imsize'][:2]
        dtype = np.uint8
        masks_folder = os.path.join(folder,filename)
        
        # Load remains class for scene of image and init image with it.
        remains_class = conf.remains_classes[img_data['scene'][0]]
        if color:
            img = np.zeros([img_size[0],img_size[1],3], dtype=dtype)
            img[:] = remains_class.cv2color
        else:
            img = np.zeros([img_size[0],img_size[1]], dtype=dtype)
            img[:] = remains_class.id+1
        
        # Draw all matches on top
        zero_mask = np.zeros(img_size,dtype=dtype)
        for i, t_class in enumerate(conf.content_classes):
            match = matches[i]
            
            # Combine all masks of matched objects
            mask = zero_mask
            for obj in match:
                obj_id = obj['id']
                newmask = cv2.imread(
                        os.path.join(masks_folder,f"instance_{obj_id:03}_{filename}.png"),
                        cv2.IMREAD_GRAYSCALE
                    )
                mask = cv2.bitwise_or(mask,newmask)
            img[mask > 0] = t_class.cv2color if color else t_class.id+1
        if stats: return img, matches
        else: return img
 
  
class adeindex(object):
    """Methods for loading and handling the index pkl file

    - filename: 
      array of length N=27574 with the image file names

    - folder: 
      array of length N with the image folder names.

    - scene: 
      array of length N providing the scene name (same classes as the Places database) for each image.

    - objectIsPart: 
      array of size [C, N] counting how many times an object is a part in each image. 
      objectIsPart[c,i]=m if in image i object class c is a part of another object m times. For objects, objectIsPart[c,i]=0, and for parts we will find  objectIsPart[c,i] = objectPresence(c,i)

    - objectPresence: 
      array of size [C, N] with the object counts per image. objectPresence(c,i)=n if in image i there are n instances of object class c.

    - objectcounts: 
      array of length C with the number of instances for each object class.

    - objectnames: 
      array of length C with the object class names.

    - proportionClassIsPart: 
      array of length C with the proportion of times that class c behaves as a part. If proportionClassIsPart[c]=0 then it means that this is a main object (e.g., car, chair, ...). See bellow for a discussion on the utility of this variable.

    - wordnet_found: 
      array of length C. It indicates if the objectname was found in Wordnet.

    - wordnet_level1: 
      list of length C. WordNet associated.

    - wordnet_synset: 
      list of length C. WordNet synset for each object name. Shows the full hierarchy separated by .

    - wordnet_hypernym: 
      list of length C. WordNet hypernyms for each object name.

    - wordnet_gloss: 
      list of length C. WordNet definition.

    - wordnet_synonyms: 
      list of length C. Synonyms for the WordNet definition.

    - wordnet_frequency: 
      array of length C. How many times each wordnet appears

    """

    @staticmethod
    def load():
        """Load the index file from the default location.

        Returns:
            object: The loaded ADE20k index data
        """
        print("Loading ade_index... ",end="")
        data_file = open(general_conf.ade_index_path, "rb")
        ade_index = pickle.load(data_file)
        data_file.close()
        print("done.")
        return ade_index

    @staticmethod
    def load_img(ade_index, img_index, load_imgdata=False):
        """Load the image with the given index

        Args:
            ade_index (dict): Index dict
            img_index (int): Index of the image
            load_imgdata (bool, optional): Whether to also load the annotations. Defaults to False.

        Returns:
            cv2 image or (image, dict): image and optionally the annotations.
        """

        filename = ade_index['filename'][img_index][:-4]
        foldername = ade_index['folder'][img_index]
        img = cv2.imread(os.path.join(project_root_folder,foldername, filename+".jpg"))
        if load_imgdata:
            img_data = imgdata.load(foldername, filename)
            return (img, img_data)
        else:
            return img

    @staticmethod
    def classes_containing(ade_index, text):
        """Return all indices of classes whose name contains the given text

        Args:
            ade_index (dict): Data dict read from ADE20k pickle file
            text (str): Text to search for
        """

        result = []
        for i, name in enumerate(ade_index['objectnames']):
            if text in name:
                result.append(i)

        return result

    @staticmethod
    def class_index(ade_index, name):
        """Find index of class of exactly this name

        Args:
            data (dict): Data dict
            name (str): Class name

        Returns:
            int: Index of class. 

        Throws:
            ValueError if not present.
        """
        if name == "NONE":
            return -1
        else:
            return ade_index['objectnames'].index(name)

    @staticmethod
    def matches_one_class(ade_index, classes, img_index):
        """Check, if the image of the given index matches one of the classes.

        Args:
            ade_index (dict): Data dict
            classes (list): List of class indices
            img_index (int): Index of image

        Returns:
            boolean: True, iff the image contains objects of one of the classes
        """

        for c in classes:
            if ade_index['objectPresence'][c, img_index] > 0:
                return True
        return False

    @staticmethod
    def matches_all_classes(ade_index, classes, img_index):
        """Check, if the image of the given index matches all of the classes.

        Args:
            ade_index (dict): Data dict
            classes (list): List of class indices
            img_index (int): Index of image

        Returns:
            boolean: True, iff the image contains objects of one of the classes
        """

        for c in classes:
            if ade_index['objectPresence'][c, img_index] == 0:
                return False
        return True

    @staticmethod
    def images_with_class(ade_index, class_id, count=num_images, start_index=0):
        """Iterator over all image IDs of images containing the given class

        Args:
            ade_index (dict): Index dict
            class_id (int): ID of the class to search for
            count (int, optional): Number of images to yield at max. Defaults to num_images.
            start_index (int, optional): Index to start the search from. Defaults to 0.

        Yields:
            int: ID of images containing the given class.
        """
        found = 0
        for i in range(start_index, num_images):
            if ade_index['objectPresence'][class_id, i] > 0:
                if found >= count:
                    return
                found += 1
                yield i

    @staticmethod
    def images_with_classes(ade_index, class_ids, count, start_index=0):
        found = 0
        for i in range(start_index, num_images):
            if adeindex.matches_all_classes(ade_index, class_ids, i):
                if found >= count:
                    return
                found += 1
                yield i

    @staticmethod
    def classname(ade_index,index):
        return ade_index['objectnames'][index] if index >= 0 else "NONE"
    
    @staticmethod
    def classnames(ade_index,ids):
        for i in ids:
            yield adeindex.classname(ade_index,i)


imgdata_encs = {}
class imgdata(object):
    """Methods for handling the annotations json of a single image"""
    @staticmethod
    def loadi(ade_index : dict,img_index : int):
        return imgdata.load(ade_index['folder'][img_index],
            os.path.splitext(ade_index['filename'][img_index])[0])
    
    @staticmethod
    def load(folder : str, name : str):
        """Load the annotations json for the given image

        Args:
            folder (str): Folder name
            name (str): File name (without extension!)

        Returns:
            dict: Image annotation data.
        """
        path = os.path.join(project_root_folder,folder, name + ".json")
        with open(path, "rb") as rawfile:
            raw = rawfile.read()
            encoding = chardet.detect(raw)['encoding']
            if not encoding in imgdata_encs:
                imgdata_encs[encoding] = 0
            imgdata_encs[encoding] += 1
            data = json.loads(raw.decode(encoding))['annotation']
            for obj in data['object']:
                obj['name_ndx'] = obj['name_ndx'] - 1
        return data

    @staticmethod
    def find_obj_by_id(img_data, index):
        """Find object instance in an image annotation by integer index

        For some reason, the index in the 'object' list and the corresponding entry's 'id' field
        do not always match. This function detects such a mismatch and makes sure, that the object with 
        the index is really found

        Args:
            img_data (dict): Data loaded from an image's annotation JSON.
            index (int): Integer index of the wanted object instance.

        Returns:
            dict: The Object instance, an element from the 'object' list in img_data.
        """

        if index < len(img_data['object']) and img_data['object'][index]['id'] == id:
            return img_data['object'][index]
        for i, obj in enumerate(img_data['object']):
            if obj['id'] == index:
                return obj
        print(f"!!! no object with id {index}")
        return None

    @staticmethod
    def objects_of_class(img_data, class_id):
        """Iterator over all objects of given class in image

        Args:
            img_data (dict): Image annotation data
            class_id (int): ID of the class

        Yields:
            dict: Data of the object instance
        """
        #print(f"type {type(img_data['object'][0]['name_ndx'])} bla {type(class_id)}")
        for obj in img_data['object']:
            if obj['name_ndx'] == class_id:
                yield obj
          
           

class classes_new(object):
    """Methods for loading and dealing with the new classes_new.pkl file, generated from the
    per-image json files.

    The loaded object has integer keys for the class-ids and contains:
    - 'parents' : a dict mapping class-ids (or -1 for NONE) to the number this parent-class is assumed
    - 'object_count' : number of object instances found 
    - 'name' : name of the class 
    """
    
    path = ""

    @staticmethod
    def load():
        with open("classes_new.pkl", "rb") as pkl_file:
            classes = pickle.load(pkl_file)

        return classes


class plots(object):

    @staticmethod
    def parent_stats(ade_index, classes : dict, class_id : int, save_path : str):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(
            ["NONE" if idd == -1 else ade_index['objectname'][idd]
                for idd in classes[class_id]['parents'].keys()],
            classes[class_id]['parents'].values())
        plt.savefig(save_path)


class html(object):
    """ContextManager class opening a file and writing html to it. Automatically creates the
    header and opens and closes the <body> tag. The returned object is callable with a str param
    which will be appended inside the <body> tag.
    """

    def __enter__(self):
        folder = os.path.dirname(self.filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.f = open(self.filepath, "w")

        self.__call__(f'''<!doctype html><html>
        <head>
        <meta charset="utf-8">
        <title>{self.title}</title>
        <link href="../style.css" rel="stylesheet">
        <script src="../masonry.pkgd.min.js"></script>
        </head>
        <body>''')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__call__("</body></html>")
        self.f.close()

    def __init__(self, filepath : str, title : str = "ADE20k"):
        """Construct

        Args:
            filepath (str): Path to the file to open and write to.
            title (str, optional): Title of the page. Defaults to "ADE20k".
        """
        self.filepath = filepath
        self.title = title

    def __call__(self, string : str):
        """Append text inside the <body> tag

        Args:
            string (str): Text to insert
        """
        self.f.write(string)
    
    @staticmethod
    def color(cl):
        """Returns the given color codes as css rgb-string

        Args:
            cl (List[int]): List of R,G,B ints (0-255)

        Returns:
            str: rgb(r,g,b) like string
        """
        return f"rgb({cl[0]},{cl[1]},{cl[2]})"
        
    @staticmethod
    def item(title,val):
        return f"<p class='item{' numeric-item' if isinstance(val,(int,float,complex)) else ''}'><span class='title'>{title}</span><span class='val'>{val}</span></p>"
