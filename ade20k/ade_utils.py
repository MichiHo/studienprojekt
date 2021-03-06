"""Single-import file containing utilities for exploration and re-annotation of the ADE20k dataset,
including the configuration of filters. The contents are:

-   AdeTargetClass : Description of a target class in terms of how it is derived from ADE20k
-   configuration : Description of the full conf, including a list of AdeTargetClasses
-   image (statis): Utilities for image manipulation
-   adeindex  (static): Utilities for querying the index pkl file
-   imgdata (static): Utilities for querying the json of a single image
-   ade_stats (static): Utilities for querying the new stats file generated from ADE20k
-   plots (static): Plotting utils
-   html : A ContextManager class to open and write a html summary
"""

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
                    synonyms[AdeIndex.class_index(ade_index, syn)] = {}
                elif type(syn) == dict:
                    synonyms[AdeIndex.class_index(ade_index, syn['name'])] = {AdeIndex.class_index(
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
                parent = ImgData.find_obj_by_id(img_data,obj['parts']['ispartof'])
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
    def load(ade_index : dict, filepath : str = general_conf.annotate_filers_conf):
        with open(filepath, "r") as f:
            data = json.load(f)
            
        target_classes = [
            AdeTargetClass.from_json(ade_index, class_json)
            for class_json in data['classes']]
        target_classes = [cl for cl in target_classes if cl is not None]
            
        target_classes = {cl.id : cl for cl in target_classes}
        for conf_class in general_conf.classes:
            if not conf_class.id in target_classes: 
                raise ValueError(f"Target class {conf_class.name} missing in ade-specific json conf!")
            
        return AdeConfiguration(target_classes, data['detection_threshold'])

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


class Images(object):
    """Utils for dealing with image data."""
    @staticmethod
    def instance_outline(img, instance_data, color, thickness=None, lineType=None, shift=None, point_scaling=1.0):
        """Draws the outline of a single object instance on top of the image.

        Args:
            img (cv2 image): The image, as loaded by cv2
            instance_data (dict): Data for an object instance from an annotation json
            color (tuple): Color in BGR for the outline
            thickness (int, optional): Integer thickness of the outline. Defaults to None.
            lineType (str, optional): lineType argument for cv2.polylines. Defaults to None.
            shift (optional): shift argument for cv2.polylines. Defaults to None.
            point_scaling (float, optional): Factor to scale the object's outline polygon with. Does not apply to the image. Defaults to 1.0.

        Returns:
            cv2 image: img with outline on top
        """
        points = np.array([
            [instance_data['polygon']['x'][i], instance_data['polygon']['y'][i]]
            for i in range(len(instance_data['polygon']['x']))]) * point_scaling
        points = points.reshape((-1, 1, 2)).astype(np.int32)
        return cv2.polylines(img, [points], True, color, thickness=thickness, lineType=lineType, shift=shift)

    @staticmethod
    def class_outlines(img, img_data, classes_colors, legend=True, add_info=False, highlight_instances=[],scaling=1.0):
        """Create image with outlines of class instances of given classes. The given classes are painted
        in that order. The color is in RGB, image is expected as loaded from cv2 (with BGR, conversion
        is done here)

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
        
        #size = (int(img.shape[0] * scaling),int(img.shape[1]*scaling))
        img = cv2.resize(img,[0,0],fx=scaling,fy=scaling,interpolation=cv2.INTER_AREA)

        for inst in highlight_instances:
            col = (inst['color'][2], inst['color'][1], inst['color'][0])
            img = Images.instance_outline(img, ImgData.find_obj_by_id(
                img_data, inst['id']), col, inst['thickness'],point_scaling=scaling)

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
                    parent = ImgData.find_obj_by_id(img_data, parent_id)
                    if parent['name'] != partof:
                        continue
                count += 1
                img = Images.instance_outline(img, o, col, thickness,point_scaling=scaling)

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
            img_data = ImgData.load(folder,filename)
        
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
        masks_folder = os.path.join(project_root_folder,folder,filename)
        
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
 
  
class AdeIndex(object):
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
    def img_path(ade_index, img_index):
        return os.path.join(project_root_folder,ade_index['folder'][img_index],ade_index['filename'][img_index])

    @staticmethod
    def img_folder(ade_index, img_index):
        return os.path.join(project_root_folder,ade_index['folder'][img_index],ade_index['filename'][img_index][:-4])
        
    @staticmethod
    def load_img(ade_index, img_index, load_imgdata=False, load_training_image=True, pillow=False):
        """Loads, for the given index, the training image from .jpg and/or the imgdata from .json

        Args:
            ade_index (dict): Index dict
            img_index (int): Index of the image
            load_imgdata (bool, optional): Whether to load the annotations. Defaults to False.
            load_training_image (bool, optional): Whether to load the annotations. Defaults to True.

        Returns:
            cv2 image or (image, dict) or dict: image and/or annotations.
        """

        filename = ade_index['filename'][img_index][:-4]
        foldername = ade_index['folder'][img_index]
        if load_training_image:
            if pillow:
                img = Image.open(AdeIndex.img_path(ade_index,img_index))
            else:
                img = cv2.imread(AdeIndex.img_path(ade_index,img_index))
        if load_imgdata:
            img_data = ImgData.load(foldername, filename)
            if load_training_image: return (img, img_data)
            else: return img_data
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
    def class_guess(ade_index,guess):
        return list(filter(lambda name: guess.lower() in name,ade_index['objectnames']))

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
    def images_with_class(ade_index, class_id, count=num_images, start_index=0, random=False):
        """Iterator over all image IDs of images containing the given class

        Args:
            ade_index (dict): Index dict
            class_id (int): ID of the class to search for
            count (int, optional): Number of images to yield at max. Defaults to num_images.
            start_index (int, optional): Index to start the search from. Defaults to 0.
            random (bool): Whether to randomize the order of images. Default: False

        Yields:
            int: ID of images containing the given class.
        """
        found = 0
        if random: permutation = np.random.permutation(num_images)
        for i in range(start_index, num_images):
            if random: i2 = permutation[i]
            else: i2 = i
            
            if ade_index['objectPresence'][class_id, i2] > 0:
                if found >= count:
                    return
                found += 1
                yield i2

    @staticmethod
    def images_with_classes(ade_index, class_ids, count=num_images, start_index=0, random=False):
        """Iterator over all image IDs of images containing all given classes

        Args:
            ade_index (dict): Index dict
            class_id (List[int]): IDs of the classes to search for
            count (int, optional): Number of images to yield at max. Defaults to num_images.
            start_index (int, optional): Index to start the search from. Defaults to 0.
            random (bool): Whether to randomize the order of images. Default: False

        Yields:
            int: ID of images containing all of the given classes.
        """
        if count == None: count = num_images
        found = 0
        if random: permutation = np.random.permutation(num_images)
        for i in range(start_index, num_images):
            if random: i2 = permutation[i]
            else: i2 = i
            if AdeIndex.matches_all_classes(ade_index, class_ids, i2):
                if found >= count:
                    return
                found += 1
                yield i2
    
    @staticmethod
    def any_images(count=num_images,random=False):
        """Iterate over all image IDs, optionally randomized

        Args:
            count (int, optional): How many images to return. Defaults to all of them.
            random (bool, optional): Whether to randomize the order. Defaults to False.

        Returns:
            int iterator over all IDs
        """
        if random: 
            iterr = np.random.permutation(num_images)
            
        for i in range(count):
            if random: yield iterr[i]
            else: 
                yield i
        
    
    @staticmethod
    def classname(ade_index,index):
        """Lookup the classname for the given class index in the ade_index object

        Args:
            ade_index (dict): ADE20k index file data
            index (int): Index of the class. If < 0, "NONE" is returned

        Returns:
            str: Classname, or "NONE" if index < 0 
        
        Raises:
            ValueError if the index is out of bounds
        """
        return ade_index['objectnames'][index] if index >= 0 else "NONE"
    
    @staticmethod
    def classnames(ade_index,ids):
        """Iterator over the classnames for the given class indices, from the ade_index object

        Args:
            ade_index (dict): ADE20k index file data
            ids (List[int]): Class indices. Index < 0 translates to to "NONE".

        Yields:
            str: Classname, or "NONE" if index < 0, for each index in ids
        
        Raises:
            ValueError if an index is out of bounds
        """
        for i in ids:
            yield AdeIndex.classname(ade_index,i)


imgdata_encs = {}
class ImgData(object):
    """Methods for handling the annotations json of a single image"""
    @staticmethod
    def loadi(ade_index : dict,img_index : int):
        return ImgData.load(ade_index['folder'][img_index],
            os.path.splitext(ade_index['filename'][img_index])[0])
    
    @staticmethod
    def load(folder : str, name : str):
        """Load the annotations json for the given image

        Args:
            folder (str): Folder name, relative to project_root
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
          
           

class AdeStats(object):
    """Methods for loading and dealing with the new ade_stats.pkl file, generated from the
    per-image json files.

    -   classes: contains for each class (indexed by integer class-id):
        -   name
        -   scenes: scenes it is present in, dict with val=occurrence
        -   parents: a dict mapping class-ids (or -1 for NONE) to the number this parent-class is assumed
        -   object_count: total number of class instances in the dataset
        -   image_count: number of images containing this class
    -   scenes: dict of scene-names mapped to image count. only the first element of each images 'scene'
        attribute is used, which mostly contains either 'indoor' or 'outdoor'
    """
    
    path = ""

    @staticmethod
    def load():
        with open(general_conf.ade_stats_path, "rb") as pkl_file:
            classes = pickle.load(pkl_file)

        return classes


class Plots(object):

    @staticmethod
    def parent_stats(ade_index, ade_stats : dict, class_id : int, save_path : str):
        def trim(st):
            l = 20
            if len(st) > l+2:
                return st[:l] + ".."
            return st
            
        fig, ax = plt.subplots(1,2,figsize=[14,6],gridspec_kw={'width_ratios': [3, 1]})
        
        parents = ade_stats['classes'][class_id]['parents']
        parents = sorted(parents.items(),key=lambda item: item[1],reverse=True)
        parents = dict(parents)
        
        ticklabels = ["NONE" if idd == -1 else trim(AdeIndex.classname(ade_index, idd))
                for idd in parents.keys()]
        ticks = np.arange(len(ticklabels))
        ax[0].set_title("Parent occurrence for "+AdeIndex.classname(ade_index, class_id))
        ax[0].bar(ticks,
            parents.values())
        ax[0].set_xticks(ticks)
        ax[0].set_xticklabels(ticklabels,rotation="vertical",fontsize=7)
        
        scenes = ade_stats['classes'][class_id]['scenes']
        ticklabels = scenes.keys()
        ticks = np.arange(len(ticklabels))
        ax[1].set_title("Scene occurrence for "+AdeIndex.classname(ade_index, class_id))
        ax[1].bar(ticks,
            scenes.values())
        ax[1].set_xticks(ticks)
        ax[1].set_xticklabels(ticklabels,rotation="vertical")
        
        plt.tight_layout()
        plt.savefig(save_path)

