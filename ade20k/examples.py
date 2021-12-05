"""Find one example each for a given list of classes and save the img with outlines.
"""
import os

import cv2

import ade_utils as utils

ade_index = utils.adeindex.load()

out = os.path.expanduser("~/ssh_transfer/door_synonyms")
if not os.path.exists(out):
    os.makedirs(out)
    

classes = [
    "rocky wall",
    "stairs, steps",
    "stairway, staircase",
    "step, stair",
    "escalator, moving staircase, moving stairway",
    "staircase",
    
]

for cname in classes:
    class_id = utils.adeindex.class_index(ade_index,cname)
    img_id = next(utils.adeindex.images_with_class(ade_index,class_id))
    img, img_data = utils.adeindex.load_img(ade_index,img_id,True)
    img = utils.image.class_outlines(img,img_data,[[class_id,None,(255,0,0),4]])
    
    some_instance = next(utils.imgdata.objects_of_class(img_data,class_id))
    out_name = f"{cname}_{len(some_instance['parts']['hasparts'])}parts_{'with' if type(some_instance['parts']['ispartof']) == int else 'no'}Parent.jpg"
    cv2.imwrite(os.path.join(out,out_name),img)
    
    