import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
from numbers import Number
import sys

import conf

def choice(l):
    for i,f in enumerate(l):
        print(f"[{i:2d}] {f}")
    try:
        ind = int(input("Choice: "))
        return l[ind]
    except BaseException:
        return None

def findclass(pixel):
    if hasattr(pixel, '__iter__'):
        for cl in conf.classes:
            if (cl.cv2color == pixel).all():
                return cl.id
        return None
    if isinstance(pixel,Number):
        if pixel < 0 or pixel >= len(conf.classes):
            print(f"Invalid index {pixel}")
            return None
        else: return conf.classes[pixel].id
    return None

############ GET FILENAMES
if len(sys.argv) > 1: root_folder = sys.argv[1]
else: root_folder = input("Root folder: ")
id_shift = 0
if len(sys.argv) > 2: id_shift = int(sys.argv[2])

file_endings = {".png",".jpg"}
print("Annotation:")
files = [f for f in os.listdir(root_folder) if os.path.splitext(f)[1].lower() in file_endings]
ann_name = choice(files)
if ann_name is None: exit()

print("Image:")
img_name = choice(files)
if img_name is None:
    img_name = ann_name

img_path = os.path.join(root_folder,img_name)
ann_path = os.path.join(root_folder,ann_name)


############ LOAD IMAGES
#img_pil = Image.open(img_path)
ann_pil = Image.open(ann_path)

# save annotation image with correct palette
if ann_pil.mode == "P":
    ann_pil.putpalette(conf.palette.flatten())
    ann_pil.save(ann_path)
else: print(f"Annotation image {ann_path} is not P-Mode, but {ann_pil.mode} mode")

ann = np.array(ann_pil,dtype=np.uint8) + id_shift
if img_path == ann_path:
    if ann_pil.mode == 'P': 
        img_shifted = Image.fromarray(ann,'P')
        img_shifted.putpalette(conf.palette.flatten())
        img_rgb = np.array(img_shifted.convert('RGB'))[:,:,[2,1,0]]
    else: img_rgb = ann[:,:,[2,1,0]]
else:
    img_rgb = cv2.imread(img_path)

# if ann_pil.mode == 'P':
#     bins = np.concatenate([np.arange(len(conf.classes)+1)])
#     ax = plt.subplot()
#     ax.hist(ann.flatten(),bins=bins)
#     ax.set_xticks(bins[:-1]+0.5)
#     ax.set_xticklabels([cl.name for cl in conf.classes],rotation="vertical")
#     plt.show()

############ PAINTING
sidebar_width = 220
vis_img = np.concatenate([img_rgb,np.zeros([img_rgb.shape[0],sidebar_width,3])],axis=1).astype(np.uint8)

region_img, regioncount = label(ann,return_num=True)
props = regionprops(region_img)
print("Found",regioncount,"objects")

objects_per_class = {cl.id : [] for cl in conf.classes}
for i,region in enumerate(props):
    x = region.centroid[0]
    y = region.centroid[1]
    cl = findclass(ann[int(x),int(y)])
    if cl is not None:
        objects_per_class[cl].append(region)
    else: print("none")
print("Found",len(objects_per_class),"different target classes.")

margin = 10
x = img_rgb.shape[1] + margin
fontsize = 10
i = 0
for class_i, region_list in objects_per_class.items():
    i += 1
    cl = conf.classes[class_i]
    y = (margin+fontsize)*i
    cv2.putText(vis_img, f"[{class_i:2}] {cl.name} x{len(region_list)}", [x,y], cv2.FONT_HERSHEY_SIMPLEX,0.5,cl.cv2color)
    #if class_i > 0:
    for j, region in enumerate(region_list):
        center = (int(region.centroid[1]),int(region.centroid[0]))
        cv2.circle(vis_img, center, int(region.equivalent_diameter), cl.cv2color)
        #cv2.circle(vis_img,center,3,cl.cv2color,thickness=-1)
        #cv2.putText(vis_img,str(j),(center[0]-10,center[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,0,0])
        cv2.line(vis_img,center,[x-3,y-5],cl.cv2color,1)

filename_no_ext = os.path.splitext(img_name)[0]
cv2.imwrite(os.path.join(root_folder,filename_no_ext+"_vis.png"), vis_img)
print(np.max(vis_img),np.min(vis_img),vis_img.dtype)
plt.imshow(vis_img.astype(int)[:,:,[2,1,0]])
plt.show()
