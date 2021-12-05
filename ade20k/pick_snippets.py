import shutil
import os
import random
from tqdm import tqdm

in_folder = "mmseg/data/studienprojekt/inout_extended/"
ann_folder = os.path.join(in_folder,"annotations/train")
img_folder = os.path.join(in_folder,"images/train")
out_folder = os.path.expanduser("~/ssh_transfer/snippets")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

filenames = os.listdir(img_folder)
random.shuffle(filenames)

for img_name in tqdm(filenames[:20]):
    ann_name = img_name[:-4] + ".png"
    shutil.copy(os.path.join(img_folder,img_name), os.path.join(out_folder,img_name))
    shutil.copy(os.path.join(ann_folder,ann_name), os.path.join(out_folder,ann_name))