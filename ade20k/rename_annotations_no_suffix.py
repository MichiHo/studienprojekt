import os

import ade_utils as utils

mmseg_folder = "/glusterfs/dfs-gfs-dist/hochmumi/mmseg"
if not os.path.exists(mmseg_folder):
    print("MMSeg folder path wrong:",mmseg_folder)
    exit()

root_folder = os.path.join(mmseg_folder,"data/kramstuff/outdoor")
#train_img_folder = os.path.join(root_folder,"img_dir/train")
#val_img_folder   = os.path.join(root_folder,"img_dir/val")
train_ann_folder = os.path.join(root_folder,"annotations/train")
val_ann_folder   = os.path.join(root_folder,"annotations/val")
folders = [train_ann_folder,val_ann_folder]
for f in folders:
    if not os.path.exists(f):
        print("Missing",f)
        exit()
prev_prefix = "_reseg.png"
new_prefix = ".png"

for i,folder in enumerate(folders):
    filelist = os.listdir(folder)
    for j,oldname in enumerate(filelist):
        print("Folder",
            utils.progress_bar(i,len(folders),length=len(folders),add_numbers=True),
            "File",
            utils.progress_bar(j,len(filelist),add_numbers=True),
            end="\r")
            
        os.rename(
            os.path.join(folder,oldname), 
            os.path.join(folder,oldname[:-len(prev_prefix)]+new_prefix))
print("")
