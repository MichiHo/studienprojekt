import os

import ade_utils as utils

out_file_path = os.path.expanduser("~/ssh_transfer/scene_names.csv")

with open(out_file_path,"w") as out_file:
    # only training
    # training > l1 > l2 > json files
    lvl0_path = utils.training_images_path
    for lvl1_name in os.listdir(lvl0_path):
        lvl1_path = os.path.join(lvl0_path,lvl1_name)
        
        for lvl2_name in os.listdir(lvl1_path):
            lvl2_path = os.path.join(lvl1_path,lvl2_name)
            
            sample_file = None
            for f in os.listdir(lvl2_path):
                if f.endswith(".json"):
                    sample_file = f
                    break
            if sample_file == None:
                print(f"No .json found in {lvl2_path}!")
                continue
            sample_ann = utils.imgdata.load(lvl2_path,sample_file[:-5])
            scene_string = ", ".join(sample_ann['scene'])
            out_file.write(f"{lvl2_path}; {scene_string}\n")
        