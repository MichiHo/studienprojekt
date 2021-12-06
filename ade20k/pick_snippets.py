"""Pick random snippets from a segmentation dataset. Required are a folder each for images (.jpg) and annotations (.png), with images of same name except the ending. It shuffles the list of images and stores the first N into an output folder."""
import argparse
import os
import random
import shutil

from tqdm import tqdm


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--number', type=int, default=20, help='the number of image-annotation pairs to extract.')
parser.add_argument('--output-dir', type=path_arg, default="dataset_snippets", help='the folder to store the snippets in.')
parser.add_argument('--images-dir', required=True, type=path_arg, help='the folder to load the images from.')
parser.add_argument('--annotations-dir', required=True, type=path_arg, help='the folder to load the annotations from.')
args = parser.parse_args()


if not os.path.exists(out_folder):
    os.makedirs(args.output_dir)

filenames = [f for f in os.listdir(args.images_dir) if f.lower().endswith(".jpg")]
random.shuffle(filenames)

for img_name in tqdm(filenames[:20]):
    ann_name = img_name[:-4] + ".png"
    shutil.copy(os.path.join(args.images_dir,img_name), os.path.join(args.output_dir,img_name))
    shutil.copy(os.path.join(args.annotations_dir,ann_name), os.path.join(args.output_dir,ann_name))
