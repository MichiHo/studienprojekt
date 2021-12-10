"""Takes a filter configuration and quickly counts the images for different
detection thresholds, based on Synonym-matches only. Use this as a rough estimate
of how many target classes you should require at minimum ("detection_threshold" in filters.json)."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import path_arg, conf
import ade_utils as utils

parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(show_plot=False)
parser.add_argument('--conf-path', type=path_arg, default=conf.annotate_filers_conf, help='the path of the filter configuration. (default from configuration)')
parser.add_argument('--show-plot', action="store_true",dest="show_plot", help='whether to show a histogram plot of the results. (default: False)')
parser.add_argument('--save-file',type=path_arg, help='a path to save a histogram plot of the results to. (default: None / no saving)')
args = parser.parse_args()

ade_index = utils.AdeIndex.load()
conf = utils.AdeConfiguration.load(ade_index,args.conf_path)

matches_hist = dict()
for img_index in tqdm(range(utils.num_images),desc="Process all images"):
    try:
        classes = conf.syn_match(ade_index,img_index,True)
        det = len(classes)
        if not det in matches_hist:
            matches_hist[det] = 1
        else:
            matches_hist[det] += 1
    except KeyboardInterrupt:
        exit()
    except BaseException as e:
        print("\n",e)
        stats['errors'].append({
            'img_id': img_index,
            'error': e,
            'error_print': str(e)
        })
print()
print("threshold, number of matched images")
for matches,count in sorted(matches_hist.items(),key=lambda item:item[0]):
    print(f"{matches:9d}, {count:5d} ")        

if args.show_plot or args.save_file is not None:
    plt.bar(matches_hist.keys(),matches_hist.values())
    plt.xlabel("Number of matched classes")
    plt.ylabel("Number of images")
    plt.xticks(np.arange(max(matches_hist.keys())+1))
    if args.save_file is not None:
        plt.savefig(args.save_file)
    if args.show_plot: plt.show()
