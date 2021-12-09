import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import ade_utils as utils

ade_index = utils.AdeIndex.load()
conf = utils.AdeConfiguration.load(ade_index,"filters.json")

matches_hist = dict()
matches_classes = dict()
for img_index in range(utils.num_images):
    try:
        classes = conf.syn_match(ade_index,img_index,True)
        det = len(classes)
        if not det in matches_hist:
            matches_hist[det] = 1
            matches_classes[det] = set(classes)
        else:
            matches_hist[det] += 1
            matches_classes[det].update(classes)
    except KeyboardInterrupt:
        exit()
    except BaseException as e:
        print("\n",e)
        stats['errors'].append({
            'img_id': img_index,
            'error': e,
            'error_print': str(e)
        })

for matches,count in matches_hist.items():
    print(f"{matches:05d} : {count:04d} : {', '.join([cl.name for cl in matches_classes[matches]])}")        
    
plt.bar(matches_hist.keys(),matches_hist.values())
plt.show()
