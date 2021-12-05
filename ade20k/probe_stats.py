import pickle

import conf
from tqdm import tqdm

import ade_utils as utils

ade_index = utils.adeindex.load()

with open("stats.pkl","rb") as ff:
    stats = pickle.load(ff)

print(stats['images'][12]['matches'])

class_search = {
    "indoor" : {cl.name for cl in conf.classes if cl.scene == "outdoor"},
    "outdoor" : {cl.name for cl in conf.classes if cl.scene == "indoor"}
}
print(class_search)
results = {
    "indoor": dict(),
    "outdoor": dict()
}

for img in tqdm(stats['images']):
    sc = img['scene']
    for cl in class_search[sc]:
        if cl in img['matches']:
            if not cl in results[sc]:
                results[sc][cl] = {
                    "example": ade_index['filename'][img['id']],
                    "count": 0
                }
            results[sc][cl]["count"] += 1

for scene,result in results.items():
    print(f"Non-{scene} classes found in {scene} images:")
    for cl_name,cl_results in results[scene]:
        print(f"- {cl_name}: {cl_results['count']} images, e.g. {cl_results['example']}")
