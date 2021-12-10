"""Takes all configurations from a directory and computes class-wise highscores of given properties
(defaults to both IoU and Acc)."""

import numpy as np
import matplotlib.pyplot as plt
import os 
import json
import argparse

from utils import *

props = ["IoU","Acc"]
parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(omit_zero=False)
parser.add_argument('--log-dir', type=path_arg, default=conf.segmentation_logs_path,
    help='the folder to look for logfiles in when interactive. (default from configuration)')
parser.add_argument('--properties', default=["IoU", "Acc"], nargs='+', choices=props, 
    help='the properties, for each of which class-wise highscores are shown. (default: ["IoU", "Acc"])')
parser.add_argument('--omit-zero', dest="omit_zero", action="store_true", 
    help='whether to hide a configuration, if it not made any highscore for a property.')
parser.add_argument('--second-bests', type=int, default=100,
    help='how many ranks below the best to also show, in gray.')
args = parser.parse_args()

props = args.properties

files_pre = [f for f in os.listdir(args.log_dir) if f.lower().endswith(".json")]
files_pre.sort()
files = []
values = {p:[] for p in props}
IoUs = []
Accs = []
final_stats = {
    "mIoU": [],
    "aAcc": [],
    "mAcc": []
}
titles = []
for log_name in files_pre:
    try:
        data = multi_root_json_parse(open(os.path.join(args.log_dir,log_name),"r").read())
    except json.JSONDecodeError:
        print("Skipped (json error) ",log_name)
        continue
    this_info = data[0]
    data = data[-1]
    if not "IoU.background" in data:
        print("Skipped (old labels) ",log_name)
        continue
    files.append(log_name)
    
    for prop in props:
        values[prop].append(np.array([(data[f"{prop}.{cl.name}"] if (f'{prop}.{cl.name}' in data) else np.nan) for cl in conf.classes]))
        
    for k in final_stats.keys():
        final_stats[k].append(data[k])
    this_title = nice_config_title(this_info)
    titles.append(this_title)
    print("Processed            ",log_name)

for prop in props:
    values[prop] = np.array(values[prop])


## SCORE PLOT
# xx = np.arange(len(titles))
# for prop,values in final_stats.items():
#     plt.plot(xx,values,"o-",label=prop)
# plt.xticks(xx,titles,rotation="vertical")
# plt.legend()
# plt.grid(axis='x')
# plt.tight_layout()
# plt.show()
# exit()

## WRITE SCORES TO FILE
# with open("class_scores_IoU.csv","w") as csvfile:
#     csvfile.write("Configuration; ")
#     for cl in conf.classes:
#         csvfile.write(cl.name + "; ")
#     csvfile.write("\n ; ")
#     for cl in conf.classes:
#         csvfile.write(cl.scene[0]+"; ")
#     for file_i, title in enumerate(titles):
#         csvfile.write("\n"+title)
#         for val in IoUs[file_i]:
#             csvfile.write("; " + str(val))

# with open("class_scores_Acc.csv","w") as csvfile:
#     csvfile.write("Configuration; ")
#     for cl in conf.classes:
#         csvfile.write(cl.name + "; ")
#     csvfile.write("\n ; ")
#     for cl in conf.classes:
#         csvfile.write(cl.scene[0]+"; ")
#     for file_i, title in enumerate(titles):
#         csvfile.write("\n"+title)
#         for val in Accs[file_i]:
#             csvfile.write("; " + str(val))
            
def fmt(n):
    return f"{n:.3f}"

def fmtlist(lis):
    return " ".join([fmt(n) for n in lis])

def winners(lis):
    lis2 = list([(i,v) for i,v in enumerate(lis) if not np.isnan(v)])
    lis2.sort(key=lambda t:t[1],reverse=True)
    return [t[0] for t in lis2]
    
highscores = {p: [(i,[]) for i in range(len(values[p]))] for p in props}
other_winners = {p: [[]] * len(conf.classes) for p in props}
other_values = {p: [[]] * len(conf.classes) for p in props}

# IoU_highscores = [(i,[]) for i in range(len(IoUs))]
# Acc_highscores = [(i,[]) for i in range(len(Accs))]

# other_IoU_winners = [[]] * len(conf.classes)
# other_IoUs = [[]] * len(conf.classes)
# other_Acc_winners = [[]] * len(conf.classes)
# other_Accs = [[]] * len(conf.classes)

for class_index,cl in enumerate(conf.classes):
    for prop in props:
        these_values = values[prop][:,class_index]
        these_winners = winners(these_values)
        highscores[prop][these_winners[0]][1].append(class_index)
        other_winners[prop][class_index] = these_winners
        other_values[prop][class_index] = these_values[these_winners]
    
    # this_IoU = IoUs[:,class_index]
    # IoU_winners = winners(this_IoU)
    # IoU_highscores[IoU_winners[0]][1].append(class_index)
    # other_IoU_winners[class_index] = IoU_winners
    # other_IoUs[class_index] = this_IoU[other_IoU_winners[class_index]]
    
    # this_Acc = Accs[:,class_index]
    # Acc_winners = winners(this_Acc)
    # Acc_highscores[Acc_winners[0]][1].append(class_index)
    # other_Acc_winners[class_index] = Acc_winners
    # other_Accs[class_index] = this_Acc[other_Acc_winners[class_index]]

for prop in props:
    highscores[prop].sort(key=lambda h:len(h[1]),reverse=True)
# IoU_highscores.sort(key=lambda h:len(h[1]),reverse=True)
# Acc_highscores.sort(key=lambda h:len(h[1]),reverse=True)

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple
Gy  = '\033[90m' # Gray

def class_title(l,t):
    return f"## {l} classes best segmented by {B}{t:30s}{W}"

def class_line(class_index,file_index,prop):
    win_val = values[prop][file_index][class_index]
    #other_vals = other_IoUs if prop == "IoU" else other_Accs
    class_col = {"indoor":G, "outdoor":P,"both":W}[conf.classes[class_index].scene]
    return f" | {class_col}{conf.classes[class_index].name[:10]:10s}{W}: {prop}={fmt(win_val)} {Gy}{fmtlist(other_values[prop][class_index][:args.second_bests])}{W}"


for prop in props:
    print("\n\n################################",prop)
    for file_index,classes in highscores[prop]:
        if len(classes) == 0 and args.omit_zero: continue
        
        print(class_title(len(classes),titles[file_index]))
        for class_index in classes:
            print(class_line(class_index,file_index,prop))

# print("\n\n################################ IoU")
# for file_index,classes in IoU_highscores:
#     #if len(classes) == 0: continue
    
#     print(class_title(len(classes),titles[file_index]))
#     for class_index in classes:
#         print(class_line(class_index,file_index,"IoU"))


# print("\n\n################################ Acc")
# for file_index,classes in Acc_highscores:
#     #if len(classes) == 0: continue
    
#     print(class_title(len(classes),titles[file_index]))
#     for class_index in classes:
#         print(class_line(class_index,file_index,"Acc"))