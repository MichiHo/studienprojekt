"""Show the performance of individual classes over time given the training log file. If no logfile argument is given,
it interactively shows all files in the log-dir and gives a choice."""
import numpy as np
import matplotlib.pyplot as plt
import os 
import json
import sys 
import argparse

from utils import *

"""
Log File structure:
"mode": "train",
"epoch": 1,
"iter": 150,
"lr": 1e-05,
"memory": 13059,
"data_time": 0.01308,
"decode.loss_seg": 1.46337,
"decode.acc_seg": 27.17729,
"aux.loss_seg": 0.65213,
"aux.acc_seg": 5.07046,
"loss": 2.1155,
"time": 0.50696

"mode": "val",
"epoch": 6,
"iter": 186,
"lr": 5e-05,
"aAcc": 0.8577,
"mIoU": 0.303,
"mAcc": 0.3452,
"IoU.<CLASSNAME>": 0.475,
"Acc.<CLASSNAME>": 0.999,
"""

def gridarg(strr):
    'Must be of the format NxM with N and M being positive integers.'
    strr = strr.lower()
    if not 'x' in strr: raise ValueError(__doc__)
    grid = strr.split('x')
    if len(grid) != 2: raise ValueError(__doc__)
    grid = [int(x) for x in grid]
    if grid[0]<1 or grid[1]<1: raise ValueError(__doc__)
    return grid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('logfile', type=path_arg, nargs='?', default=None, help='the training log json to visualize.')
parser.add_argument('--log-dir', type=path_arg, default=conf.segmentation_logs_path,help='the folder to look for logfiles in when interactive. (default from configuration)')
parser.add_argument('--plots-dir', type=path_arg, default=conf.segmentation_plots_path,help='the folder to store generated plots into. Will be created if not existent. No files are overwritten. (default from configuration)')
parser.add_argument('--save-plots', default=True, action=argparse.BooleanOptionalAction, help='whether to save plots as svg into a folder.')
parser.add_argument('--exclude', type=str, nargs='+',default=[],help='the names of classes to exclude.')
parser.add_argument('--only', type=str, nargs='+',default=[],help='which classes to display only.')
parser.add_argument('--grid',type=gridarg,default=[2,3],help='Size of the grid to split the plots into. 1x1 shows all in a single plot. '+gridarg.__doc__)
parser.add_argument('--epochs', default=False, action=argparse.BooleanOptionalAction, help='whether to use epochs instead of iterations as time value.')
parser.add_argument('--property', default="IoU", choices=['IoU','Acc'], help='the property whose values to display.')
args = parser.parse_args()

def multi_root_json_parse(string):
    """Parse a json string with multiple root elements into a list of dicts"""
    import re
    regex = re.compile("\{|\}")
    level = 0
    start = 0
    jsons = []
    for match in regex.finditer(string):
        if match.group() == "{": level+=1
        elif match.group() == "}":
            if level <= 0:
                raise ValueError("JSON bracket level conflict!! Too many }'s")
            level -= 1
            if level == 0:
                end = match.span()[1]
                jsons.append(json.loads(string[start:end]))
                start = end+1
    return jsons


classnames = [cl.name for cl in conf.classes]
if len(args.exclude) > 0:
    classnames = [cl for cl in classnames if cl not in args.exclude]
if len(args.only) > 0:
    classnames = [cl for cl in classnames if cl in args.only]


colors = ["red","blue","green","orange","purple","pink"]

if args.logfile is not None:
    log_path = args.logfile
    if not os.path.exists(log_path):
        print("Log file",log_path,"does not exist!")
        exit()
else:
    if not os.path.exists(args.log_dir):
        print("Log folder",args.log_dir,"does not exist!")
        exit()
    all_files = os.listdir(args.log_dir)
    log_files = list(filter(lambda it: str(it).endswith(".json"),all_files))
    if (len(log_files) == 0):
        print("No log files found in",args.log_dir)
        exit()
    print(f"Found {len(log_files)} log files:")
    for i,f in enumerate(log_files):
        print(f"[{i:2}] {f}")
    choice = int(input("Which one to choose? "))
    log_path = os.path.join(args.log_dir,log_files[choice])

if args.save_plots:
    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)
        print("Generated plot output folder",args.plots_dir)

max_length = []
train_steps = []
val_steps = []
last_train_iter = 0
with open(log_path) as f:
    jsons = multi_root_json_parse(f.read())
    
this_info = jsons[0]
for line_json in jsons[1:]:
    if not 'mode' in line_json.keys():
        print("Skip")
        continue
    if line_json['mode'] == "train":
        train_steps.append(line_json)
        last_train_iter = line_json['iter']
    else:
        line_json['last_train_iter'] = last_train_iter
        val_steps.append(line_json)
print(f"{log_path}: {len(train_steps)} train steps, {len(val_steps)} val steps")
if args.epochs:
    max_length = max(val_steps[-1]['epoch'],train_steps[-1]['epoch'])
else:
    max_length = last_train_iter

def classplot(axis,val,setup=True,prop="IoU",label="",color="blue",i=0,length=1,classnames=None,index_shift=0):
    barwidth = 0.6
    w = barwidth / length
    shift = -0.5*barwidth + 0.5*w + w*i
    indices = []
    values = []
    ticks = classnames
    if f"{prop}.grass" in val:
        if classnames is None:
            classnames = ade_to_mydataset.values()
            ticks = ade_to_mydataset.keys()
        else:
            classnames = [ade_to_mydataset[k] for k in ticks if k in ade_to_mydataset]
        
    elif classnames is None:
        classnames = [p[4:] for p in val.keys() if p.startswith(f"{prop}.") and not np.isnan(val[p])]
        ticks = classnames
    
    for i,cl in enumerate(classnames):
        propname = f"{prop}.{cl}"
        if propname in val.keys():
            i2 = i + index_shift
            if i2 < 0: continue
            indices.append(i2)
            values.append(val[propname])
    indices = np.array(indices)
    axis.bar(indices+shift,values,w,label=label,color=color)
    if setup:
        axis.set_xticks(np.arange(len(ticks)))
        axis.set_xticklabels([tick[:10] for tick in ticks],rotation="vertical")
        axis.set_ylabel(prop)
        axis.set_title(f"{prop} for classes")

grid = [2,3]
class_groups = [
    ["background","window","door"],
    ["ceiling","floor","stairs"],
    [""]
]
plots_per_cell = int(np.ceil(len(classnames) / (args.grid[0]*args.grid[1])))
print(f"Grid {args.grid} with {plots_per_cell} plots per cell.")

fig,ax = plt.subplots(nrows=args.grid[0],ncols=args.grid[1],squeeze=False,sharex=True,sharey=True,figsize=(10,10))

plot_filename = ""
print("Plot",log_path)
this_title = nice_config_title(this_info)
plot_filename += this_title + "__"
plot_x = -1
plot_y = -1
plot_i = 0
axis = None
time_axis = [item['epoch'] if args.epochs else item['last_train_iter']/1000 for item in val_steps]
improvements = np.zeros((len(time_axis)))

print("\nTable of class highscores per",('epoch' if args.epochs else 'iteration (x 1000)'),":")
print("classname, ",", ".join([str(t) for t in time_axis]))
for class_index,class_name in enumerate(classnames):
    if class_index % plots_per_cell == 0:
        if plot_i > 0:
            axis.legend()
        axis = ax[plot_i % args.grid[0],int(plot_i/args.grid[0])]
        axis.set_xlabel('Epoch' if args.epochs else 'Iteration (x 1000)')
        axis.set_ylabel(args.property)
        plot_i+=1
    propname = args.property+"."+class_name
    values = [item[propname] for item in val_steps]
    color = conf.by_name[class_name].color / 255.0
    imprv_indices = []
    maxx = val_steps[0][propname]
    print(class_name,", ",end="")
    for i,step in enumerate(val_steps[1:]):
        if step[propname] > maxx:
            maxx = step[propname]
            imprv_indices.append(i+1)
            print("x, ",end="")
        else:
            print(" , ",end="")
    print("")
    improvements[imprv_indices] += 1
    axis.plot(
        time_axis,
        values,"o-",color=color,label=class_name,
        markevery=imprv_indices)
    
print("")

axis.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

if args.save_plots:
    plot_filename += f"classprog_{args.property}_{'epochs' if args.epochs else 'iterations'}"
    plot_filepath = os.path.join(args.plots_dir,plot_filename+".svg")
    i = 0
    while os.path.exists(plot_filepath):
        plot_filepath = os.path.join(args.plots_dir,plot_filename+f"_{i}.svg")
        i += 1
    plt.savefig(plot_filepath)
    print("Saved plot to",plot_filepath)

plt.show()
