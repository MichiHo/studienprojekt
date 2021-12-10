"""Creates a two-part plot showing statistics for one or multiple training logs. The left plot shows
a global statistic (mIoU, aAcc or mAcc) over time and the right plot shows a class statistic (IoU or Acc)
for each class. If --mode joint is set, all training logs are shown in the same plot, with different
colors."""
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


parser = argparse.ArgumentParser(description=__doc__)
parser.set_defaults(save_plots=True,epochs=False)
parser.add_argument('logfiles', type=path_arg, nargs='*', default=None,  help='the training log json to visualize.')
parser.add_argument('--log-dir', type=path_arg, default=conf.segmentation_logs_path, help='the folder to look for logfiles in when interactive. (default from configuration)')
parser.add_argument('--plots-dir', type=path_arg, default=conf.segmentation_plots_path, help='the folder to store generated plots into. Will be created if not existent. No files are overwritten. (default from configuration)')
parser.add_argument('--no-save-plots', dest="save_plots", action="save_false",  help='dont save plots as svg into a folder.')
parser.add_argument('--epochs', dest="epochs", action="save_true",  help='use epochs instead of iterations as time value.')
parser.add_argument('--mode', default='joint', choices=['joint','separate'], help='whether to display multiple training logs in one joint plot or in separate plots. (default: joint)')
parser.add_argument('--class-property', default="IoU", choices=['IoU','Acc'],  help='the property displayed in the class-wise plot. (default: IoU)')
parser.add_argument('--global-property', default="mIoU", choices=['mIoU','aAcc','mAcc','lr'],  help='the property displayed over time in the total plot. (default: mIoU)')
parser.add_argument('--second-global-property', default=None, choices=['mIoU','aAcc','mAcc','lr'],  help='which other property to plot along the global-property on the left plot. (default: None)')
parser.add_argument('--colors',type=str,default=["red","blue","green","orange","purple","pink"],nargs='+', help='colors to assign to the training logs in that order.')
parser.add_argument('--names',type=str,nargs='+',default=[], help='names to assign to the training logs in that order. (defaults to the main part of the filename or the name specified in the log)')
parser.add_argument('--time-range',choices=['min','max','ask'],default='ask', help="how to behave in case of different time spans of the training logs. 'min' crops all logs by the shortest length, 'max' displays all logs completely and 'ask' asks interactively. (default: ask)")
args = parser.parse_args()

classnames = [cl.name for cl in conf.classes]



if args.save_plots:
    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)
        print("Generated plot output folder",args.plots_dir)

if args.logfiles is not None and len(args.logfiles) > 0:
    paths = args.logfiles
else:
    all_files = os.listdir(args.log_dir)
    log_files = list(filter(lambda it: str(it).endswith(".json"),all_files))
    if (len(log_files) == 0):
        print("No log files found.")
        exit()
    print(f"Found {len(log_files)} log files:")
    for i,f in enumerate(log_files):
        print(f"[{i:2}] {f}")
    choice = int(input("Which one to choose? "))
    paths = [os.path.join(args.log_dir,log_files[choice])]
    while True:
        choice2 = input("Next? (leave empty to stop) ").strip()
        if choice2 != "":
            choice2 = int(choice2)
            paths.append(os.path.join(args.log_dir,log_files[choice2]))
        else: break

train_steps = []
val_steps = []
infos = []
max_length = []
min_acc = 1.0
max_acc = 0.0
print("Process ...")
for log_path in paths:
    these_train_steps = []
    these_val_steps = []
    last_train_iter = 0
    with open(log_path) as f:
        jsons = multi_root_json_parse(f.read())
        
    infos.append(jsons[0])
    skipped = 0
    for line_json in jsons[1:]:
        if not 'mode' in line_json.keys():
            #print("Skip")
            skipped += 1
            continue
        if line_json['mode'] == "train":
            these_train_steps.append(line_json)
            last_train_iter = line_json['iter']
        else:
            line_json['last_train_iter'] = last_train_iter
            these_val_steps.append(line_json)
            min_acc = min(min_acc,line_json['aAcc'])
            max_acc = max(max_acc,line_json['aAcc'])
    print(f" -> {log_path}: {len(these_train_steps)} train steps, {len(these_val_steps)} val steps {skipped} skipped lines")
    train_steps.append(these_train_steps)
    val_steps.append(these_val_steps)
    if args.epochs:
        max_length.append(max(these_val_steps[-1]['epoch'],these_train_steps[-1]['epoch']))
    else:
        max_length.append(last_train_iter)

common_length = min(max_length)
if args.time_range == 'min' or (args.time_range == 'ask' and input(f"Common length {common_length} {'epochs' if args.epochs else 'iterations'}? [y/n] ").lower() == "y"):
    for i in range(len(train_steps)):
        train_steps[i] = list(filter(lambda it: 
                it['epoch' if args.epochs else 'iter']<= common_length,train_steps[i]))
        val_steps[i] = list(filter(lambda it: 
                it['epoch' if args.epochs else 'last_train_iter']<= common_length,val_steps[i]))
    


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


if args.mode == 'separate':
    fig,ax = plt.subplots(len(paths),2,figsize=(10,10))
    if len(paths) == 1:
        ax = ax.reshape([1,-1])
elif args.mode == 'joint':
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    #if not epoch_x: otherax = ax[0].twinx()
print("Plot...")
plot_filename = ""
for i,this_path in enumerate(paths):
    print(" -> ",this_path)
    these_train_steps = train_steps[i]
    these_val_steps = val_steps[i]
    time_steps = [item['epoch'] for item in these_val_steps] if args.epochs else [item['last_train_iter']/1000 for item in these_val_steps]
    color = args.colors[i % len(args.colors)]
    this_info = infos[i]
    if len(args.names) > i:
        this_title = args.names[i]
    else:
        this_title = nice_config_title(this_info)
    plot_filename += this_title + "__"
    
    if args.mode == 'separate':
        normal_ax = ax[i,0]
        normal_ax.set_xlabel('Epoch' if args.epochs else 'Iteration (x 1000)')
        normal_ax.set_ylabel(args.global_property+" (Validation)")
        normal_ax.set_title(this_title)
        
        normal_ax.plot(time_steps,
            [item[args.global_property] for item in these_val_steps],"o-",color=color,label=f"{args.global_property} (Validation)")
        if args.second_global_property is not None:
            other_ax = normal_ax.twinx()
            other_ax.plot(time_steps,[item[args.second_global_property] for item in these_val_steps],'--',color=color,label=f"{args.second_global_property} (Validation)")
            other_ax.set_ylabel(args.second_global_property+" (Validation) - dashed line")
            
        classplot(ax[i,1], these_val_steps[-1],prop=args.class_property)
    elif args.mode == 'joint':
        if i==0:
            #ax[0].set_ylim([min_acc,1.0])
            normal_ax = ax[0]
            normal_ax.set_xlabel('Epoch' if args.epochs else 'Iteration (x 1000)')
            normal_ax.set_ylabel(args.global_property+" (Validation)")
            normal_ax.set_title(args.global_property + " over time")
            if args.second_global_property is not None:
                other_ax = ax[0].twinx()
                other_ax.set_ylabel(args.second_global_property + " (Validation) - dashed line")
            #if not epoch_x: otherax.set_ylabel('Epoch (transparent)')
        normal_ax.plot(
            [item['epoch'] if args.epochs else item['last_train_iter']/1000 for item in these_val_steps],
            [item[args.global_property] for item in these_val_steps],"o-",color=color)
        if args.second_global_property is not None:
            other_ax.plot(
                [item['epoch'] if args.epochs else item['last_train_iter']/1000 for item in these_val_steps],
                [item[args.second_global_property] for item in these_val_steps],"--",color=color)
        classplot(ax[1],these_val_steps[-1],
                  setup=i==0, prop=args.class_property,
                  label=f"{this_title} | {these_val_steps[-1]['epoch']} epochs", color=color,
                  i=i,length=len(paths),classnames=classnames) #,index_shift=(-1 if 'converted_classes' in this_info else 0)
if args.mode == 'joint':
    li,la = ax[1].get_legend_handles_labels()
    ax[0].legend(li,la,loc=0)

    
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
if args.save_plots:
    plot_filename += f"{args.mode}-plots_{args.global_property}"
    if args.second_global_property is not None:
        plot_filename += f"+{args.second_global_property}"
    plot_filename += f"_{args.class_property}_{'epochs' if args.epochs else 'iterations'}"
    plot_filepath = os.path.join(args.plots_dir,plot_filename+".svg")
    i = 0
    while os.path.exists(plot_filepath):
        plot_filepath = os.path.join(args.plots_dir,plot_filename+f"_{i}.svg")
        i += 1
    plt.savefig(plot_filepath)
    print("Saved plot to",plot_filepath)
plt.show()
