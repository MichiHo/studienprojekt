import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

labels_path = "labelmefacade/labels/"
images_path = "labelmefacade/images/"

label_colors = {
    "various": (0, 0, 0),
    "building": (1.0, 0, 0),
    "car": (1.0, 0, 1.0),
    "door": (1.0, 1.0, 0),
    "pavement": (1.0, 1.0, 1.0),
    "road": (1.0, 0.5, 0),
    "sky": (0, 1.0, 1.0),
    "vegetation": (0, 1.0, 0),
    "window": (0, 0, 1.0)
}

def seg_legend(colors : dict,ax : plt.Axes):
    artists = []
    for label, color in colors.items():
        artists.append(plt.Rectangle((0,0),10,10,0.0,color=color))
    ax.legend(artists,colors.keys(),bbox_to_anchor=(1,1), loc="upper left")

def img_path_to_lab_path(path):
    return path[:-4] + ".png"

img_batch = 3
i = 0
rows = []
for filename in os.listdir(images_path):
    i = i+1
    img_path = os.path.join(images_path, filename)
    lab_path = os.path.join(labels_path, img_path_to_lab_path(filename))

    print(img_path, lab_path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    lab = cv2.imread(lab_path)
    lab = cv2.cvtColor(lab,cv2.COLOR_BGR2RGB)

    #print(lab.shape)
    #lab[np.where((lab!=label_colors["window"]).all(axis=2))] = [0,0,0]
    #lab[lab != label_colors["window"]] = (0,0,0)

    mix = cv2.addWeighted(img, 0.4, lab, 0.6, 0)
    rows.append(np.concatenate((img, mix, lab), axis=1))

    if(i % img_batch == 0):

        ax = plt.subplot()
        ax.imshow(np.concatenate(rows,axis=0))
        #ax.imshow(mix)
        seg_legend(label_colors,ax)
        plt.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        rows = []
