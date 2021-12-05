# Setup

This repository contains tools for exploring and re-annotating ADE20k and for using MMSegmentation (training, inference) and visualizing its results. There are a few resources that need to be obtained before using it. The script `setup.sh` tries to automate some of this, except the dataset-downloads.

## Python requirements

The requirements from `requirements.txt` need to be installed, preferrably in a virtual environment. 

## MMSegmentation

Needed for: Training, Inference

Apart from installing the MMSegmentation package for python, it must also be cloned in order to access the tools and add the configurations. I created a fork from the version 0.17.0 which the project is using, containing the new dataset and training configurations (https://github.com/MichiHo/mmsegmentation under the branch `studienprojekt`) which must be cloned from the root directory, creating a folder called `mmsegmentation/`.

### Imagenet-Pretrained model

The pretrained weights for the algorithm must also be downloaded, and converted with a script. They can be found [here](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models).

The one I used comes from the "largest" configuration for ADE20k, with the largest windowsize and the larger ImageNet, `swin_base_patch4_window12_384_22k`. 
It can be directly downloaded from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth).

The **conversion** to the MMSegmentation format is done with the script `swin2mmseg.py`, as described [here](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/README.md#usage).

## Trained Models

Needed for: Inference

I pre-trained models on the modified dataset, which are available HERE???

For each model, there is the .pth file with the weights and a .py file with the configuration. Both are required for loading and using it (as far as I understand MMSegmentation). If the models are stored as directories (of any chosen name) under `mmsegmentation/work_dirs`, they are found by default by the `segmentation/inference_test.py` script.

## Studienprojekt dataset

Needed for: Training

If the training wants to be reproduced, or different training configurations used, the modified dataset must be downloaded into the `mmsegmentation/data` folder. There are a few variations of the dataset, which should be subdirectories of the data folder, to be found by the training configurations:

-   `indoor`: Only indoor images from ADE20k
-   `outdoor`: Only outdoor images from ADE20k
-   `inout`: All re-annotated images from ADE20k
-   `outdoor_extended`: outdoor + additional images from other datasets
-   `inout_extended`: inout + additional images from other datasets

## ADE20k dataset

Needed for: (Re)Creating the custom dataset

If the re-annotation results want to be reproduced, or a different configuration used (different target classes, etc.), the whole dataset must be downloaded. For that, an account needs to be created [at the dataset website](https://groups.csail.mit.edu/vision/datasets/ADE20K/), within which access can be requested. After access is granted, the dataset can be downloaded and extracted into the root folder. The result should be a directory `ADE20K_2021_17_01`, containing an `images` folder and the three files `index_ade20k.mat`, `index_ade20k.pkl`, and `objects.txt`. 

Certain scripts only use the `index_ade20k.pkl` file, which can be downloaded without the rest of the dataset [here](http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl).

## Extension datasets

Needed for: (Re)Creating the custom dataset with extensions

I included scripts to extend the dataset with cmp_facade and labelmefacade, which need to be downloaded to create the extended datasets. I created a script `other_datasets/download_datasets.sh` that does this.

-   **Labelmefacade** can be downloaded by cloning [this repo](https://github.com/cvjena/labelmefacade.git).
-   **CMP_facade** can be downloaded (base and extended) [here](https://cmp.felk.cvut.cz/~tylecr1/facade/). Then all images from base and extended must be copied into `other_datasets/cmp_facade/all`.


# Scripts

I created various scripts during the process, which make up most of this repo and which I want to quickly outline here. The data sources used by those scripts are:

-   The **ADE20k-index**, i.e. the file `index_ade20k.pkl`, shipped with ADE20k (and also [here](http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl)) which contains metadata about the dataset, including classnames and occurences, which can be used without the dataset itself to query its statistics.
-   The **ADE20k dataset** itself, obtainable via a 5.6GB download from https://groups.csail.mit.edu/vision/datasets/ADE20K (access must first be requested and an account created).
-   The **classes_new** file is created by `additional_dataset_stats.py` and contains more accurate stats of which class can be part of which, and how often that is the case.
-   The **general configuration**, containing the classes with name, id, scene and color loaded from `classes.json`, which happens by including `conf.py` automatically.
-   The **ADE-specific configuration**, containing the synonyms and parent constraints and the classnames, for re-annotating ADE20k. By splitting the configurations, the classnames, colors and indices can be changed independently of the datasets to re-annotate.



## For exploring ADE20k:

-   `ade20k/class_query.py` lets the user interactively select classes for which either a number of examples is extracted from the dataset, or the total number of images containing the class is counted and displayed.
-   `ade20k/class_stats.py` can be optionally given a classname. It will then compute stats for either only this class or all classes. When computing stats for all classes, it counts the images containing each class and the total number of instances of each class and outputs those, along with the 'objectcounts' and 'proportionClassIsPart' properties from the ADE20k-index, into a csv-file. When computing stats for a single class, it loads the json file for each image containing the class, counts how often the objects of this class are part of which or no class and outputs those into a csv-file.
-   `ade20k/examples.py` extracts one example for each class in a list, with outlines drawn around the class instances.
-   `ade20k/img_details.py` interactively takes image IDs (which are used in the ADE-index) and shows contained classes and counts.
-   `ade20k/img_outlines.py` takes a folder containing images and their corresponding json files, and is configured with a list of ADE-class names and colors and saves the images, with colored outlines of the class instances, in an output folder.
-   `ade20k/additional_dataset_stats.py` takes the whole dataset and creates a pkl-file containing parents (with counts), objectcounts and scene presence for each ADE-class and number of images for each scene.
-   `ade20k/make_summary.py` takes a set of ADE-classes and creates a HTML file with examples and statistics.
-   `ade20k/objects.py` takes the classes_new file and lets one pick classes interactively, for which all possible parent classes are shown with the count of how often it is the case, and also how often the class appears in which scene.

TODO: one script for one/multiple examples of one/multiple ade-classes, optionally outlined
and one script for stats.

## For exploring filter configurations:

-   `ade20k/count_matches.py` uses the ADE-specific configuration and the ADE20k-index to count, how many images match how many target classes and display it as a histogram.
-   `ade20k/filter_test.py` uses the ADE-specific configuration and the whole dataset, to find examples for images matched by each of the target classes, in order to test a filter configuration. Instances are outlined, and the results are included in a generated html-file along with some statistics.

## For processing ADE20k:

-   `ade20k/annotate.py` uses general and ADE-specific configuration files and the whole ADE20k dataset to generate the re-annotated and filtered custom dataset. It also creates a `stats.pkl` file in the newly created dataset's folder, containing image-wise statistics (number of synonym- and full matches, scene, list of all matches)

## For visualizing segmentation results:

-   `segmentation/inference_test.py` is run on the GPU machine to select trained models from the work directory and let them process custom training images. 
-   `segmentation/grid.py` can comparatively show inference results. It takes several folders as arguments and comparatively shows images of same name, which are present in all of them, side by side and in multiple rows. Can be used to quickly compare segmentation outputs of different algorithms.
-   `segmentation/training_plot.py` visualizes overall training progress (IoU over time) and final class-wise performance (IoU) for multiple training processes given their log json-files.
-   `segmentation/class_progression.py` draws the progression of the class-wise performance (IoU or Acc) over time for each class. The classes can be distributed to a 2D-grid of subplots or shown all in the same figure.
-   `segmentation/class_highscores.py` takes all training logs from a folder and ranks them by how many classes they segmented best according to IoU and/or Acc 

## For the other datasets:

-   `explore_labelmefacade.py` shows images from the dataset and their annotations and an overlay of both, as a grid, interactively.
-   `labelme_convert.py` converts the labelme dataset and puts the images into the extended dataset folders. Those are created by copying the outdoor and inout portions of the un-extended dataset, if they are not present yet.
-   `cmp_convert.py` converts the cmp dataset and puts the images into the extended dataset folders. Those are created by copying the outdoor and inout portions of the un-extended dataset, if they are not present yet.