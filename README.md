
This repository is the code submission for a university project. It contains tools for exploring and re-annotating [the ADE20k dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/), extending it with two other datasets and for using MMSegmentation (training, inference) and visualizing its results. 

# Setup

There are a few resources that need to be obtained before using it. The script `setup.sh` tries to automate most of it, except the dataset-downloads.

## Python requirements

The requirements from `requirements.txt` need to be installed, preferrably in a virtual environment. I use Python Version 3.7 or above, including among others mmcv 1.3.9, mmsegmentation 0.17.0, torch 1.9.0, torchvision 0.10.0. PyTorch and MMCV require special versions of each other and also special download sources. I tried to include everything into `requirements.txt`.

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

I created various scripts during the process, which make up most of this repo and which I want to quickly outline here. Every script can show help on itself without doing anything else by running it like `python3 some_script.py --help`. 

The data sources used by those scripts are:

-   The **ADE20k-index**, i.e. the file `index_ade20k.pkl`, shipped with ADE20k (and also [here](http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl)) which contains metadata about the dataset, including classnames and occurences, which can be used without the dataset itself to query its statistics.
-   The **ADE20k dataset** itself, obtainable via a 5.6GB download from https://groups.csail.mit.edu/vision/datasets/ADE20K (access must first be requested and an account created).
-   The **ade_stats** file is created by `create_ade_stats.py` and contains more accurate stats, like parents (with counts), objectcounts and scene presence for each ADE-class and number of images for each scene. It is a more accurate alternative to the `objects.txt` file shipped with ADE20k
-   The **general configuration**, containing the classes with name, id, scene and color loaded from `classes.json`, which happens by including `conf.py` automatically.
-   The **ADE-specific configuration**, containing the synonyms and parent constraints and the classnames, for re-annotating ADE20k. By splitting the configurations, the classnames, colors and indices can be changed independently of the datasets to re-annotate.



## For exploring ADE20k:

-   `ade20k/create_ade_stats.py` needs the whole dataset and creates the `ade_stats.pkl`-file with more detailed stats about ADE20k.
-   `ade20k/create_ade_summary.py` needs `ade_stats.pkl` and the full dataset and creates a HTML file with examples and statistics for classnames, class combinations and parent-child relationships.
-   `ade20k/list_parents.py` does the almost the same as `create_ade_stats` for single classes. It takes classnames as input and creates a csv file each with all parents instances of this class can have and how often that is the case. It needs the full dataset.
-   `ade20k/ade_class_examples.py` extracts examples for each given class, with outlines drawn around the class instances.
-   `ade20k/query_ade_stats.py` needs `ade_stats.pkl` and lets one pick classes interactively, for which all possible parent classes are shown with the count of how often it is the case, and also how often the class appears in which scene.
-   `ade20k/pick_snippets.py` picks a given number of random snippets from a dataset.

## For exploring filter configurations:

-   `ade20k/create_filter_summary.py` uses the ADE-specific configuration and the whole dataset, to find examples for images matched by each of the target classes, in order to test a filter configuration. Instances are outlined, and the results are included in a generated html-file along with some statistics.
-   `ade20k/threshold_compare.py` uses the ADE-specific configuration and the ADE20k-index to count, how many images match how many target classes and display it as a histogram.

## For processing ADE20k:

-   `ade20k/annotate.py` uses general and ADE-specific configuration files and the whole ADE20k dataset to generate the re-annotated and filtered custom dataset. It also creates a `stats.pkl` file in the newly created dataset's folder, containing image-wise statistics (number of synonym- and full matches, scene, list of all matches)
-   `ade20k/index_transform.py` applies a Lookup-Table to all indices in the dataset. Useful if changes to the indices want to be made without re-annotating everything (like "start at index 1" or "merge class X and Y")

## For running a trained algorithm:

-   `segmentation/inference_test.py` is run on the GPU machine to select trained models from the work directory and let them process custom training images. 
-   
## For visualizing segmentation results:

-   `segmentation/visualize.py` visualizes single inference results with a class legend and pointers to individual 'objects' in the scene.
-   `segmentation/grid.py` can comparatively show inference results. It takes several folders as arguments (or you can select them interactively) and comparatively shows images of same name, which are present in all of them, side by side and in multiple rows. Can be used to quickly compare segmentation outputs of different algorithms.
-   `segmentation/training_plot.py` visualizes overall training progress (IoU over time) and final class-wise performance (IoU) for multiple training processes given their log json-files.
-   `segmentation/class_progression.py` draws the progression of the class-wise performance (IoU or Acc) over time for each class. The classes can be distributed to a 2D-grid of subplots or shown all in the same figure.
-   `segmentation/class_highscores.py` takes all training logs from a folder and ranks them by how many classes they segmented best according to IoU and/or Acc 

## For the other datasets:

-   `other_datasets/explore_labelmefacade.py` shows images from the dataset and their annotations and an overlay of both, as a grid, interactively.
-   `other_datasets/labelme_convert.py` converts the labelme dataset and puts the images into the extended dataset folders. Those are created by copying the outdoor and inout portions of the un-extended dataset, if they are not present yet.
-   `other_datasets/cmp_convert.py` converts the cmp dataset and puts the images into the extended dataset folders. Those are created by copying the outdoor and inout portions of the un-extended dataset, if they are not present yet.

## Others:
-   `class_table.py` creates a HTML file with a table of all classes, their scenes and their colors.


# Training

Training is done as described by MMSegmentation docs [here](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/train.md), only with the new configuration files from the forked MMSegmentation repo. The files are under `mmsegmentation/configs/swin/` and cover the setups I tried during my studies. As I only adapted the nested configuration structure as it was for the original full ade20k-setup, each "highest-level" setup builds on two layers of "smaller" setups, with the following naming scheme, from "high" to "low":
1.  `upernet_swin_base_patch4_window12_512x512_`**SETUP**`_pretrain_22K.py`
2.  `upernet_swin_base_patch4_window12_512x512_`**SETUP**`_pretrain_1K.py`
3.  `upernet_swin_tiny_patch4_window7_512x512_`**SETUP**`_pretrain_224x224_1K.py`

I always used the "highest" configuration. My chosen final algorithm replaces **SETUP** with `80k_studienprojekt_inout_extended_weighted`. Training with this configuration would be the following command from within the `mmsegmentation` folder, with `NUM_GPUS` replaced with the number of GPUs to use:

`./tools/dist_train.sh ./configs/swin/upernet_swin_base_patch4_window12_512x512_80k_studienprojekt_inout_extended_weighted_pretrain_384x384_22K.py NUM_GPUS`