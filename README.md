[![DOI](https://zenodo.org/badge/519150416.svg)](https://zenodo.org/badge/latestdoi/519150416)
# SciAugment
SciAugment aims to provide tools for image augmentation for object detection (YOLO) based on machine on which images were taken.

The tools are created around idea that to create more robust detecton or classification, augmentation needs to be done with respect of how the imaging process works. That is mainly problematic with respect to brightness values. Albumentaion package (which is ingeniuos as is for RGB data) is used for augmentation as flips, rotations or addition of noise. But other channel focused augmentation needs to be added in future.

So far two experimental approaches exists.

**Default** setting, which only flips and rotates images. 

**fluorescece_microscopy**  setting, which aims to reproduce noise and brightness shifts, and has to be tested and tuned.

The input file type can be specified, expects RGB image and is done by OpenCV. The results is in .jpeg - which is not ideal, and there will be possibility to change it in future. But for now it is enough for testing object detection/classification with YOLO_v5.

SciAugment creates one image for each augmentation, and marks resulting image (so user have more controll over results and feedback). The result is preapred as train_data folder with 70/30 distribution for train/test and can be directly used for training of YOLO_v5.

# Examples
Example Google Colab notebook shows simple use of augmentation of both image and YOLO anotation. In folder is small test set of images and annotations (made in [https://www.makesense.ai/](https://www.makesense.ai/)), result is prepared train_data folder with 70/30 train/test distribution.

# SciCount
SciAugment is part of [SciCount](https://github.com/martinschatz-cz/SciCount) project.
