[![DOI](https://zenodo.org/badge/519150416.svg)](https://zenodo.org/badge/latestdoi/519150416)
# SciAugment v0.2.0
SciAugment aims to provide tools for image augmentation for object detection (YOLO) based on machine on which images were taken.

The tools are created around idea that to create more robust detecton or classification, augmentation needs to be done with respect of how the imaging process works. That is mainly problematic with respect to brightness values. Albumentaion package (which is ingeniuos as is for RGB data) is used for augmentation as flips, rotations or addition of noise. But other channel focused augmentation needs to be added in future.

So far three experimental approaches exists.

*aug_type* (str)

**Default** setting, which only flips and rotates images. 

**fluorescece_microscopy**  setting, which aims to reproduce noise and brightness shifts, and has to be tested and tuned.

**all** Will aplly all augmentation.

**no_augment** No augment setting will only divide images and labels to train_data folder

*channel_aug* (boolean)

**True/False** False as default. This setting will allow channel wise augmentation


The input file type can be specified, expects RGB image and is done by OpenCV. The results by deafult is .jpeg input .png, and there is possibility to change it when applying the augmentation.

SciAugment creates one image for each augmentation, and marks resulting image (so user have more controll over results and feedback). The result is preapred as train_data folder with 70/30 distribution for train/test and can be directly used for training of YOLO_v5.

## Examples
Example Google Colab notebook shows simple use of augmentation of both image and YOLO anotation. In folder is small test set of images and annotations (made in [https://www.makesense.ai/](https://www.makesense.ai/)), result is prepared train_data folder with 70/30 train/test distribution.

Examples of object detection with YOLO v5 (in form of Colab Notebooks) can be found in [SciCount](https://github.com/martinschatz-cz/SciCount) project.

## Easy use

Instal package
```bash
pip install git+https://github.com/martinschatz-cz/SciAugment.git
```

Import all tools
```python
from SciAugment.SciAug_tools import *
```

Create and augmentation type object, and apply channel wise augmentation on folde with images and YOLO anotations
```python
aug2 = SciAugment(aug_type = 'fluorescece_microscopy', channel_aug = True)
input_images_folder = 'data/for_training'
input_image_format = '.jpeg'
out_format = '.png'
aug2.augment_data(images_path=input_images_folder, image_format=input_image_format, train = 0.7, output_image_format = out_format)
```

Results is train_data folder wit (default) 70/30 train/test random distribution.

More examples of how to use ScuAugment are in folder with [examples](https://github.com/martinschatz-cz/SciAugment/tree/main/example_notebooks).

## To Do

- [x] Image augmentation
- [x] Channel augmentation
- [ ] Custom hyperparameter and augmentation settings YAML to overwrite YOLO settings
- [ ] Custem setup for channel and image augmentation
- [ ] Automatic setting export for easier reproducibility
- [ ] Import for reproducibility
- [ ] Default and custom settings for each augmentation.
- [ ] Manual for reading optimal values from imaged data.


# SciCount
SciAugment is part of [SciCount](https://github.com/martinschatz-cz/SciCount) project.

# How to cite

```
@MISC{Schatz2022-SciAugment,
  title     = "{martinschatz-cz/SciAugment}: v0.1.0",
  author    = "Sch{\"a}tz, Martin",
  abstract  = "The whole tool was rewritten as a class, and two options for
               prepared augmentations were set up. Everything is shown in the
               example Google Colab notebook on a small included annotated
               (YOLO) image dataset.",
  publisher = "Zenodo",
  year      =  2022
}
```
