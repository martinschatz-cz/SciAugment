#@title
import albumentations as A
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import os
from xml.dom import minidom
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import copy
import glob

# Functions

def read_image(images_path: str, filename: str):
    """
    Uses OpenCV to read RGB images
    :param images_path (str):
    :param filename (str):
    :return: RGB image
    """

    # OpenCV uses BGR channels
    img = cv2.imread(images_path + filename)
    return img


def read_yolo(filename: str) -> float:
    """
    Reads YOLO type of object anotation for albumentation to read
    :param filename (str):
    :return yolo_coords:
    """
    yolo_coords = []
    with open(filename, 'r') as fname:
        for yolo in fname:
            x = yolo.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            # print(x)
            yolo_coords.append(x)
    return yolo_coords


def write_yolo(coords: float, name: str):
    """
    Write YOLO type object coordinates to txt file
    :param coords (float): list of coordinates
    :param name (str): name of file to write to
    :return:
    """
    with open(name + '.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))
    return 0


def get_transform(loop: int):
    """
    Generating image an label augmentation for YOLO with name tag of augmentations.

    :param loop (int): Number of augmentation from 0 to 8

    :return:
    A: transform function with BboxParams set for Yolo
    name (string): Name where relevant to position is bit length 11 for:
      1:Shift
      2:Scale
      3:Rotate
      4:VerticalFlip
      5:HorizontalFlip
      6:RandomBrightnessContrast
      7:MultiplicativeNoise(multiplier=0.5, p=0.2)
      8:RandomSizedBBoxSafeCrop (250, 250, erosion_rate=0.0, interpolation=1, p=1.0)
      9:Blur(blur_limit=(50, 50), p=0)
      10:Transpose
      11:RandomRotate90
    """
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00001000000'
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000100000'
    elif loop == 2:
        transform = A.Compose([
            A.HorizontalFlip(p=0),
            A.MultiplicativeNoise(multiplier=0.5, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000010000'
    elif loop == 3:
        transform = A.Compose([
            # A.CenterCrop(width=250, height=250, p=1)
            A.RandomSizedBBoxSafeCrop(250, 250, erosion_rate=0.0, interpolation=1, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000001000'
    elif loop == 4:
        transform = A.Compose([
            A.Blur(blur_limit=(50, 50), p=0)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000000100'
    elif loop == 5:
        transform = A.Compose([
            A.Transpose(1)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000000010'
    elif loop == 6:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00000000001'
    elif loop == 7:
        transform = A.Compose([
            A.ShiftScaleRotate(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '11100000000'
    elif loop == 8:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
        name = '00010000000'
    # elif loop == 7:
    #     transform = A.Compose([
    #         A.ImageCompression(quality_lower=0, quality_upper=1, p=0.2)
    #     ], bbox_params=A.BboxParams(format='yolo'))
    # elif loop == 8:
    #     transform = A.Compose([
    #         A.CoarseDropout(max_holes=50, max_height=40,
    #                  max_width=40, fill_value=128, p=0)
    #     ], bbox_params=A.BboxParams(format='pascal_voc'))

    # transform = A.Compose([
    #         A.HorizontalFlip(p=1),
    #     ], bbox_params=A.BboxParams(format='yolo'))

    return transform, name

def create_train_data_folder():
  try:
    dir = 'train_data'
    # !mkdir $dir
    os.mkdir(dir)
    dir_image_train = 'train_data/images/train'
    dir_image_val = 'train_data/images/val'
    dir_label_train = 'train_data/labels/train'
    dir_label_val = 'train_data/labels/val'

    # !mkdir 'train_data/images'
    # !mkdir 'train_data/labels'
    # !mkdir $dir_image_train
    # !mkdir $dir_image_val
    # !mkdir $dir_label_train
    # !mkdir $dir_label_val

    os.mkdir('train_data/images')
    os.mkdir('train_data/labels')
    os.mkdir(dir_image_train)
    os.mkdir(dir_image_val)
    os.mkdir(dir_label_train)
    os.mkdir(dir_label_val)

    return 0
  except:
    print("Error while creating train_data folder structure!")
    return 1


# rozdelit na train a test!!
def augment_data(images_path: str):
  #, train:float = 0.7, image_format:str = ".png"
    """
    Augment input images and YOLO files as defined in get_transform() function and save then in train_dir prepared for training and test/val based on 70/30 rule.
    :param images_path (str):
    :param train (float): train/val parameter for generating training data set, default 70% train 30% val
    :param image_format (str): image format name ('.png' default)
    :return:
    """
    count = 0
    # dir = 'train_data'
    # !mkdir $dir
    dir_image_train = 'train_data/images/train'
    dir_image_val = 'train_data/images/val'
    dir_label_train = 'train_data/labels/train'
    dir_label_val = 'train_data/labels/val'

    create_train_data_folder()

    # !mkdir 'train_data/images'
    # !mkdir 'train_data/labels'
    # !mkdir $dir_image_train
    # !mkdir $dir_image_val
    # !mkdir $dir_label_train
    # !mkdir $dir_label_val

    image_format = ".png"
    train = 0.7
    test = 1 - train
    files_to_process = sorted(os.listdir(images_path))
    print('Num of files: ' + str(len(files_to_process)))
    # udelat sort souboru!!!

    for filename in files_to_process:
        print('Processing: ' + filename)

        if filename.endswith(image_format.lower()) or filename.endswith(image_format.upper()):
            title, ext = os.path.splitext(os.path.basename(filename))
            image = read_image(images_path, filename)
        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = read_yolo(images_path + xmlTitle + '.txt')
                print(images_path + xmlTitle + '.txt')
                for i in range(0, 9):
                    img = copy.deepcopy(image)
                    transform, name_tag = get_transform(i)
                    dice = random.uniform(0, 1)
                    try:
                        transformed = transform(image=img, bboxes=bboxes)
                        transformed_image = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        name = title + '_' + str(count) + '_' + name_tag + '.jpg'
                        # print(name)
                        if dice <= train:
                            p_name = '/content/' + dir_image_train + '/' + name
                        else:
                            p_name = '/content/' + dir_image_val + '/' + name

                        cv2.imwrite(p_name, transformed_image)
                        print('Writing ' + name)
                        # print(transformed_bboxes)
                        # writeVoc(transformed_bboxes, count, transformed_image)
                        # pTitle='/content/'+dir+'/'+title
                        if dice <= train:
                            p_title = '/content/' + dir_label_train + '/' + title + '_' + str(count) + '_' + name_tag
                        else:
                            p_title = '/content/' + dir_label_val + '/' + title + '_' + str(count) + '_' + name_tag

                        write_yolo(transformed_bboxes, p_title)
                        count = count + 1
                    except:
                        print("Bounding Box exception!!!")
                        pass

                # bboxes = [[int(float(j)) for j in i] for i in bb]
