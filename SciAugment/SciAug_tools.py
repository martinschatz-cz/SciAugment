#SciAug_tools
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
from sys import stdout
from shutil import rmtree


class SciAugment:
  version = '0.2.0'
  source = 'https://github.com/martinschatz-cz/SciAugment'
  author = 'Martin Schätz'
  aug_type = ''
  augment = []
  channel_augment = []
  aug_dict = {-1:'no augmentation',
              0:'HorizontalFlip(p=1)',
              1:'RandomBrightnessContrast(contrast_limit=0.2,p=1)',
              2:'MultiplicativeNoise(multiplier=0.5, p=1)',
              3:'RandomSizedBBoxSafeCrop(250, 250, erosion_rate=0.0, interpolation=1, p=1.0)',
              4:'Blur(blur_limit=(10, 10), p=0)',
              5:'Transpose(1)',
              6:'RandomRotate90(p=1)',
              7:'ShiftScaleRotate(p=1)',
              8:'VerticalFlip(p=1)',
              9:'RandomBrightnessContrast(brightness_limit=0.2,p=1)'
              }

  aug_channel_dict = {-1:'no augmentation',
                      0:'RandomBrightnessContrast(contrast_limit=0.2,p=1)',
                      1:'MultiplicativeNoise(multiplier=0.5, p=1)',
                      2:'Blur(blur_limit=(10, 10), p=0)',
                      3:'RandomBrightnessContrast(brightness_limit=0.2,p=1)',
                      4:'Superpixels (p_replace=0.1, n_segments=20, max_size=64, interpolation=1, p=1)',
                      5:'GaussNoise (var_limit=(10.0, 50.0), mean=0, p=1)',
                      }

#######general functions###########
  def __new__(cls, *args, **kwargs):
        print("New instance of SciAugment.")
        return super().__new__(cls)

  def __init__(self, aug_type:str = 'Default', channel_aug:bool = False):
      self.augmet_type = aug_type;
      print("Selected augmentation type: {}" .format(self.augmet_type))
      if aug_type == 'Default':
        self.augment = [-1,0,3,5,6,7,8]
        if channel_aug:
          self.channel_augment = range(-1,5,1)

      if aug_type == 'fluorescece_microscopy':
        self.augment = range(-1,9,1)
        if channel_aug:
          self.channel_augment = [-1,1,2,3,4,5]

      if aug_type == 'all':
        self.augment = range(-1,9,1)
        if channel_aug:
          self.channel_augment = range(-1,5,1)

      if aug_type == 'no_augment':
        self.augment = [-1]
        print('No augment setting will only divide images and labels to train_data folder.')
        self.channel_augment = []

      print('\n')
      self.explain()

  def explain(self):
    print('Version: {}'.format(self.version))
    print('\n')
    print('Selected augmentation:')
    for aug in self.augment:
      print(self.aug_dict[aug])

    if self.channel_augment:
      print('\n')
      print('Selected channel wise augmentation:')
      for ch_aug in self.channel_augment:
        print(self.aug_channel_dict[ch_aug])


  def info(self):
    print('Version: {}'.format(self.version))
    print('\n')
    print('Selected augmentation:')
    for aug in self.augment:
      print(self.aug_dict[aug])
    
    if self.channel_augment:
      print('\n')
      print('Selected channel wise augmentation:')
      for ch_aug in self.channel_augment:
        print(self.aug_channel_dict[ch_aug])

    print('\n')
    print('\n')
    print('Source: {}'.format(self.source))
    print('Author: {}'.format(self.author))


  def read_image(self, images_path: str, filename: str):
    """
    Uses OpenCV to read RGB images
    :param images_path (str):
    :param filename (str):
    :return: RGB image
    """

    # OpenCV uses BGR channels
    img = cv2.imread(images_path + filename)
    return img

  def read_image_unchanged(self, images_path: str, filename: str):
    """
    Uses OpenCV to read RGBA or 4 channel (maximum) images
    :param images_path (str):
    :param filename (str):
    :return: RGB image
    """
    # https://learnopencv.com/read-an-image-in-opencv-python-cpp/
    # OpenCV uses BGR channels
    img = cv2.imread(images_path + filename, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    return img


  def read_yolo(self, filename: str) -> float:
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


  def write_yolo(self, coords: float, name: str):
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

  def create_train_data_folder(self):
    try:
      dir = 'train_data'
      if os.path.exists(dir):
        print('Train_data folder already exists!') #, and will be removed.')
        return 1
        # shutil.rmtree(dir)
      else:
        os.mkdir(dir)

      # info_path = dir + '/info.txt'
      # with open(info_path, 'w') as f:
      #   f.writelines(self.info())

      dir_image_train = 'train_data/images/train'
      dir_image_val = 'train_data/images/val'
      dir_label_train = 'train_data/labels/train'
      dir_label_val = 'train_data/labels/val'

      os.mkdir('train_data/images')
      os.mkdir('train_data/labels')
      os.mkdir(dir_image_train)
      os.mkdir(dir_image_val)
      os.mkdir(dir_label_train)
      os.mkdir(dir_label_val)

      return 0
    except:
      print("Error while creating new train_data folder structure!")
      return 1

  #########transform functions########
  def get_transform(loop: int):
      """
      !!!OLD FUNCTION, DO NOT USE!!!
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
        6:RandomContrast
        7:MultiplicativeNoise(multiplier=0.5, p=1)
        8:RandomSizedBBoxSafeCrop (250, 250, erosion_rate=0.0, interpolation=1, p=1.0)
        9:Blur(blur_limit=(10, 10), p=0)
        10:Transpose
        11:RandomRotate90
        12:RandomBrightness
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
              A.MultiplicativeNoise(multiplier=0.5, p=1),
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
              A.Blur(blur_limit=(10, 10), p=1)
          ], bbox_params=A.BboxParams(format='yolo'))
          name = '00000000100'
      elif loop == 5:
          transform = A.Compose([
              A.Transpose(p=11)
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


  def _h_flip():
      transform = A.Compose([
          A.HorizontalFlip(p=1),
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000010000000'
      return transform, name


  def _rand_contrast():
      transform = A.Compose([
          A.RandomBrightnessContrast(contrast_limit=0.2,p=1),
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000001000000'
      return transform, name


  def _multi_noise():
      transform = A.Compose([
          A.HorizontalFlip(p=0),
          A.MultiplicativeNoise(multiplier=0.5, p=1),
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000100000'
      return transform, name


  def _rand_size_crop():
      transform = A.Compose([
          # A.CenterCrop(width=250, height=250, p=1)
          A.RandomSizedBBoxSafeCrop(250, 250, erosion_rate=0.0, interpolation=1, p=1.0)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000010000'
      return transform, name


  def _im_blur():
      transform = A.Compose([
          A.Blur(blur_limit=(10, 10), p=1)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000001000'
      return transform, name


  def _im_transpose():
      transform = A.Compose([
          A.Transpose(p=1)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000000100'
      return transform, name


  def _rand_rotate():
      transform = A.Compose([
          A.RandomRotate90(p=1)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000000010'
      return transform, name


  def _shift_scale_rotate():
      transform = A.Compose([
          A.ShiftScaleRotate(p=1)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '111000000000'
      return transform, name


  def _v_flip():
      transform = A.Compose([
          A.VerticalFlip(p=1)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000100000000'
      return transform, name

  def _no_augment():
      transform = A.Compose([
          A.VerticalFlip(p=0)
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000000000'
      return transform, name

  def _rand_brightness():
      transform = A.Compose([
          A.RandomBrightnessContrast(brightness_limit=0.2, p=1),
      ], bbox_params=A.BboxParams(format='yolo'))
      name = '000000000001'
      return transform, name

  # for 16 bit A.ToFloat(max_value=65535.0),
  #     """
  #       1:Shift
  #       2:Scale
  #       3:Rotate
  #       4:VerticalFlip
  #       5:HorizontalFlip
  #       6:RandomBrightnessContrast
  #       7:MultiplicativeNoise(multiplier=0.5, p=1)
  #       8:RandomSizedBBoxSafeCrop (250, 250, erosion_rate=0.0, interpolation=1, p=1.0)
  #       9:Blur(blur_limit=(10, 10), p=0)
  #       10:Transpose
  #       11:RandomRotate90
  #     """

  aug_functions = {-1:_no_augment,
                    0:_h_flip,
                    1:_rand_contrast,
                    2:_multi_noise,
                    3:_rand_size_crop,
                    4:_im_blur,
                    5:_im_transpose,
                    6:_rand_rotate,
                    7:_shift_scale_rotate,
                    8:_v_flip,
                    9:_rand_brightness,
                    }
# add functions
  def _no_augment_ch():
      transform = A.Compose([
          A.VerticalFlip(p=0)
      ])
      name = '-NA'
      return transform, name

  def _rand_contrast_ch():
      transform = A.Compose([
          A.RandomBrightnessContrast(contrast_limit=0.2,p=1)
      ])
      name = '-RC'
      return transform, name

  def _multi_noise_ch():
      transform = A.Compose([
          A.MultiplicativeNoise(multiplier=0.5, p=1)
      ])
      name = '-MN'
      return transform, name

  def _im_blur_ch():
      transform = A.Compose([
          A.Blur(blur_limit=(10, 10), p=1)
      ])
      name = '-B'
      return transform, name

  def _rand_brightness_ch():
      transform = A.Compose([
          A.RandomBrightnessContrast(brightness_limit=0.2,p=1)
      ])
      name = '-RB'
      return transform, name

  def _superpixels_ch():
      transform = A.Compose([
          A.Superpixels (p_replace=0.1, n_segments=20, max_size=64, interpolation=1, p=1)
          ])
      name = '-SP'
      return transform, name

  def _gauss_noise_ch():
      transform = A.Compose([
          A.GaussNoise (var_limit=(10.0, 50.0), mean=0, p=1)
          ])
      name = '-GN'
      return transform, name

  channel_aug_functions = {-1:_no_augment_ch,
                            0:_rand_contrast_ch,
                            1:_multi_noise_ch,
                            2:_im_blur_ch,
                            3:_rand_brightness_ch,
                            4:_superpixels_ch,
                            5:_gauss_noise_ch,
                          }


  #####apply functions######
  def augment_data(self, images_path: str, train:float = 0.7, image_format:str = ".png", output_image_format:str = ".jpg"):
    """
    Augment input images and YOLO files as defined in get_transform() function and save then in train_dir prepared for training and test/val based on 70/30 rule.
    :param images_path (str):
    :param train (float): train/val parameter for generating training data set, default 70% train 30% val
    :param image_format (str): image format name ('.png' default)
    :return:
    """
    count = 0
    # dir = 'train_data'

    dir_image_train = 'train_data/images/train'
    dir_image_val = 'train_data/images/val'
    dir_label_train = 'train_data/labels/train'
    dir_label_val = 'train_data/labels/val'

    folder_err = self.create_train_data_folder()
    if folder_err:
      print('Please remove existing trin_data folder and try again.')
      return 1

    # image_format = ".png"
    # train = 0.7
    test = 1 - train
    files_to_process = sorted(os.listdir(images_path))
    print('Num of files: ' + str(len(files_to_process)))
    # udelat sort souboru!!!

    for filename in files_to_process:
        print('Processing: ' + filename)

        if filename.endswith(image_format.lower()) or filename.endswith(image_format.upper()):
            title, ext = os.path.splitext(os.path.basename(filename))
            print(images_path)
            print(filename)
            image = self.read_image(images_path, filename)
        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = self.read_yolo(images_path + xmlTitle + '.txt')
                print(images_path + xmlTitle + '.txt')
                for aug in self.augment:
                    img = copy.deepcopy(image)
                    transform, name_tag = self.aug_functions[aug]()
                    dice = random.uniform(0, 1)
                    #try:
                    transformed = transform(image=img, bboxes=bboxes)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    name = title + '_' + str(count) + '_' + name_tag + output_image_format
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

                    self.write_yolo(transformed_bboxes, p_title)
                    count = count + 1
                    #except:
                        #print("Bounding Box exception!!!")
                        #pass
    print('Process created {} images'.format(count))

  def augment_data_per_channel(self, images_path: str, train:float = 0.7, image_format:str = ".png", output_image_format:str = ".png"):
    """
    Augment input images and YOLO files as defined in get_transform() function and save then in train_dir prepared for training and test/val based on 70/30 rule.
    :param images_path (str):
    :param train (float): train/val parameter for generating training data set, default 70% train 30% val
    :param image_format (str): image format name ('.png' default)
    :return:
    """
    count = 0
    # dir = 'train_data'

    dir_image_train = 'train_data/images/train'
    dir_image_val = 'train_data/images/val'
    dir_label_train = 'train_data/labels/train'
    dir_label_val = 'train_data/labels/val'

    folder_err = self.create_train_data_folder()
    if folder_err:
      print('Please remove existing trin_data folder and try again.')
      return 1

    # image_format = ".png"
    # train = 0.7
    test = 1 - train
    files_to_process = sorted(os.listdir(images_path))
    print('Num of files: ' + str(len(files_to_process)))

    for filename in files_to_process:
        print('Processing: ' + filename)

        if filename.endswith(image_format.lower()) or filename.endswith(image_format.upper()):
            title, ext = os.path.splitext(os.path.basename(filename))
            print(images_path)
            print(filename)
            image = self.read_image_unchanged(images_path, filename)
        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = self.read_yolo(images_path + xmlTitle + '.txt')
                print(images_path + xmlTitle + '.txt')
                for aug in self.augment:
                    img = copy.deepcopy(image)
                    transform, name_tag = self.aug_functions[aug]()
                    dice = random.uniform(0, 1)
                    #####apply also channel augmentation#####
                    # try:
                    transformed = transform(image=img, bboxes=bboxes)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    # go through channel transports
                    for ch_aug in self.channel_augment:
                      
                      transform_ch, ch_name_tag = self.channel_aug_functions[ch_aug]()

                      # split channels
                      im_split = cv2.split(transformed_image)

                      for ch in range(0,len(im_split),1):
                        channel=im_split[ch]

                        transformed = transform_ch(image=channel)
                        transformed_channel = transformed['image']

                        im_merge=copy.deepcopy(im_split)
                        im_merge[ch]=transformed_channel
                        merged_transformed = cv2.merge(im_merge)
                        name_tag_ch='_ch-' + str(ch+1) + ch_name_tag
                        # print(name_ch)
                        # cv2_imshow(merged)
                        #save
                        name = title + '_' + str(count) + '_' + name_tag + name_tag_ch 
                        # print(name)
                        if dice <= train:
                            p_name = '/content/' + dir_image_train + '/' + name + output_image_format
                        else:
                            p_name = '/content/' + dir_image_val + '/' + name + output_image_format

                        cv2.imwrite(p_name, merged_transformed)
                        print('Writing ' + name)
                        # print(transformed_bboxes)
                        # writeVoc(transformed_bboxes, count, transformed_image)
                        # pTitle='/content/'+dir+'/'+title
                        if dice <= train:
                            p_title = '/content/' + dir_label_train + '/' + name
                        else:
                            p_title = '/content/' + dir_label_val + '/' + name

                        self.write_yolo(transformed_bboxes, p_title)
                        count = count + 1
                    # except:
                        # print("Augmentation exception!!!")
                        # pass
    print('Process created {} images'.format(count))
