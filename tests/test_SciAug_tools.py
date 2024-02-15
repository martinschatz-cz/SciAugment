# test_SciAug_tools.py
import os

import SciAugment as SA
from SciAugment.SciAug_tools import *


def test_version():
    version = SA.__version__
    print("SciAugment version:", version)
    assert isinstance(
        version, str
    ), f"Expected result to be of type str, but got {type(result)}"


def test_augment_data():
    # Ensure that the example images are downloaded and extracted
    # os.system("wget https://github.com/martinschatz-cz/SciAugment/raw/main/example_notebooks/subsection320.zip")
    # os.system("unzip -q ./subsection320.zip -d ./")

    # Define input parameters
    input_images_folder = "./"
    input_image_format = ".jpeg"

    # Instantiate SciAugment and perform augmentation
    aug1 = SciAugment()
    aug1.augment_data(images_path=input_images_folder, image_format=input_image_format)

    # Add assertions to check the outcome of the augmentation process
    # assert os.path.exists("./train_data/"), "Augmented images folder does not exist"
    # assert len(os.listdir("./train_data/")) > 0, "No augmented images generated"

    # Cleanup: remove downloaded and extracted files
    # os.remove("./subsection320.zip")
    # os.system("rm -rf ./subsection320/")
