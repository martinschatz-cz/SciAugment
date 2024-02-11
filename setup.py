import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='SciAugment',
    version='0.2.0',
    author='Martin Schätz',
    author_email='martin.schatz.cz@gmail.com',
    description='Package for augmentation of scientific images with YOLO annotations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/martinschatz-cz/SciAugment',
    project_urls = {
        "Bug Tracker": "https://github.com/martinschatz-cz/SciAugment/issues"
    },
    license='BSD-3-Clause license',
    packages=['SciAugment'],
    package_data={'SciAugment': ['SciAugment/SciAug_tools.py']},  # Include SciAug_tools.py
    install_requires=[
        'git+https://github.com/albu/albumentations',
        'opencv-python-headless>=3.4.18.65,<4.2',  # specify the version range
        'imgaug'
    ],
)
