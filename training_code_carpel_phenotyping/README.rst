Wheat carpel phenotpying
========================

Training models
---------------

To train models:

a) Collect together training data. These should be image files with accompanying ImageJ/FIJI Regions of Interest (ROIs) which

a) Create a `dtool` dataset from the raw image data and ROI files (see https://dtool.readthedocs.io/en/latest/).

c) Run the `create_training_dataset_from_yaml.py` script. This script takes a YAML config file as an argument, examples are in the `runfiles` directory.

d) Install the `aiutils` python package from https://github.com/JIC-CSB/aiutils, then use the `train_unet.py` script.

Applying trained models to make measurements
--------------------------------------------

To apply the model:

a) Create a dtool dataset from the images on which measurements should be applied.

b) Run `analyse_from_yml.py`, with a YAML config file. Examples are in the `runfiles` directory.