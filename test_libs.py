import torch
# from AmazonDataset import IbamaInpe25km_Dataset, IbamaDETER1km_Dataset
from pathlib import Path
import numpy as np
# from openstl.utils import create_parser, default_parser
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
# import geopandas as gpd
# from BaseExperiment import BaseExperiment
# from preprocess import reconstruct_time_patches, load_tif_image, load_npy_image
import os
import argparse

# try:
#     from osgeo import gdal
#     gdal.PushErrorHandler('CPLQuietErrorHandler')
# except ImportError:
#     print("osgeo module is not installed. Please install it with pip install GDAL")
from osgeo import gdal

if __name__ == '__main__':
    print('debug World')
    1/0