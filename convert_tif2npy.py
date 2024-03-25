import numpy as np
from osgeo import gdal

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

# tif_path = 'data/DETER/deter_increments_1km_1week.tif'
tif_path = 'data/IBAMA_INPE/1K/tiff_filled/mask.tif'

tif_img = load_tif_image(tif_path)

# np.save('data/DETER/deter_increments_1km_1week.npy', tif_img)
np.save('data/IBAMA_INPE/1K/tiff_filled/mask.npy', tif_img)
# data/IBAMA_INPE/1K/tiff_filled/mask.tif