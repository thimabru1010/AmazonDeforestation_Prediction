import numpy as np
from osgeo import gdal

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

# # tif_path = 'data/DETER/deter_increments_1km_1week.tif'
# tif_path = 'data/IBAMA_INPE/1K/tiff_filled/mask.tif'

# tif_img = load_tif_image(tif_path)

# # np.save('data/DETER/deter_increments_1km_1week.npy', tif_img)
# np.save('data/IBAMA_INPE/1K/tiff_filled/mask.npy', tif_img)
# # data/IBAMA_INPE/1K/tiff_filled/mask.tif

preds = np.load('work_dirs/exp02/preds.npy')
print(preds.shape)

preds = preds[:, 1]
preds = preds.max(axis=1)

print(preds.shape)

# Save array in tif
def save_as_tif(array, file_path):
    """
    Save a numpy array as a TIFF file using GDAL.

    Parameters:
        array (numpy.ndarray): The numpy array to be saved.
        file_path (str): The file path where the TIFF file will be saved.
    """
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(file_path, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    
save_as_tif(preds, 'work_dirs/exp02/preds.tif')

