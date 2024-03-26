import fire
from pathlib import Path
from datetime import date, timedelta
from osgeo import gdal, ogr, gdalconst, osr
from sklearn.metrics import f1_score
import numpy as np
from tempfile import TemporaryDirectory
from tqdm import tqdm

'''
python evaluate.py PATH_TO_DETER_FILE.shp PATH_TO_TIFF_PREDICTIONS_VALUES.tif FIRST_DATE

Neste código, estou assumindo que os dados do tiff estão todos juntos, onde cada banda representa a predição para as próximas 2 semanas, semana a semana, a partir da data FIRST_DATE.

Por exemplo, se rodar:

python evaluate.py deter.shp predictions.tif 2023-11-27 (não precisa usar aspas)

supondo que o arquivo predictions.tif tem 3 bandas, estou assumindo que:
a banda 1 represente o desmatamento das próximas 2 semanas a contar de 27/11
a banda 2 represente o desmatamento das próximas de 2 semanas a contar de 4/12  e
a banda 3 represente o desmatamento das próximas de 2 semanas a contar de 11/12.
'''


def main(
        deter_shp_path: str,
        pred_tiff_path: str,
        from_date: str,
        #n_off_weeks: str

):
    thresholds = np.linspace(0, 1, 100)
    prediction_dataset = gdal.Open(pred_tiff_path, gdalconst.GA_ReadOnly)

    geo_transform = prediction_dataset.GetGeoTransform()
    x_res = prediction_dataset.RasterXSize
    y_res = prediction_dataset.RasterYSize
    crs = prediction_dataset.GetSpatialRef()
    proj = prediction_dataset.GetProjection()

    no_data_value = prediction_dataset.GetRasterBand(1).GetNoDataValue()

    prediction_data = prediction_dataset.ReadAsArray()

    mask_data_flatten = (prediction_data != no_data_value).flatten()

    if len(prediction_data.shape) == 2:
        prediction_data = np.expand_dims(prediction_data, axis=0)

    prediction_data = np.random.random(prediction_data.shape)

    deter_v = ogr.Open(deter_shp_path)
    deter_l = deter_v.GetLayer()

    true_data_array = np.empty_like(prediction_data)

    with TemporaryDirectory() as dir:
        first_date = date.fromisoformat(from_date)
        threshold_max = 0
        for i in range(prediction_data.shape[0]):
            current_date_begin = first_date + timedelta(days = 7 * i)
            current_date_end = current_date_begin + timedelta(days = 14)
            where = f'("VIEW_DATE" >= \'{str(current_date_begin)}\' AND "VIEW_DATE" < \'{str(current_date_end)}\') AND ("CLASSNAME" = \'MINERACAO\' OR "CLASSNAME" = \'DESMATAMENTO_CR\' OR "CLASSNAME" = \'DESMATAMENTO_VEG\')'
            deter_l.SetAttributeFilter(where)
            target_test = gdal.GetDriverByName('GTiff').Create(str(Path(dir)/f'test_{i}.tif'), x_res, y_res, 1, gdal.GDT_Byte)
            target_test.SetGeoTransform(geo_transform)
            target_test.SetSpatialRef(crs)
            target_test.SetProjection(proj)

            band = target_test.GetRasterBand(1)
            band.FlushCache()

            gdal.RasterizeLayer(target_test, [1], deter_l, burn_values=[1], options = ['ALL_TOUCHED=TRUE'])

            true_data_array[i] = target_test.ReadAsArray().copy()
            
            target_test = None

            temp_file = (Path(dir)/f'test_{i}.tif')
            temp_file.unlink()

    f1_max = 0
    
    for threshold in tqdm(thresholds):
        prediction_binary = (prediction_data > threshold).astype(np.int8)
        f1_i = f1_score(true_data_array.flatten()[mask_data_flatten], prediction_binary.flatten()[mask_data_flatten])
        if f1_i > f1_max:
            f1_max = f1_i
            threshold_max = threshold
        #print(f'{threshold} - {f1_i} - {f1_max}')
    print(f'Maximum Threshold: {threshold_max} - Maximum F1-Score: {f1_max}')

if __name__ == '__main__':
    fire.Fire(main)
