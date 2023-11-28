import numpy as np
import pandas as pd
import geopandas as gpd
import GiovConfig as config
from tqdm import tqdm
from GiovanniDataset import GiovanniDataset
import torch

def load_data(config):
    #! Loads data
    # load legal amazon limits
    am_bounds = gpd.read_file(config.AMAZON_FRONTIER_DATA)
    # load frames idx detail
    frames_idx = pd.read_csv(config.TR_FRAMES_IDX, index_col=0)
    # load frames deforestation area history
    deforestation = pd.read_csv(config.TR_DEFORESTATION, index_col=0)
    deforestation["quarter_date"] = pd.to_datetime(deforestation["quarter_date"])

    # counties
    frames_county = pd.read_csv(config.TR_COUNTIES, index_col=0)
    counties_defor = pd.read_csv(config.TR_COUNTIES_DEFOR, index_col=0)

    # precipitations
    precip = pd.read_csv(config.TR_RAIN_AVG)
    precip["quarter_date"] = pd.to_datetime(precip["dt"])

    # terrain position index
    tpi = pd.read_csv(config.TR_TPI, skiprows=1)\
        .rename(columns={"Unnamed: 0": "frame_id"})

    # past scores
    past_scores = pd.read_csv(config.TR_PAST_SCORES)
    past_scores["variable"] = pd.to_datetime(past_scores["variable"])

    # night lights
    night_light = pd.read_csv(config.TR_NIGHT_LIGHT)
    night_light["dt"] = pd.to_datetime(night_light["dt"])
    
    return am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light

def create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light):
    #! create limits history grid
    time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in enumerate(config.TIME_STEPS):
        defor_area = (
            deforestation[
                deforestation["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        time_grid[t, :, :] = defor_area.values.reshape(time_grid[0, :, :].shape)
    
    # Population and density
    county_data = np.zeros((2, frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    county_data[0] = (
        frames_county.set_index("frame_id")["populacao"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])

    county_data[1] = (
        frames_county.set_index("frame_id")["densidade"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])
        
    # Deforestation
    frames_counties_defor = pd.merge(
        counties_defor,
        frames_county[["frame_id", "county_id"]],
        on="county_id",
        how="right"
    )
    frames_counties_defor["quarter_date"] = pd.to_datetime(frames_counties_defor["quarter_date"])

    counties_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        defor_area = (
            frames_counties_defor[
                frames_counties_defor["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        counties_time_grid[t, :, :] = defor_area.values.reshape(counties_time_grid[0, :, :].shape)

    # Precipitations
    # create limits history grid
    precip_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        precip_sum = (
            precip[
                precip["quarter_date"] == dt
            ].set_index("frame_id")["precipitation"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        precip_time_grid[t, :, :] = precip_sum.values.reshape(counties_time_grid[0, :, :].shape)

    # Terrain
    cols = ["mean", "min", "max", "std"]
    tpi_array = np.zeros((len(cols), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for icol, col in enumerate(cols):
        v = (
            tpi.set_index("frame_id")[col] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        tpi_array[icol, :, :] = v.values.reshape(tpi_array[0, :, :].shape)

    # Past Scores
    # create history grid for scores
    scores_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        t_scores = (
            past_scores[
                past_scores["variable"] == dt
            ].set_index("frame_id")["value"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        scores_time_grid[t, :, :] = t_scores.values.reshape(scores_time_grid[0, :, :].shape)
        
    # Night Lights
    # create history grid for scores
    night_time_grid = np.zeros((2, len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        avg_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["avg_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[0, t, :, :] = avg_light.values.reshape(night_time_grid[0, 0, :, :].shape)
        
        max_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["max_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[1, t, :, :] = max_light.values.reshape(night_time_grid[0, 0, :, :].shape)

    print(time_grid.shape, county_data.shape, night_time_grid.shape)
    return time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid

def compute_frame_patches(frames_idx, deforestation, out_condition='both'):
    # Gather all the grids and creates input patches ? Also filters deforestation patches
    #! OBS: Alterate only the indexes, not the grids.
    # out_condition = "both"  # deforestation | borders | both
    bundle_step = 32
    patches = []
    for ix in tqdm(list(range(frames_idx["x"].min(), frames_idx["x"].max()+1, bundle_step))):
        fx = ix + config.INPUT_BOXES_SIZE
        for iy in range(frames_idx["y"].min(), frames_idx["y"].max()+1, bundle_step):
            fy = iy + config.INPUT_BOXES_SIZE

            iframes = frames_idx[
                (frames_idx["x"] >= ix) & 
                (frames_idx["x"] < fx) &
                (frames_idx["y"] >= iy) &
                (frames_idx["y"] < fy)
            ]
            
            if out_condition == "borders":
                if iframes["in_borders"].mean() >= 0.5:  # condition: bundle has to be at least half inside borders
                    patches.append(iframes.index)
                    
            elif out_condition == "deforestation":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if out_of_borders_frames < len(iframes):  # condition: bundle has to contain some deforestation
                    patches.append(iframes.index) 

            elif out_condition == "both":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if (out_of_borders_frames < len(iframes)) and (iframes["in_borders"].mean() >= 0.5):
                    patches.append(iframes.index)
    return patches

def normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid):
    # Data normalization
    # Normalizes population data
    
    pop_median = np.median(county_data[0, :, :])
    pop_std = 1e5
    den_median = np.median(county_data[1, :, :])
    den_std = 30
    norm_pop = (county_data[0, :, :] - pop_median) / pop_std
    norm_den = (county_data[1, :, :] - np.median(county_data[1, :, :])) / 30

    county_data[0, :, :] = norm_pop
    county_data[1, :, :] = norm_den

    # Normalizies counties and precipitation grids
    counties_time_grid = (counties_time_grid-counties_time_grid[train_time_idx, :, :].mean()) / counties_time_grid[train_time_idx, :, :].std()
    precip_time_grid = (precip_time_grid-precip_time_grid[train_time_idx, :, :].mean()) / precip_time_grid[train_time_idx, :, :].std()

    # Normalizes tpi ?
    for i in range(tpi_array.shape[0]):
        tpi_array[i, :, :] = (tpi_array[i, :, :] - tpi_array[i, :, :].mean()) / tpi_array[i, :, :].std()

    # Normalizes Night Images
    for i in [0, 1]:
        s = (
            (
                night_time_grid[i, :, :, :] - 
                night_time_grid[i, train_time_idx, :, :].mean()
            ) / night_time_grid[i, train_time_idx, :, :].std()
        )
        s[np.where(s > 3)] = 3
        night_time_grid[i, :, :, :] = s.copy()
    return county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid

def prep4dataset(config):
    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    train_time_idx = range(12)
    val_time_idx = range(12, 20)
    # test_time_idx = range(20,28) #! Not used yet
    train_data = time_grid[train_time_idx, :, :]
    val_data = time_grid[val_time_idx, :, :]
    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid)
    patches = compute_frame_patches(frames_idx, deforestation, out_condition='both')
    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]
    patches_sample_train = patches
    patches_sample_val = patches
    return train_data, val_data, patches_sample_train, patches_sample_val, frames_idx, county_data, counties_time_grid, \
        precip_time_grid, tpi_array, scores_time_grid, night_time_grid

def prep4dataset_test(config):
    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    test_time_idx = range(20,28)
    test_data = time_grid[test_time_idx, :, :]
    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, test_time_idx, precip_time_grid, tpi_array, night_time_grid)
    patches = compute_frame_patches(frames_idx, deforestation, out_condition='both')
    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]
    patches_sample_train = patches
    patches_sample_val = patches
    return test_data, patches_sample_train, patches_sample_val, frames_idx, county_data, counties_time_grid, \
        precip_time_grid, tpi_array, scores_time_grid, night_time_grid
        
if __name__=='__main__':
    # set device to GPU
    # dev = "cuda:0"

    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    
    time_grid, county_data, frames_counties_defor, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    
    # remove patches that represent reduced regions
    # patches = [b for b in patches if (len(b)==len(patches[0]))]

    train_time_idx = range(12)
    val_time_idx = range(12, 20)
    test_time_idx = range(20,28)

    train_data = time_grid[train_time_idx, :, :]
    val_data = time_grid[val_time_idx, :, :]

    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid)

    patches = compute_frame_patches(frames_idx, out_condition='both')

    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]

    print(train_data.shape, val_data.shape)
    print(len(patches))

    patches_sample_train = patches
    patches_sample_val = patches

    train_set = GiovanniDataset(
        train_data, 
        patches_sample_train, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=dev
    )

    val_set = GiovanniDataset(
        val_data, #previously = test_data
        patches_sample_val, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=dev
    )

    trainloader = torch.utils.data.DataLoader(train_set)
    valloader = torch.utils.data.DataLoader(val_set)

    for inputs, labels in trainloader:
        print(inputs.shape, labels.shape)
        print(labels[0])
        break

    # from segmentation_models_pytorch.losses import FocalLoss

    # f_loss = FocalLoss("binary", gamma=3).to(dev)

    # # baseline: all zero
    # base_train_err = 0
    # for inputs, labels in trainloader:
    #     y_pred = torch.tensor(np.zeros(labels.shape)).to(dev)
    #     y_pred[:, 0, :, :] = 1
    #     print(y_pred.shape, labels.shape)
    #     # base_train_err += ce_loss(input=y_pred, target=labels)
    #     base_train_err += f_loss(y_pred=y_pred, y_true=labels)
    # base_train_err = base_train_err / len(trainloader)

    # print(f"Baseline Error (Train) = {base_train_err:.6f}")