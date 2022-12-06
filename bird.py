#!/usr/local/env python

#%%
# https://github.com/fmidev/opendata-resources/blob/master/examples/python/radar-rhi-from-hdf5.ipynb
# import xarray, h5py
# import os, boto3
import numpy as np
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axisartist.grid_finder import FixedLocator

# from osgeo import gdal
# import wradlib as wrl
# from wradlib.io.xarray_depr import CfRadial, OdimH5

# from birdcloud.birdcloud import BirdCloud
import pandas as pd
import open3d as o3d

# %%
def normalize_to_01(v):
    minimum_value = v.min()
    maximum_value = v.max()
    return (v - minimum_value) / (maximum_value - minimum_value)

def create_pcd_one_attribute(df, attribute):
    num_points = df.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].to_numpy())
    pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate(
            (normalize_to_01(df[[attribute]].to_numpy()), np.zeros((num_points, 2))),
            axis = 1
        )
    )
    return pcd

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])
# %%
df = pd.read_csv('data/22/NLDHL_pvol_20190322T0000_6234.csv.gz')
# Columns:
#   'elevation', 'azimuth', 'range', 'x', 'y', 'z', 'DBZH', 'DBZV', 'TH',
#   'TV', 'VRADH', 'VRADV', 'WRADH', 'WRADV', 'PHIDP', 'RHOHV', 'KDP',
#   'ZDR', 'biology'
attribute = 'biology'
pcd = create_pcd_one_attribute(df, attribute)

visualize_pcd(pcd)
# %%
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image('tmp.png')
vis.destroy_window()
# %%
