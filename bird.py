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
import open3d as o3d
import os
import pandas as pd
from myaux import *

# %%
# link = 'https://surfdrive.surf.nl/files/index.php/s/95EIQinJzj0JlV8/download?path=%2Fcsv%2FNL%2FDHL%2F2019%2F04%2F17'\
#     '&files=NLDHL_pvol_20190417T1000_6234.csv.gz'
# df = pd.read_csv(link, compression='gzip')
# df = pd.read_csv('data/22/NLDHL_pvol_20190322T0000_6234.csv.gz')
df = pd.read_csv('data/manual_annotations/NLDHL_pvol_20190417T2100_6234.h5.csv.gz')
df = df[0::10].reset_index()
df['ISBIRD'] = ((1 <= df.CLASS) & (df.CLASS <= 32)).astype(int)
#%%
# Columns:
#   'elevation', 'azimuth', 'range', 'x', 'y', 'z', 'DBZH', 'DBZV', 'TH',
#   'TV', 'VRADH', 'VRADV', 'WRADH', 'WRADV', 'PHIDP', 'RHOHV', 'KDP',
#   'ZDR', 'biology'
# BDZ* - reflectivity (filtered for echoes that are moving, DOppler shift)
# T* - reflectivity (unfiltered)
# VRAD* - Radial velocity
# WRAD* -
# PHIDP - Phase shift of the signal
# RHOHV - Correlation between horizontal and vertical pulses (important)
# KDP -
# ZDR - Difference
attribute = 'ISBIRD'
pcd = create_pcd_from_attributes(df, attribute, scale_z=10)
visualize_pcd(pcd)

#%%
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
df_clean = smooth_biology(
    df,
    pcd,
    pcd_tree,
    num_required_bio=10,
)
#%%
pcd_clean = create_pcd_from_attributes(df_clean, 'biology', 'new_biology', scale_z=10)
visualize_pcd(pcd_clean)
# [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[10500], 500)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 1]
# [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[10500], 10.0)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
#%%
if False:
    alpha = 0.5
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd,
        alpha,
    )
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# %%
attribute = 'ISBIRD'
suffix = "manual_annotations"
for filename in os.listdir(f"data/{suffix}"):
    df = pd.read_csv(os.path.join(f'data/{suffix}', filename))
    df['ISBIRD'] = ((1 <= df.CLASS) & (df.CLASS <= 32)).astype(int)

    pcd = create_pcd_from_attributes(df, attribute, scale_z=10)
    outname = filename.split('.')[0] + '.png'
    save_pcd_visualization(pcd, f'vis/{suffix}', outname)
# %%
