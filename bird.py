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

# %%
def normalize_to_01(v):
    minimum_value = v.min()
    maximum_value = v.max()
    return (v - minimum_value) / (maximum_value - minimum_value)

def create_pcd_one_attribute(df, attribute, attribute2=None, scale_z=1.0):
    num_points = df.shape[0]
    scale = np.array([1, 1, scale_z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].to_numpy() * scale)
    if attribute:
        if attribute2:
            pcd.colors = o3d.utility.Vector3dVector(
                np.concatenate(
                    (
                        normalize_to_01(df[[attribute]].to_numpy()),
                        normalize_to_01(df[[attribute2]].to_numpy()),
                        np.zeros((num_points, 1)),
                    ),
                    axis = 1
                )
            )
        else:
            pcd.colors = o3d.utility.Vector3dVector(
                np.concatenate(
                    (
                        normalize_to_01(df[[attribute]].to_numpy()),
                        np.zeros((num_points, 2)),
                    ),
                    axis = 1
                )
            )
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((num_points, 3)))
    return pcd

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def smooth_biology(df, pcd, pcd_tree, radius = 5.0, num_required_bio = 10):
    df_clean = df.copy()
    bio_idxs = df.index[df.biology == 1]
    count = 0
    df_clean['new_biology'] = df_clean.biology
    for i in bio_idxs:
        print(count / len(bio_idxs) * 100)
        count += 1
        print(i)
        [k, idx, _] = pcd_tree.search_radius_vector_3d(
            pcd.points[i],
            radius,
        )
        if sum(df_clean.loc[idx, 'biology'] == 1) < num_required_bio:
            df_clean.loc[i,'new_biology'] = 0
    return df_clean

# %%
link = 'https://surfdrive.surf.nl/files/index.php/s/95EIQinJzj0JlV8/download?path=%2Fcsv%2FNL%2FDHL%2F2019%2F04%2F17'\
    '&files=NLDHL_pvol_20190417T1000_6234.csv.gz'
# df = pd.read_csv('data/22/NLDHL_pvol_20190322T0000_6234.csv.gz')
df = pd.read_csv(link, compression='gzip')
df = df[0::10].reset_index()
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
attribute = 'biology'
pcd = create_pcd_one_attribute(df, attribute, scale_z=10)
visualize_pcd(pcd)

#%%
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
df_clean = smooth_biology(
    df,
    pcd,
    pcd_tree,
    num_required_bio=50,
)
#%%
pcd_clean = create_pcd_one_attribute(df_clean, 'biology', 'new_biology', scale_z=10)
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
for filename in os.listdir("data/22"):
    df = pd.read_csv(os.path.join('data/22', filename))
    pcd = create_pcd_one_attribute(df, attribute, scale_z=10)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    outname = filename.split('.')[0] + '.png'
    print(outname)
    vis.capture_screen_image(os.path.join('vis/22', outname))
    vis.destroy_window()
# %%
