#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
from scipy.spatial import KDTree
from math import ceil
from myaux import *

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# def get_close_enough(tree, p0, maximum_distance):
#     xyz = ['x', 'y', 'z']
#     return df[(df - p0).loc[:, xyz].apply(lambda row: np.linalg.norm(row), axis=1) < maximum_distance]

def polar_to_cartesian(radius, azimuth, elevation):
    """Simplified polar to cartesian transformation"""
    x = radius * np.cos(np.pi / 180 * elevation) * np.sin(np.pi / 180 * azimuth)
    y = radius * np.cos(np.pi / 180 * elevation) * np.cos(np.pi / 180 * azimuth)
    z = radius * np.sin(np.pi / 180 * elevation) + 343

    return x, y, z

def generate_data(
    filename,
    num_points = 2**13,
    max_range = 300000,
    add_na = False,
):
    azimuth_skip = 2**1
    elevations = np.array([0.3, 0.8, 1.2, 2, 2.8, 4.5, 6, 8, 10, 12, 15, 20, 25])
    df = pd.DataFrame({
        'range': 50 + np.exp(-5 * np.random.rand(num_points)) * max_range,
        'azimuth': np.tile(np.arange(0, 360, azimuth_skip) + 0.5, ceil(num_points / 360 * azimuth_skip))[0:num_points],
        'elevation': sorted(np.tile(elevations, ceil(num_points / len(elevations)))[0:num_points]),
        'useless_feature': np.random.randn(num_points),
    })

    df['x'], df['y'], df['z'] = polar_to_cartesian(df.range, df.azimuth, df.elevation)

    # Randomnly aggregate some points
    # maximum_distance = max_range / 5
    xyz = ['x', 'y', 'z']
    # pull_strength = 0.3 # between 0 and 1
    tree = KDTree(df.loc[:,xyz])

    # for k in range(0, 20):
    #     i = np.random.randint(0, num_points)
    #     p0 = df.iloc[i,:]
    #     close_enough = get_close_enough(df, p0, maximum_distance)
    #     me = df.loc[close_enough.index, xyz]
    #     df.loc[close_enough.index, xyz] = me * (1 - pull_strength) + p0.loc[xyz] * pull_strength

    df['feat1'] = sigmoid((df.x + df.y - df.z) / max_range)
    df['feat2'] = sigmoid((df.z - df.z.mean()) / df.z.std())
    df['feat3'] = (np.cos(np.exp(-0.3 * (df.x - 1)**2 - 0.2 * (df.y + 0.3)**2)) + 1) / 2
    hidden1 = (df.feat1 + df.feat2**2) / df.feat3
    hidden1 = (hidden1 - np.mean(hidden1)) / np.std(hidden1)
    hidden2 = np.log(1 + df.feat1**2) - np.sin(4*np.pi * df.feat2)
    hidden2 = (hidden2 - np.mean(hidden2)) / np.std(hidden2)


    # distance_matrix = tree.sparse_distance_matrix(tree, 300)
    df['neighbours'] = df.apply(lambda row: len(tree.query_ball_point(row[['x', 'y', 'z']], 300)), axis=1)
    # df['neighbours'] = df.apply(lambda row: (distance_matrix[row.name,:] > 0).sum(), axis=1)
    aux = (df['neighbours'] > 5).astype('int32')
    df['class'] = np.round(sigmoid(0.5 * hidden1 + 0.2 * hidden2 + 3 * aux))

    if add_na:
        df.loc[np.random.randint(0, num_points, num_points // 100), 'feat2'] = None
        df.loc[np.random.randint(0, num_points, num_points // 20), 'feat2'] = None

    df.to_csv(filename)
    return df

df_examples = generate_data('example_data1.csv')
print(df_examples['class'].value_counts())
print(df_examples.neighbours.value_counts())
pcd = create_pcd_from_attributes(df_examples, 'class', scale_z=2)
visualize_pcd(pcd)

# %%
df_examples = generate_data('example_data2.csv', add_na=True)

# %%
