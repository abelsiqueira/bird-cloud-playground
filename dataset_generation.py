#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
from myaux import *

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

#%%

def get_close_enough(df, p0, maximum_distance):
    xyz = ['x', 'y', 'z']
    return df[(df - p0).loc[:, xyz].apply(lambda row: np.linalg.norm(row), axis=1) < maximum_distance]

def generate_data(
    filename,
    num_points = 2**10,
    spatial_dimension = 3.0,
):
    df = pd.DataFrame({
        'x': np.random.rand(num_points) * spatial_dimension - spatial_dimension / 2,
        'y': np.random.rand(num_points) * spatial_dimension - spatial_dimension / 2,
        'z': np.random.rand(num_points) * spatial_dimension - spatial_dimension / 2,
        'useless_feature': np.random.randn(num_points),
    })

    # Randomnly aggregate some points
    maximum_distance = spatial_dimension / 5
    xyz = ['x', 'y', 'z']
    pull_strength = 0.3 # between 0 and 1

    for k in range(0, 20):
        i = np.random.randint(0, num_points)
        p0 = df.iloc[i,:]
        close_enough = get_close_enough(df, p0, maximum_distance)
        me = df.loc[close_enough.index, xyz]
        df.loc[close_enough.index, xyz] = me * (1 - pull_strength) + p0.loc[xyz] * pull_strength

    df['feat1'] = sigmoid(df.x + df.y - df.z)
    df['feat2'] = sigmoid(df.z - spatial_dimension / 2) - sigmoid(df.z + spatial_dimension / 2)
    df['feat3'] = np.exp(-0.3 * (df.x - 1)**2 - 0.2 * (df.y + 0.3)**2)
    hidden1 = (df.feat1 + df.feat2**2) / df.feat3
    hidden1 = (hidden1 - np.mean(hidden1)) / np.std(hidden1)
    hidden2 = np.log(1 + df.feat1**2) - np.sin(4*np.pi * df.feat2)
    hidden2 = (hidden2 - np.mean(hidden2)) / np.std(hidden2)
    df['neighbours'] = df.apply(lambda row: get_close_enough(df, row, 0.3).shape[0], axis=1)
    aux = (df['neighbours'] - np.mean(df['neighbours'])) / np.std(df['neighbours'])
    df['class'] = np.round(sigmoid(0.5 * hidden1 + 0.2 * hidden2 + 5 * aux))

    df.to_csv(filename)
    return df

# %%
df = generate_data('example_data.csv')
print(df['class'].value_counts())
pcd = create_pcd_from_attributes(df, 'class')
visualize_pcd(pcd)

# %%

# %%
df = generate_data('example_data2.csv')

# %%
