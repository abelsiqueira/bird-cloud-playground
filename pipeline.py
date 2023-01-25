#%%
import numpy as np
import pandas as pd
import open3d as o3d
from myaux import *

#%%
def create_tree_from_point_cloud(pcd):
    tree = o3d.geometry.KDTreeFlann(pcd)
    return tree

def neighbours(point, tree, radius):
    [num_neighbours, idx, distances] = tree.search_radius_vector_3d(point, radius)
    return num_neighbours, idx, distances

def find_points_with_enough_neighbours(tree, radius, min_neighbors):
    output = {}
    for (i, p) in enumerate(pcd.points):
        nn, idx, dists = neighbours(p, tree, radius)
        if nn >= min_neighbors:
            output[i] = {'indexes': idx, 'distances': dists}
    return output

def define_label_from_neighbourhood(df, core_point, target, weight_decay = 1.0, cutoff = 0.5):
    idx = core_point['indexes']
    target_values = df.loc[idx,target].values
    weights = np.exp(-weight_decay * np.array(core_point['distances']))
    core_point['prob_' + target] = np.sum(weights * target_values) / np.sum(weights)
    core_point[target] = (core_point['prob_' + target] > cutoff).astype('int32')

    return

def define_label_from_neighbourhood_for_all(df, core_points, target, *kwargs):
    for c in core_points:
        define_label_from_neighbourhood(df, core_points[c], target, *kwargs)

# %%
df = pd.read_csv('example_data1.csv')

#%%
pcd = create_pcd_from_attributes(df, 'class')
visualize_pcd(pcd)
#%%
tree = create_tree_from_point_cloud(pcd)

# %%
neighbours(pcd.points[0], tree, 0.5)
# %%
core_points = find_points_with_enough_neighbours(tree, 0.3, 10)
# %%
for i in core_points:
    np.asarray(pcd.colors)[core_points[i]['indexes'], 1:] = [0.5, 0.8]

# %%
define_label_from_neighbourhood_for_all(df, core_points, 'class', weight_decay = 0.5)

# %%
visualize_pcd(pcd)

# %%
