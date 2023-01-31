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

def create_edges_from_tree(tree, radius):
    df_edges = pd.DataFrame({
        'source': pd.Series(dtype='int32'),
        'target': pd.Series(dtype='int32'),
        'distance': pd.Series(dtype='float')
    })
    for (i, p) in enumerate(pcd.points):
        nn, idx, dists = neighbours(pcd.points[i], tree, radius)
        df_tmp = pd.DataFrame({
            'source': np.repeat(i, nn),
            'target': idx,
            'distance': dists,
        })
        df_edges = pd.concat((df_edges, df_tmp))
    df_edges.reset_index(drop=True, inplace=True)
    df_edges.drop(df_edges[df_edges.source == df_edges.target].index, inplace=True)
    df_edges.reset_index(drop=True, inplace=True)

    return df_edges

def find_points_with_enough_neighbours(
    tree,
    radius,
    min_neighbors,
    max_neighbors = 1e6,
    select_closest = True,
    num_selected = -1,
):
    if select_closest and num_selected <= 0:
        raise Exception("if select_closes, then num_selected must be > 0")

    output = {}
    for (i, p) in enumerate(pcd.points):
        nn, idx, dists = neighbours(p, tree, radius)
        if nn >= min_neighbors and nn <= max_neighbors:
            if select_closest:
                output[i] = {
                    'indexes': idx[:num_selected],
                    'distances': dists[:num_selected]
                }
    return output

def define_label_from_neighbourhood(df, core_point, target, weight_decay = 1.0, cutoff = 0.5):
    idx = core_point['indexes']
    target_values = df.loc[idx,target].values
    weights = np.exp(-weight_decay * np.array(core_point['distances']))
    core_point['prob_' + target] = np.sum(weights * target_values) / np.sum(weights)
    core_point[target] = (core_point['prob_' + target] > cutoff).astype('int32')

    return

def define_label_from_neighbourhood_for_all(df, core_points, target, **kwargs):
    for c in core_points:
        define_label_from_neighbourhood(df, core_points[c], target, **kwargs)

# %%
df = pd.read_csv('example_data1.csv')

#%%
pcd = create_pcd_from_attributes(df, 'class')
visualize_pcd(pcd)
#%%
tree = create_tree_from_point_cloud(pcd)

#%%
df_edges = create_edges_from_tree(tree, 0.3)
df_edges.to_csv('example_edges1.csv')

#%%

# %%
neighbours(pcd.points[0], tree, 0.5)
# %%
core_points = find_points_with_enough_neighbours(tree, 0.3, 10, num_selected=10)

# %%
define_label_from_neighbourhood_for_all(df, core_points, 'class', weight_decay = 0.5)

# %%
for i in core_points:
    if core_points[i]['class'] == 1:
        np.asarray(pcd.colors)[core_points[i]['indexes'], 1:] = [0.5, 0.8]
    else:
        np.asarray(pcd.colors)[core_points[i]['indexes'], 1:] = [0.8, 0.2]
# The true 1 are red
# The chosen as 1 have added green 0.5 and blue 0.8
# The chosen as 0 have added green 0.6 and blue 0.2
# - true 1 and chosen 1 are pink
# - true 0 and chosen 1 are blue
# - true 1 and chosen 0 are yellow/orange
# - true 0 and chosen 0 are green
# - true 1 and not chosen are red
# - true 0 and not chosen are black


# %%
visualize_pcd(pcd)

# %%
