# If we take the radar and select nodes at least N neighbours, how many do we have?
# This is relevant if we create a graph for each neighbour.
#%%
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import os

#%%
# Prefix needs to be set according to there you started an interactive session :/
data_folder = 'data/manual_annotations'
data_file = os.listdir(data_folder)[0]
df = pd.read_csv(os.path.join(data_folder, data_file))
# df = df[::100]

#%%
print(df.shape)
df = df.drop(df[np.logical_or(df.range > 100000, np.logical_or(df.z > 10000, df.range < 5000))].index).reset_index(drop=True)
print(df.shape)
#%%
def value_counts(v):
    unique_v = np.unique(v)
    counts = {vi: sum(v == vi) for vi in unique_v if not np.isnan(vi)}
    if np.any(np.isnan(unique_v)):
        counts[np.nan] = np.sum(np.isnan(v))
    return counts

features = df.columns[0:16]
target = 'BIOLOGY'
value_counts(df[target])

#%%
df_notna = df.drop(df[df[target].isna()].index).reset_index(drop=True)
print(df_notna.shape)
value_counts(df_notna[target])
# %%
tree = KDTree(df.loc[:, ['x', 'y', 'z']])
tree_notna = KDTree(df_notna.loc[:, ['x', 'y', 'z']])
# %%
max_distance = 500

# neighbours_vector = df[::100].apply(lambda row: len(tree.query_ball_point(row.loc[['x', 'y', 'z']], max_distance)), axis=1)
# %%
distance_matrix = tree_notna.sparse_distance_matrix(tree, max_distance)

# %%
number_neighbours = np.array(np.sum(distance_matrix > 0, axis = 1)).reshape(-1)
value_counts(number_neighbours)
# %%
min_neighbours = 100
points_of_interest = np.where(number_neighbours >= min_neighbours)[0]


# %%
graphs = []
for p in points_of_interest:
    distances, indexes = tree.query(df.loc[p, ['x', 'y', 'z']], min_neighbours)
    graphs.append({
        'main_index': p,
        'neighbourhood': indexes,
    })
    # subtree = KDTree(df.loc[indexes, ['x', 'y', 'z']])
    # submatrix = subtree.sparse_distance_matrix(subtree, max_distance)
    # graphs.append(
    #     (
    #         submatrix,
    #         df.loc[indexes, features],
    #         df.loc[indexes, target],
    #     )
    # )
# %%
import dgl
import torch

indexes = graphs[0]['neighbourhood']
local_tree = KDTree(df.loc[indexes, ['x', 'y', 'z']])
D = local_tree.sparse_distance_matrix(local_tree, 1500, output_type='coo_matrix')
g = dgl.graph((D.row, D.col))
g.ndata["x"] = torch.tensor(df.loc[indexes, features].values)
g.edata["a"] = torch.tensor(D.data)
# %%
