#%%
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import os

data_folder = 'data/manual_annotations'
target = 'BIOLOGY'

def value_counts(v):
    unique_v = np.unique(v)
    counts = {vi: sum(v == vi) for vi in unique_v if not np.isnan(vi)}
    if np.any(np.isnan(unique_v)):
        counts[np.nan] = np.sum(np.isnan(v))
    return counts

def find_num_neigh_within_distance(df, tree, max_distance = 500):
    num_neighbours_per_point = {}
    for p in df[df[target].notna()].index:
        num_neighbours = tree.query_ball_point(df.loc[p, ['x', 'y', 'z']], max_distance, return_length=True)
        num_neighbours_per_point[p] = num_neighbours

    return num_neighbours_per_point

def find_distances_of_closest_neigh(df, tree, min_neighbours = 50):
    distance_list_per_point = {}
    for p in df[df[target].notna()].index:
        distances, indexes = tree.query(df.loc[p, ['x', 'y', 'z']], min_neighbours)
        distance_list_per_point[p] = distances

    return distance_list_per_point

#%%
def compute_file_statistics(max_distance = 1000, min_neighbours = 200):
    stats = []
    for filename in os.listdir(data_folder)[0:2]:
        print(f'Processing file {filename}')
        df = pd.read_csv(os.path.join(data_folder, filename))
        df = df.drop(df[np.logical_or(df.range > 100000, np.logical_or(df.z > 10000, df.range < 5000))].index).reset_index(drop=True)
        # df = df[::50]
        tree = KDTree(df.loc[:, ['x', 'y', 'z']])

        nnwd = find_num_neigh_within_distance(df, tree, max_distance)
        dcn = find_distances_of_closest_neigh(df, tree, min_neighbours)
        notna_target = df[df[target].notna()][target].values

        biology_value_counts = value_counts(df[target])
        for i in [0, 1, np.nan]:
            if not i in biology_value_counts:
                biology_value_counts[i] = 0

        nnwd_pet_target = [
            value_counts([
                nnwd[k]
                for k in nnwd
                if df.loc[k, target] == target_i
            ])
            for target_i in [0, 1]
        ]

        dcn_list = list(dcn.values())
        dcn_per_target = [
            [] if biology_value_counts[target_i] == 0 else np.concatenate([
                D for (i, D) in enumerate(dcn_list)
                if notna_target[i] == target_i
            ])
            for target_i in [0, 1]
        ]
        dcn_list = sorted(np.concatenate(dcn_list))

        stats.append({
            'biology_value_counts': biology_value_counts,
            'filename': filename,
            'num_neigh_value_counts': value_counts(list(nnwd.values())),
            'num_neigh_value_counts_per_target': nnwd_pet_target,
            'distances_list': dcn_list,
            'distances_list_per_target': dcn_per_target,
            # '_target': notna_target,
            # '_num_neigh_within_distance': nnwd,
            # '_distances_of_closest_neigh': dcn,
        })
    return stats

stats = compute_file_statistics(max_distance=500, min_neighbours=100)

# %%
def sum_dict(D_list):
    D_sum = {}
    for D in D_list:
        for key in D.keys():
            if not key in D_sum:
                D_sum[key] = 0
            D_sum[key] += D[key]
    return D_sum

def aggregate_file_statistics(stats):
    single_stats = {
        'biology_value_counts': sum_dict(
            S['biology_value_counts'] for S in stats
        ),
        'num_neigh_value_counts': sum_dict(
            S['num_neigh_value_counts'] for S in stats
        ),
        'num_neigh_value_counts_per_target': [
            sum_dict(
                S['num_neigh_value_counts_per_target'][target_i] for S in stats
            )
            for target_i in [0, 1]
        ],
        'distances_list': sorted(np.concatenate([
            S['distances_list'] for S in stats
        ])),
        'distances_list_per_target': [
            sorted(np.concatenate([
                S['distances_list_per_target'][target_i] for S in stats
            ]))
            for target_i in [0, 1]
        ],
    }
    return single_stats

single_stats = aggregate_file_statistics(stats)
# %%
### Visualizing single stats
import matplotlib.pyplot as plt

def dict_to_x_y(dict):
    x = []
    y = []
    labels = []
    for k in dict.keys():
        v = dict[k]
        if np.isnan(k):
            x.append(-1)
            labels.append('nan')
        else:
            x.append(k)
            labels.append(str(k))
        y.append(v)
    return x, y, labels

x, y, labels = dict_to_x_y(single_stats['biology_value_counts'])
plt.bar(x, y, tick_label=labels, log=True)
# %%
x, y, labels = dict_to_x_y(single_stats['num_neigh_value_counts'])
plt.bar(x, y, tick_label=labels, log=True)
# %%
for target_i in [0, 1]:
    x, y, labels = dict_to_x_y(single_stats['num_neigh_value_counts_per_target'][target_i])
    plt.bar(
        np.array(x) + 0.4 * target_i,
        y,
        tick_label=labels,
        log=True,
        alpha=0.5,
        align='edge',
        width=0.4,
        label=f'biology={target_i}: num points: {np.sum(y)}'
    )
plt.legend(loc='upper right')
plt.show()
# %%
D = single_stats['distances_list']
plt.hist(D, bins=100)
plt.show()
# %%
for target_i in [0, 1]:
    D = single_stats['distances_list_per_target'][target_i]
    plt.hist(
        D,
        bins=100,
        alpha=0.5,
        density=True,
        label=f'biology={target_i}: num points: {len(D)}',
    )
plt.legend(loc='upper right')
plt.show()
# %%
