import numpy as np
import open3d as o3d
import os

def normalize_to_01(v):
    minimum_value = v.min()
    maximum_value = v.max()
    return (v - minimum_value) / (maximum_value - minimum_value)

def create_pcd_from_attributes(df, attribute, attribute2=None, scale_z=1.0):
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

def save_pcd_visualization(pcd, outdir, outname):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(outdir, outname))
    vis.destroy_window()

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
