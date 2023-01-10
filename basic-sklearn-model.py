#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import open3d as o3d
import os
from myaux import *

# TODO: Move to myaux.py
def create_df_from_input_files(input_files):
    df = pd.read_csv(input_files[0])
    for (i, file) in enumerate(input_files):
        if i == 0:
            continue
        df = pd.concat([df, pd.read_csv(file)])
    return df

input_files = [
    'data/manual_annotations/NLDHL_pvol_20190416T1945_6234.h5.csv.gz',
    'data/manual_annotations/NLDHL_pvol_20190417T2100_6234.h5.csv.gz',
    'data/manual_annotations/NLDHL_pvol_20191005T2100_6234.h5.csv.gz',
    'data/manual_annotations/NLDHL_pvol_20191023T2100_6234.h5.csv.gz',
    'data/manual_annotations/NLHRW_pvol_20191005T2100_6356.h5.csv.gz'
]
df = create_df_from_input_files(input_files)

df['ISBIRD'] = ((1 <= df.CLASS) & (df.CLASS <= 32)).astype(int)
feature_names = [
    #'elevation', 'azimuth', 'range', 'x', 'y', 'z', # position-related
    'DBZH', 'DBZV', 'TH', 'TV',
    'VRADH', 'VRADV',
]
target_name = 'ISBIRD'
df = df[feature_names + [target_name]]
# df = df[0::10].reset_index()
df = df.dropna()
X = df.loc[:,feature_names].values
y = df['ISBIRD'].values

#%%
pcd = create_pcd_from_attributes(df, 'ISBIRD', scale_z=10)
visualize_pcd(pcd)

# %%
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

info_models = {
    'logistic': {
        'sklearn_function': LogisticRegression,
        'params': { 'max_iter': 10000, 'tol': 0.0001 },
        'grid': {
            'clf__C': np.logspace(-4, 4, 4),
        },
    },
    'svc': {
        'sklearn_function': SVC,
        'params': { 'max_iter': 100 },
        'grid': {
            'clf__C': np.logspace(-4, 4, 4),
        },
    },
}

def classification_model(X, y, model):
    info_model = info_models[model]
    SKLearnFunction = info_model['sklearn_function']
    params = info_model['params']
    grid = info_model['grid']

    scaler = StandardScaler()
    clf = SKLearnFunction(**params)
    pipe = Pipeline(steps=[("scaler", scaler), ("clf", clf)])

    search = GridSearchCV(pipe, grid, n_jobs=2)
    search.fit(X, y)
    return search

def predict_on_file(test_file, clf):
    df_vis = pd.read_csv(test_file)

    df_vis['ISBIRD'] = ((1 <= df_vis.CLASS) & (df_vis.CLASS <= 32)).astype(int)
    drop_index = df_vis[feature_names + [target_name]].dropna().index
    df_vis = df_vis.loc[drop_index,:]
    X_test = df_vis.loc[:,feature_names].values
    df_vis['PRED'] = clf.predict(X_test)

    return df_vis

test_file = 'data/manual_annotations/NLDHL_pvol_20191024T2100_6234.h5.csv.gz'
for model in info_models: # keys
    print(f'model {model}')
    search = classification_model(X, y, model)
    print("Best (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    df_vis = predict_on_file(test_file, search)
    y_test = df_vis.loc[:,target_name].values
    y_pred = df_vis['PRED']
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))

    # Red is PRED
    # Green is ISBIRD
    # Yellow = Red + Green = Correct
    # Black = Neither = Correct
    pcd = create_pcd_from_attributes(df_vis, 'PRED', 'ISBIRD', scale_z=10)
    outdir = 'vis/sklearn'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outname = f'{model}.png'
    save_pcd_visualization(pcd, outdir, outname)
