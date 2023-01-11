#%%
import open3d as o3d
import pandas as pd
import os
from myaux import *

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
    'elevation', 'azimuth', 'range', 'x', 'y', 'z', # position-related
    'DBZH', 'DBZV', 'TH', 'TV',
    'VRADH', 'VRADV',
]
target_name = 'ISBIRD'
df = df[feature_names + [target_name]]
# df = df[0::10].reset_index()
df = df.dropna()
# X = df.loc[:,feature_names].values
# y = df['ISBIRD'].values
#%%
df_train = df.sample(frac=0.8, random_state=0)
df_dev = df.drop(df_train.index)
X_train = df_train.loc[:,feature_names].values
Y_train = df_train.loc[:,'ISBIRD'].values
X_dev  = df_dev.loc[:,feature_names].values
Y_dev  = df_dev.loc[:,'ISBIRD'].values

# %%
import tensorflow as tf
from tensorflow.keras import layers as tfl

inputs = tf.keras.Input(shape=(X_train.shape[1], ))
dense1 = tfl.Dense(64, activation='relu')(inputs)
dense2 = tfl.Dense(8, activation='relu')(inputs)
outputs = tfl.Dense(1, activation='sigmoid')(dense1)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
)
model.summary()

# %%
model.fit(X_train, Y_train, epochs=10, batch_size=512)
# %%
model.evaluate(X_dev, Y_dev)
# %%
other_file = 'data/manual_annotations/NLDHL_pvol_20191024T2100_6234.h5.csv.gz'
df_test = pd.read_csv(other_file)
df_test['ISBIRD'] = ((1 <= df_test.CLASS) & (df_test.CLASS <= 32)).astype(int)
df_test = df_test[feature_names + [target_name]]
# df_test = df_test[0::10].reset_index()
df_test = df_test.dropna()
#%%
X_test = df_test.loc[:,feature_names].values
Y_test = df_test.loc[:,'ISBIRD'].values
Y_pred = model.predict(X_test)

#%%
df_test['PRED'] = Y_pred
# Red is PRED
# Green is ISBIRD
# Yellow = Red + Green = Correct
# Black = Neither = Correct
pcd = create_pcd_from_attributes(df_test, 'PRED', 'ISBIRD', scale_z=10)
outdir = 'vis/keras'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
outname = 'basic'
save_pcd_visualization(pcd, outdir, outname)

# %%
