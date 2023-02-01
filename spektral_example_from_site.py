# https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py
# Need to install fixed spektral:
# pip install --force-reinstall git+https://github.com/abelsiqueira/spektral.git@414-fix-for-numpy-1.24
#%%
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import LayerPreprocess

# %%
learning_rate = 1e-2
seed = 0
epochs = 200
patience = 10
data = "cora"
# %%
tf.random.set_seed(seed=seed)
# %%
dataset = Citation(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
# %%
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)
# %%
model = GCN(n_labels=dataset.n_labels)
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)
# %%
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)
# %%
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
# %%
