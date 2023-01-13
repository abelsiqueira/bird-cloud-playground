#%%
import pandas as pd
from myaux import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def fill_na_using_knn(df, features):
    # Each feature has to be treated separately
    for feature in features:
        index_na = df[df[feature].isna()].index
        if len(index_na) == 0:
            continue
        print(f'Filling NAN in feature {feature}')
        df_train = df[['x', 'y', feature]].drop(index_na)
        X_pred  = df.loc[index_na, ['x', 'y']].values

        knn = GridSearchCV(
            KNeighborsRegressor(),
            {
                'n_neighbors': [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
                'weights': ['uniform', 'distance'],
            },
            verbose=5,
        )
        knn.fit(df_train[['x', 'y']].values, df_train.loc[:,feature].values)
        print("Best (CV score=%0.3f):" % knn.best_score_)
        print(knn.best_params_)
        y_pred = knn.predict(X_pred)
        df.loc[index_na,feature] = y_pred

    return df

# %%
df = pd.read_csv('data/manual_annotations/NLDHL_pvol_20190416T1945_6234.h5.csv.gz')
feature = 'DBZH'
df = df[['x', 'y', 'z', feature]]
index_na = df[df[feature].isna()].index
df['ISNA'] = df[feature].isna().astype(int)
fill_na_using_knn(df, [feature])
# %%

#%%
# pcd = create_pcd_from_attributes(df.loc[index_na,:], feature, scale_z=10)
pcd = create_pcd_from_attributes(df.drop(index_na), feature, scale_z=10)
visualize_pcd(pcd)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
