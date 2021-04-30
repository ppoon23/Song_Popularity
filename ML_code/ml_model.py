import pandas as pd
from sklearn.compose import *
from sklearn.impute import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.ensemble import *
import pickle

# loading data

X = pd.read_csv('../data_2010/X.csv')
y = pd.read_csv('../data_2010/y.csv')

X = X.drop(columns=['Unnamed: 0', 'id', 'release_date', 'artists', 'name'])
y = y.drop(columns=['Unnamed: 0'])
y = y.values.ravel()

# preprocessing

col = X.columns.tolist()
onehot_cat = ['mode', 'explicit', 'key', 'year']  # one hot encode these features
con_cat = [i for i in col if i not in onehot_cat]  # continuous data

## Create onehotencoding pipeline, impute most frequent feature if missing
cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))
])

# Standardize continous data, use nearest neighbors for missing features
con_pipe = Pipeline([
    ('SS', StandardScaler()),
    ('KNN', KNNImputer(n_neighbors=5))
])

# Using column transformer to pack all features together, and assign them as categorial and continous data
preprocessor = ColumnTransformer([
    ('categorical', cat_pipe, onehot_cat),
    ('continuous', con_pipe, con_cat)
])

# Take the final model and fit the model
### NOTE THIS IS PROTOTYPE

model_final = BaggingRegressor(n_estimators=50, max_features=1.0, n_jobs=-1)
model_final_pipe = Pipeline([('preprocessor', preprocessor),
                             ('bag', TransformedTargetRegressor(regressor=model_final,
                                                                transformer=QuantileTransformer(n_quantiles=300,
                                                                                                output_distribution='normal')))])
# Train on the entire dataset
model_final_pipe.fit(X, y)

# Save the model as pkl
pickle.dump(model_final_pipe, open('../model_final_trained.pkl', 'wb'))
