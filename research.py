import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

# THIS CODE MAY NOT WORK FROM SCRATCH

train_df = pd.read_csv(os.path.join(os.getcwd(), "input/train_data.csv"))
test_df = pd.read_csv(os.path.join(os.getcwd(), "input/test_data.csv"))
sns.set(style='whitegrid', context='notebook')
cols = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15'
]

# cols = corr_cols = [
#     'price', 'bathrooms', 'sqft_living',
#     'grade', 'sqft_above','sqft_living15'
# ]
print(train_df[cols].values.T)
np.corrcoef(train_df[cols].values.T)
sns.pairplot(train_df[cols], size=2.5)
plt.show()
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                 yticklabels=cols, xticklabels=cols)
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize=8)
hm.set_xticklabels(hm.get_xticklabels(), rotation=0, fontsize=6)
plt.show()


    #prepare dataset
    #....
    #spilt dataset
Xtrn, Xtest, Ytrn, Ytest = train_test_split(train_df[cols[1:]], train_df[['price']],
                                            test_size=0.2)
models = [
    LinearRegression(),
          RandomForestRegressor(n_estimators=100, max_features='sqrt'),
          KNeighborsRegressor(n_neighbors=6),
    #       SVR(kernel='linear'),
    #       LogisticRegression()
          ]

TestModels = pd.DataFrame()
tmp = {}

for model in models:
    # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    # fit model on training dataset
    model.fit(Xtrn, Ytrn['price'])
    tmp['rmse'] = np.sqrt(mean_squared_error(Ytest['price'], model.predict(Xtest)))
    tmp['R2_Price'] = r2_score(Ytest['price'], model.predict(Xtest))
    # write obtained data
    TestModels = TestModels.append([tmp])

TestModels.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.rmse.plot(ax=axes, kind='bar', title='rmse')
plt.show()