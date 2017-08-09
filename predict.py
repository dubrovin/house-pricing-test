import pandas as pd
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv(os.path.join(os.getcwd(), "input/train_data.csv"))
test_df = pd.read_csv(os.path.join(os.getcwd(), "input/test_data.csv"))
sns.set(style='whitegrid', context='notebook')
cols = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15'
]

Xtrn = train_df[cols[1:]]
Ytrn = train_df[['price']]
Xtest = test_df[cols[1:]]
models = [RandomForestRegressor(n_estimators=100, max_features='sqrt')]

TestModels = pd.DataFrame()
tmp = {}

for model in models:
    model.fit(Xtrn, Ytrn['price'])
    test_df['predicted_price'] = model.predict(Xtest).round(2)

test_df.to_csv("output/predicted.csv")
