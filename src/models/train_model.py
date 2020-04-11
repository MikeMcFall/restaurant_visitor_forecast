import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump


def train_test_split(df):
    df_dummy = pd.get_dummies(df)
    train = df_dummy.loc[(df_dummy['is_test'] == False) & (df_dummy['was_closed_False'])].copy()
    train.dropna(inplace=True)
    test = df_dummy.loc[df_dummy['is_test']].copy()

    columns_to_drop = ['visitors', 'is_test', 'was_closed_False', 'was_closed_True']
    X_train = train.drop(columns_to_drop, axis=1)
    y_train = train.loc[train['is_test'] == False, 'visitors'].copy()
    X_test = test.drop(columns_to_drop, axis=1)

    return X_train, y_train, X_test


df = pd.read_pickle('data/processed/air_weather_features.pkl')
X_train, y_train, X_test = train_test_split(df)
gbr = GradientBoostingRegressor(n_estimators=1000, max_features='sqrt', max_depth=5)
gbr.fit(X_train, y_train)
dump(gbr, 'models/gbr_1000est_sqrt_depth5.pkl')
