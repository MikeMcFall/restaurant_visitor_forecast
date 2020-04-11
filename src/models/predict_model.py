import pandas as pd
from joblib import load


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
gbr = load('models/gbr_1000est_sqrt_depth5.pkl')
y_pred = gbr.predict(X_test)
X_test = X_test.reset_index()
submission = pd.DataFrame()
submission['id'] = X_test['air_store_id'] + '_' + X_test['calendar_date'].astype('str')
submission['visitors'] = y_pred
submission.to_csv('models/predictions_gbr_1000est_sqrt_depth5.csv', index=False)