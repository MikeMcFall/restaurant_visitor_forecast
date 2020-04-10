import pandas as pd


# This is largely a function used by Max Halford in his solution to this contest
# https://github.com/MaxHalford/kaggle-recruit-restaurant/blob/master/Solution.ipynb

def extract_statistics(df, on, group_by):

    df.sort_values(group_by + ['calendar_date'], inplace=True)

    groups = df.groupby(group_by, sort=False)

    stats = {
        'mean': [],
        'median': [],
        'std': [],
        'max': [],
        'min': []
    }

    exp_alphas = [0.1, 0.25, 0.3, 0.5, 0.75]
    stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})

    for _, group in groups:
        shift = group[on].shift()
        roll = shift.rolling(window=len(group), min_periods=1)

        stats['mean'].extend(roll.mean())
        stats['median'].extend(roll.median())
        stats['std'].extend(roll.std())
        stats['max'].extend(roll.max())
        stats['min'].extend(roll.min())

        for alpha in exp_alphas:
            exp = shift.ewm(alpha=alpha, adjust=False)
            stats['exp_{}_mean'.format(alpha)].extend(exp.mean())

    suffix = '_&_'.join(group_by)

    for stat_name, values in stats.items():
        df['{}_{}_by_{}'.format(on, stat_name, suffix)] = values


df = pd.read_pickle('data/interim/air_weather.pkl')
extract_statistics(df=df, on='visitors', group_by=['air_store_id'])
extract_statistics(df=df, on='visitors', group_by=['air_store_id', 'day_of_week'])

df.to_csv('data/processed/air_weather_features.csv')
df.to_pickle('data/processed/air_weather_features.pkl')
