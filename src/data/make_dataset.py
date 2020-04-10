# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import glob
import pandas as pd
import numpy as np


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # These are the files provided by the Kaggle competition
    visit_data = pd.read_csv('data/raw/air_visit_data.csv')
    date_info = pd.read_csv('data/raw/date_info.csv')
    test = pd.read_csv('data/raw/sample_submission.csv')

    # These files are from weather dataset. The air_store_info_with_nearest_active_station file contains the same data
    # as the air_store_info file provided in the competition, but with extra features that make it easy to merge with
    # the weather data.
    air_store_weather = pd.read_csv('data/raw/Weather/air_store_info_with_nearest_active_station.csv')
    features = pd.read_csv('data/raw/Weather/feature_manifest.csv')

    # Loading weather data. Each station is kept in a separate csv file. This loads all of them into a single DataFrame.
    path = 'data/raw/Weather/1-1-16_5-31-17_Weather/'
    all_files = glob.glob(os.path.join(path, '*.csv'))
    weather = pd.concat((pd.read_csv(f).assign(station_id=f.split('\\')[-1].split('.')[0]) for f in all_files),
                        ignore_index=True)
    weather['calendar_date'] = pd.to_datetime(weather['calendar_date'])
    weather = weather.set_index(['calendar_date', 'station_id']).sort_index()

    # There are 108 unique latitude/longitude pairs and over 1,600 weather stations.
    # This file contains this distances betweeen each pair.
    distances = (pd.read_csv('data/raw/Weather/air_station_distances.csv', index_col=0)
                 .drop(['station_latitude', 'station_longitude'], axis=1))
    retired_stations = distances[distances.index.str.contains('___')].index
    distances.drop(retired_stations, inplace=True)


    def closest_five_stations(srs):
        smallest = srs.sort_values()[0:5].index
        is_lowest = pd.Series(0, index=srs.index)
        is_lowest.loc[smallest] = 1
        return is_lowest

    closest = distances.apply(lambda srs: closest_five_stations(srs))

    assert isinstance(weather.index, object)
    dates = weather.index.get_level_values('calendar_date').unique()
    locations = closest.columns
    measurements = weather.columns
    new_index = pd.MultiIndex.from_product([locations, dates], names=['location', 'calendar_date'])
    average_weather = pd.DataFrame(0, index=new_index, columns=measurements)
    average_weather.sort_index(level=[0, 1], inplace=True)

    for date in weather.index.get_level_values('calendar_date').unique():
        current_dates_weather = weather.loc[date]
        for location in average_weather.index.get_level_values('location').unique():
            locations_to_average = closest[closest.loc[:, location].astype('bool')].index
            average_weather.loc[(location, date), :] = (current_dates_weather
                                                        .loc[locations_to_average, :]
                                                        .mean())

    air_store_weather.set_index('air_store_id', inplace=True)
    air_store_weather['location'] = ('(' + air_store_weather['latitude_str'].str.strip('\"') + ', ' +
                                     air_store_weather['longitude_str'].str.strip('\"') + ')')

    test['air_store_id'] = test['id'].str.slice(0, 20)
    test['calendar_date'] = test['id'].str.slice(21)
    test['calendar_date'] = pd.to_datetime(test['calendar_date'])
    test['visitors'] = np.nan
    test['is_test'] = True

    visit_data.columns = ['air_store_id', 'calendar_date', 'visitors']
    visit_data.index = pd.to_datetime(visit_data['calendar_date'])
    visit_data = visit_data.groupby('air_store_id').apply(lambda g: g['visitors'].resample('1d').sum()).reset_index()
    visit_data['was_closed'] = visit_data['visitors'] == 0
    visit_data_all = pd.concat((visit_data, test.drop('id', axis='columns')), sort=True)
    visit_data_all['is_test'].fillna(False, inplace=True)

    date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])
    average_weather.reset_index(inplace=True)
    average_weather['calendar_date'] = pd.to_datetime(average_weather['calendar_date'])

    df = (date_info
          .merge(visit_data_all, how='left', on='calendar_date')
          .merge(air_store_weather, how='left', on='air_store_id')
          .merge(average_weather, how='left', on=['location', 'calendar_date'])
          )

    df['city'] = [x[0] for x in df['air_area_name'].str.split()]
    df['prefecture'] = [x[1] for x in df['air_area_name'].str.split()]
    df['subprefecture'] = [x[2] for x in df['air_area_name'].str.split()]
    df.drop('air_area_name', inplace=True, axis=1)

    columns_to_drop = ['latitude_str', 'longitude_str', 'station_id', 'station_latitude', 'station_longitude',
                       'station_vincenty', 'station_great_circle', 'location', 'total_snowfall', 'deepest_snowfall']
    df.drop(columns=columns_to_drop, inplace=True)

    def fill_missing_weather(row):
        for col in missing_cols:
            if np.isnan(row[col]):
                row[col] = fill_values_city_date[col][(row['city'], row['calendar_date'])]
                if np.isnan(row[col]):
                    row[col] = fill_values_date[col][row['calendar_date']]
        return row

    missing_cols = ['solar_radiation', 'visitors', 'cloud_cover', 'avg_humidity',
                    'avg_vapor_pressure', 'avg_local_pressure', 'avg_sea_pressure', 'avg_wind_speed',
                    'high_temperature', 'low_temperature']
    fill_values_city_date = df.groupby(by=['city', 'calendar_date'])[missing_cols].mean().to_dict()
    fill_values_date = df.groupby(by=['calendar_date'])[missing_cols].mean().to_dict()
    df.loc[df.isna().any(axis=1)] = df[df.isna().any(axis=1)].apply(fill_missing_weather, axis=1)

    categoricals = ['city', 'prefecture', 'subprefecture', 'air_genre_name', 'day_of_week']
    df[categoricals] = df[categoricals].astype('category')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = df['day_of_week'].cat.reorder_categories(day_order, ordered=True)
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])
    df['year'] = df['calendar_date'].dt.year
    df['month'] = df['calendar_date'].dt.month
    df['day'] = df['calendar_date'].dt.day
    df.set_index(['air_store_id', 'calendar_date'], inplace=True)
    df.sort_index(inplace=True)

    precip95= np.percentile(df['precipitation'], 95)
    df.loc[df['precipitation'] > precip95, 'precipitation'] = precip95

    def remove_outlier(df, col):
        quartiles = np.percentile(df[col].dropna(), [25, 75])
        iqr = quartiles[1] - quartiles[0]
        whisker = quartiles[1] + (iqr * 1.5)
        df.loc[df[col] > whisker, col] = whisker

    remove_outlier(df, 'avg_wind_speed')

    df.to_csv('data/interim/air_weather.csv')
    df.to_pickle('data/interim/air_weather.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
