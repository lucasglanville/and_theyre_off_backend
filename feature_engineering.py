import pandas as pd

data = pd.read_csv("raw_data/horse_racing_raw.csv")



def class_or_rating_average(data):
    average_or_rating = data.groupby("f_class")['f_rating_or'].mean().reset_index()
    average_or_rating.rename(columns={'f_rating_or': 'average_or_rating'}, inplace=True)
    data = data.merge(average_or_rating, on='f_class')
    data['above_below_official_rating'] = data['f_rating_or']- data['average_or_rating']
    return data


data['RollingAvgTrainerFinish'] = data.groupby('f_trainer')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

data['RollingAvgJockeyFinish'] = data.groupby('f_jockey')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

data['RollingAvgHorseFinish'] = data.groupby('f_horse')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

data['PreviousPosition'] = data.groupby('f_horse')['f_place'].shift(fill_value=0)

data = data.sort_values(by=['f_id', 'f_pm_05m'])
data['PredictedRank'] = data.groupby('f_id').cumcount() + 1
