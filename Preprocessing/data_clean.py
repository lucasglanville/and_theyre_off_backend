import pandas as pd
import os
#from params import *


def get_data(LOCAL_FILEPATH):
    data = pd.read_csv(LOCAL_FILEPATH)
    print(f"data acquired with shape {data.shape}")
    return data



#This is the first 2 days worth of data where general_runs stats are all skewed
def remove_221_rows(data):
    data = data[221:]
    print(f"first 221 rows removed. New shape = {data.shape}")
    return data

## Dropping rows if we have no betting data, and imputing if we have some betting data
def dropping_no_betting_data(data):
    data = data.dropna(subset=['f_bsp', 'f_pm_15m', 'f_pm_10m', 'f_pm_05m' , 'f_pm_03m', 'f_pm_02m', 'f_pm_01m'], how='all')
    def impute_row(row):
        row = row.fillna(method='ffill').fillna(method='bfill')
        return row
    columns_to_impute = ['f_bsp', 'f_pm_15m', 'f_pm_10m', 'f_pm_05m', 'f_pm_03m', 'f_pm_02m', 'f_pm_01m']
    data[columns_to_impute] = data[columns_to_impute].apply(impute_row, axis=1)

    #deleting row that has 0s for all odds for some reason


    print(f"Cleaned up missing odds. New shape = {data.shape}")
    return data

def josh_features(data):

    #Creating country feature
    irish_tracks = [
    "SLIGO", "LIMERICK", "NAVAN", "WEXFORD", "CURRAGH",
    "GALWAY", "KILBEGGAN", "GOWRAN PARK", "BELLEWSTOWN",
    "LISTOWEL", "THURLES", "BALLINROBE", "TRAMORE",
    "LEOPARDSTOWN", "DOWN ROYAL", "ROSCOMMON", "CORK",
    "DUNDALK", "KILLARNEY", "LAYTOWN", "TIPPERARY",
    "FAIRYHOUSE", "NAAS", "DOWNPATRICK", "CLONMEL",
    "PUNCHESTOWN"
]

    data['country'] = data['f_track'].apply(lambda x: 'IRE' if x in irish_tracks else 'GB')

    #Completing f_class for Irish races

    # Calculate mean ratings for each 'f_id' group
    mean_ratings_by_id = data.groupby('f_id')['f_rating_or'].mean()

    # Define the mapping of mean ratings to f_class values
    rating_to_f_class_mapping = {
        (96, float('inf')): 1,
        (86, 96): 2,
        (76, 86): 3,
        (66, 76): 4,
        (56, 66): 5,
        (46, 56): 6,
        (-float('inf'), 46): 7
    }

    # Function to map mean ratings to f_class values
    def map_rating_to_f_class(mean_rating):
        for rating_range, f_class_value in rating_to_f_class_mapping.items():
            if rating_range[0] <= mean_rating <= rating_range[1]:
                return f_class_value

    # Apply the mapping to fill NULL values in 'f_class' column based on mean ratings
    data['f_class'] = data.apply(lambda row: map_rating_to_f_class(mean_ratings_by_id.get(row['f_id'])), axis=1)

    # Now the 'f_class' column should be filled based on the specified mapping using mean ratings

    # Merge the mean ratings back into the original DataFrame based on 'f_id'
    data = data.merge(mean_ratings_by_id, how='left', left_on='f_id', right_index=True)

    # Rename the merged mean rating column for clarity
    data.rename(columns={'f_rating_or_y': 'mean_f_rating_or_race', 'f_rating_or_x' : 'f_rating_or' }, inplace=True)

    # Create official rating vs average rating in the race feature
    data['or_rating_vs_avg_race'] = data['f_rating_or'] - data['mean_f_rating_or_race']

    # Create odds percentage and movement features
    data['15m_odds_prob'] = 1 / data['f_pm_15m']
    data['5m_odds_prob'] = 1 / data['f_pm_05m']
    data['15to5m_odds_move_perc'] = (data['5m_odds_prob'] / data['15m_odds_prob'] - 1)
    data['15to5m_odds_move_raw'] = (data['5m_odds_prob'] - data['15m_odds_prob'])
    print(f"Added Josh features. New shape = {data.shape}")
    return data

def class_or_rating_average(data):
    average_or_rating_class = data.groupby("f_class")['f_rating_or'].mean().reset_index()
    average_or_rating_class.rename(columns={'f_rating_or': 'average_or_rating_class'}, inplace=True)
    data = data.merge(average_or_rating_class, on='f_class')
    data['above_below_official_rating_class'] = data['f_rating_or']- data['average_or_rating_class']
    print(f"Added Oli features 2/4. New shape = {data.shape}")
    return data


def oli_features(data):
    data = data.sort_values(by='f_ko')
    data['PreviousPosition'] = data.groupby('f_horse')['f_place'].shift(fill_value=0)

    data = data.sort_values(by=['f_id', 'pred_isp'])
    data['PredictedRank'] = data.groupby('f_id').cumcount() + 1
    print(f"Added Oli features 4/4. New shape = {data.shape}")
    return data

# Define a lambda function to apply the conditions and fill NaN values
def fill_f_pm_01m(data):
    def fill_nan(row):
        if row['f_place'] == 0:
            return -1
        elif row['f_place'] == 1:
            return (row['f_pm_01m'] - 1) * 0.95
        else:
            return row['f_pm_01m_p_back']

# Apply the lambda function to fill NaN values in f_pm_01m_p_back
    data['linear_target'] = data.apply(fill_nan, axis=1)
    return data


if __name__ == '__main__':

    data = get_data("raw_data/raw_data_v2.1.csv")
    data = remove_221_rows(data)
    data = dropping_no_betting_data(data)
    #Dropping line that has 0 for all odds
    data = data[data.id != 16157910000382]
    data = josh_features(data)
    data = class_or_rating_average(data)
    data = oli_features(data)
    data.to_csv("raw_data/data_clean.csv", index=False)
