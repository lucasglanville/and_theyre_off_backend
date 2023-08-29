import pandas as pd

def get_data():
    data = pd.read_csv("raw_data/horse_racing_raw.csv")
    return data



def class_or_rating_average(data):
    average_or_rating = data.groupby("f_class")['f_rating_or'].mean().reset_index()
    average_or_rating.rename(columns={'f_rating_or': 'average_or_rating'}, inplace=True)
    data = data.merge(average_or_rating, on='f_class')
    data['above_below_official_rating'] = data['f_rating_or']- data['average_or_rating']
    return data

def oli_features():
    data['RollingAvgTrainerFinish'] = data.groupby('f_trainer')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

    data['RollingAvgJockeyFinish'] = data.groupby('f_jockey')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

    data['RollingAvgHorseFinish'] = data.groupby('f_horse')['f_place'].apply(
    lambda x: x.expanding().mean().shift(fill_value=0)).reset_index(level=0, drop=True)

    data['PreviousPosition'] = data.groupby('f_horse')['f_place'].shift(fill_value=0)

    data = data.sort_values(by=['f_id', 'f_pm_05m'])
    data['PredictedRank'] = data.groupby('f_id').cumcount() + 1

    return data

def josh_features():

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
    data.rename(columns={'f_rating_or_y': 'mean_f_rating_or', 'f_rating_or_x' : 'f_rating_or' }, inplace=True)

    # Create official rating vs average rating in the race feature
    data['or_rating_vs_avg'] = data['f_rating_or'] - data['mean_f_rating_or']

    # Create odds percentage and movement features
    data['15m_odds_prob'] = 1 / data['f_pm_15m']
    data['5m_odds_prob'] = 1 / data['f_pm_05m']
    data['15to5m_odds_move_perc'] = (data['5m_odds_prob'] / data['15m_odds_prob'] - 1)
    data['15to5m_odds_move_raw'] = (data['5m_odds_prob'] - data['15m_odds_prob'])
