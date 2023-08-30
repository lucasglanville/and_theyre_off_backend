
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_features(data: pd.DataFrame) -> np.ndarray:

    ### dropping columns ###

    data.drop(columns=['id.1', 'f_id.1','id','f_id', 'f_horse',
                       'f_bsp_p_back', 'f_bsp_p_lay',	'f_pm_01m_p_back',
                      'f_pm_01m_p_lay','f_pm_15m_p_back',	'f_pm_15m_p_lay',
                      'f_ip_min', 'f_ip_max'], inplace = True)


    ### CONVERT ODDS TO PROBABILITY ###
    def odds_to_prob(x):
        return 1/x
    data['pred_isp'] = data['pred_isp'].apply(odds_to_prob)
    data['f_pm_01m'] = data['f_pm_01m'].apply(odds_to_prob)
    print("✅ ODDS CONVERTED TO PROBABILITY (1/ODDS)")

    ### CODE WINNERS AS '1', REST AS '0' ###
    def winner(x):
        if x == 1:
            return 1
        else:
            return 0
    data['f_place'] = data['f_place'].apply(winner)
    print("✅ WINNERS CODED AS '1'")

    ### 'f_ko' CONVERTED TO DATETIME ###
    data['f_ko'] = data['f_ko'].astype('datetime64[ns]')
    print("✅ 'f_ko' CONVERTED TO DATETIME")

    ### ENCODE THE GOING (GROUND CONDITION) ###
    def f_going_coder(x):
        if x == 'FRM':
            return 1
        elif x == 'GTF':
            return 2
        elif x == 'GD' or x == 'GTY' or x == 'STD':
            return 3
        elif x == 'YLD' or x == 'YTS' or x == 'STSL' or x == 'GTS':
            return 4
        elif x == 'SFT':
            return 5
        elif x == 'HVY' or x == 'HTS':
            return 6

    data['f_going'] = data['f_going'].apply(f_going_coder)
    print("✅ TRACK CONDITIONS ORDINALLY ENCODED")

    ### DROP ROWS WITH NULL VALUES IN THESE COLUMNS ###
    data.dropna(axis = 0,
            inplace = True,
            subset = ['f_racetype', 'f_jockey', 'f_class',
                    'f_trainer', 'f_pace',
                    'f_rating_or'])
    print("✅ DROPPED ROWS WITH NULL VALUES")

    ### MINMAX SCALE NUMERIC FEATURES ###
    set_config(transform_output="pandas")
    numeric_features = ["f_distance", "f_class",
                        "f_age", "f_pace", "f_weight",
                        "f_runners", "f_rating_rbd",
                        "f_rating_or", "f_going", 'f_bsp',
                        'f_pm_15m',	'f_pm_10m',	'f_pm_05m',
                        'f_pm_03m',	'f_pm_02m', 'average_or_rating_class',
                        'above_below_official_rating_class',	'PreviousPosition',
                        'PredictedRank']
    numeric_transformer = Pipeline(
        steps=[("scaler", MinMaxScaler())])
    print("✅ NUMERIC FEATURES MINMAX-SCALED")



    ### O.H.E. CATEGORICAL FEATURES ###
    categorical_features = ['f_track', 'f_jockey', 'f_trainer', 'f_racetype']
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False,
                                     categories='auto')),
        ])
    print("✅ CAT. FEATURES OH-ENCODED (Track, Jockey, Trainer, Racetype)")

    ### COLUMN TRANSFORMER ###
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features),
                      ("cat", categorical_transformer, categorical_features)],
        verbose_feature_names_out = False,
        remainder = 'passthrough')
    print("✅ COLUMN TRANSFORMER ASSEMBLED")

    ### FIT_TRANSFORM FEATURES ###
    print("⏳ FIT_TRANSFORMING FEATURES...")
    data_processed = preprocessor.fit_transform(data)
    print("✅ X_PROCESSED WITH SHAPE:", data_processed.shape)

    return data_processed
