import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    ### CONVERT ODDS TO PROBABILITY ###
    def odds_to_prob(x):
        return 1/x
    X['pred_isp'] = X['pred_isp'].apply(odds_to_prob)
    X['f_pm_01m'] = X['f_pm_01m'].apply(odds_to_prob)
    print("✅ ODDS CONVERTED TO PROBABILITY (1/ODDS)")

    ### CODE WINNERS AS '1', REST AS '0' ###
    def winner(x):
        if x == 1:
            return 1
        else:
            return 0
    X['f_place'] = X['f_place'].apply(winner)
    print("✅ WINNERS CODED AS '1'")

    ### 'f_ko' CONVERTED TO DATETIME ###
    X['f_ko'] = X['f_ko'].astype('datetime64[ns]')
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

    X['f_going'] = X['f_going'].apply(f_going_coder)
    print("✅ TRACK CONDITIONS ORDINALLY ENCODED")

    ### DROP ROWS WITH NULL VALUES IN THESE COLUMNS ###
    X.dropna(axis = 0,
            inplace = True,
            subset = ['f_racetype', 'f_jockey',
                    'f_trainer', 'f_pace',
                    'f_rating_or'])
    print("✅ DROPPED ROWS WITH NULL VALUES")

    set_config(transform_output="pandas")
    ### MINMAX SCALE NUMERIC FEATURES ###
    numeric_features = ["f_distance", "f_class",
                        "f_age", "f_pace", "f_weight",
                        "f_runners", "f_rating_rbd",
                        "f_rating_or"]
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
    X_processed = preprocessor.fit_transform(X)
    print("✅ X_PROCESSED WITH SHAPE:", X_processed.shape)

    return X_processed
