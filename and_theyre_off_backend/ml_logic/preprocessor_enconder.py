import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_features(X: pd.DataFrame) -> np.ndarray:

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


    ### MINMAX SCALE NUMERIC FEATURES ###
    numeric_features = ["f_distance", "f_class",
                        "f_age", "f_pace", "f_weight",
                        "f_runners", "f_rating_rbd",
                        "f_rating_or"]
    numeric_transformer = Pipeline(
        steps=[("scaler", MinMaxScaler())])
    print("✅ NUMERIC FEATURES MINMAX-SCALED")

    ### O.H.E. CATEGORICAL FEATURES ###
    categorical_features = ['f_track', 'f_jockey', 'f_trainer']
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False,
                                     categories='auto')),
        ])
    print("✅ CAT. FEATURES OH-ENCODED (Track, Jockey, Trainer)")

    ### COLUMN TRANSFORMER ###
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])
    print("✅ COLUMN TRANSFORMER ASSEMBLED")

    ### FIT_TRANSFORM FEATURES ###
    X_processed = preprocessor.fit_transform(X)
    print("✅ X_PROCESSED WITH SHAPE:", X_processed.shape)

    return X_processed
