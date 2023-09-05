import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

def preprocess_features_v2(data: pd.DataFrame) -> np.ndarray:
    #--------------------------------------------------------------------------#
    print('number of columns: ', len(data.columns))
    ### DROP IRRELEVANT COLUMNS ###
    redundant_columns = [#"Unnamed: 0",
                         "id",
                         # "f_id",
                         "f_racetype",
                         "f_horse",
                         "f_jockey",
                         "f_trainer",
                         "f_rating_hc",
                         "f_lto_pos",
                         "f_bsp",
                         # "f_pm_15m",
                         "f_pm_10m",
                         # "f_pm_05m",
                         "f_pm_03m",
                         "f_pm_02m",
                         # "f_pm_01m",
                         "f_bsp_p_back",
                         "f_bsp_p_lay",
                         "f_pm_01m_p_back",
                         "f_pm_01m_p_lay",
                         # "f_pm_15m_p_back",
                         "f_pm_15m_p_lay",
                         "general_runs_win_at",
                         "general_runs_win_l200r",
                         "general_runs_win_l50r",
                         "general_runs_win_l16r",
                         "general_runs_at",
                         "general_runs_l200r",
                         "general_runs_l50r",
                         "general_runs_l16r",
                         "sum_bsp_trainer_at",
                         "sum_bsp_jockey_at",
                         "sum_bsp_horse_at",
                         "sum_bsp_trainer_l16r",
                         "sum_bsp_jockey_l16r",
                         "sum_bsp_trainer_l50r",
                         "sum_bsp_jockey_l50r",
                         "sum_bsp_trainer_l200r",
                         "sum_bsp_jockey_l200r",
                         "sum_bsp_horse_l10r",
                         "sum_bsp_horse_l5r",
                         "sum_bsp_horse_l2r",
                         "15to5m_odds_move_perc",
                         "15to5m_odds_move_raw",
                         "15m_odds_prob",
                         "5m_odds_prob"]
    data.drop(columns = redundant_columns, inplace = True)
    print("✅ DROPPED IRRELEVANT COLUMNS")

    #--------------------------------------------------------------------------#

    ### DROP ROWS WITH NULL VALUES IN THESE COLUMNS ###
    data.dropna(axis = 0, inplace = True,
                subset = ["f_pace",
                          "f_stall",
                          "stall_position",
                          "f_going",
                          "f_rating_or",
                          "or_rating_vs_avg_race",
                          "country",
                          "mean_f_rating_or_race",
                          "f_class",
                          "average_or_rating_class",
                          "above_below_official_rating_class",
                          "PreviousPosition",
                          "PredictedRank",
                          ])
    print("✅ DROPPED ROWS WITH NULL VALUES")

    #--------------------------------------------------------------------------#

    ### STRIP SURROUNDING WHITESPACE ###
    data['f_track'] = data['f_track'].str.strip()
    print("✅ WHITESPACE STRIPPED FROM 'f_track'")

    #--------------------------------------------------------------------------#

    ### CONVERT ODDS TO PROBABILITY ###
    def odds_to_prob(x):
        return 1/x
    data['pred_isp'] = data['pred_isp'].apply(odds_to_prob)
    print("✅ ODDS CONVERTED TO PROBABILITY (1/ODDS)")

    #--------------------------------------------------------------------------#

    ### CODE WINNERS AS '1', REST AS '0' ###
    def winner(x):
        if x == 1:
            return 1
        else:
            return 0
    data['f_place'] = data['f_place'].apply(winner)
    print("✅ WINNERS CODED AS '1', REST '0'")

    #--------------------------------------------------------------------------#

    ### 'f_ko' CONVERTED TO DATETIME ###
    data['f_ko'] = data['f_ko'].astype('datetime64[ns]')
    print("✅ 'f_ko' CONVERTED TO DATETIME")

    #--------------------------------------------------------------------------#

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

    #--------------------------------------------------------------------------#

    ### MINMAX SCALE NUMERIC FEATURES ###
    set_config(transform_output = "pandas")
    numeric_features = ["f_going",
                        "average_or_rating_class",
                        "above_below_official_rating_class",
                        "PreviousPosition",
                        "PredictedRank",
                        "f_distance",
                        "f_class",
                        "f_age",
                        "f_pace",
                        "f_weight",
                        "f_runners",
                        "f_rating_or",
                        "mean_f_rating_or_race",
                        "or_rating_vs_avg_race",
                        "f_rating_rbd",
                        "f_stall",
                        "stall_position",
                        "trainer_runs_win_at",
                        "trainer_runs_win_l200r",
                        "trainer_runs_win_l50r",
                        "trainer_runs_win_l16r",
                        "trainer_runs_at",
                        "trainer_runs_l200r",
                        "trainer_runs_l50r",
                        "trainer_runs_l16r",
                        "jockey_runs_win_at",
                        "jockey_runs_win_l200r",
                        "jockey_runs_win_l50r",
                        "jockey_runs_win_l16r",
                        "jockey_runs_at",
                        "jockey_runs_l200r",
                        "jockey_runs_l50r",
                        "jockey_runs_l16r",
                        "horse_runs_win_at",
                        "horse_runs_win_l10r",
                        "horse_runs_win_l5r",
                        "horse_runs_win_l2r",
                        "horse_runs_at",
                        "horse_runs_l10r",
                        "horse_runs_l5r",
                        "horse_runs_l2r",
                        "iv_horse_at",
                        "iv_trainer_l200r",
                        "iv_trainer_l50r",
                        "iv_trainer_l16r",
                        "iv_trainer_at",
                        "iv_jockey_l200r",
                        "iv_jockey_l50r",
                        "iv_jockey_l16r",
                        "iv_jockey_at",
                        "ae_horse_l10r",
                        "ae_horse_l5r",
                        "ae_horse_l2r",
                        "ae_horse_at",
                        "ae_trainer_l200r",
                        "ae_trainer_l50r",
                        "ae_trainer_l16r",
                        "ae_trainer_at",
                        "ae_jockey_l200r",
                        "ae_jockey_l50r",
                        "ae_jockey_l16r",
                        "ae_jockey_at",
                        "rolling_avg_trainer_finish_at",
                        "rolling_avg_trainer_finish_l200r",
                        "rolling_avg_trainer_finish_l50r",
                        "rolling_avg_trainer_finish_l16r",
                        "rolling_avg_horse_finish_at",
                        "rolling_avg_horse_finish_l10r",
                        "rolling_avg_horse_finish_l5r",
                        "rolling_avg_horse_finish_l2r",
                        "rolling_avg_jockey_finish_at",
                        "rolling_avg_jockey_finish_l200r",
                        "rolling_avg_jockey_finish_l50r",
                        "rolling_avg_jockey_finish_l16r",
                        ]
    numeric_transformer = Pipeline(
        steps=[("scaler", MinMaxScaler())])
    print("✅ NUMERIC FEATURES MINMAX-SCALED")

    #--------------------------------------------------------------------------#

    ### IMPUTE HEADGEAR NULLS WITH 'no_headgear' ###
    headgear_feature = ["f_headgear"]
    headgear_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy = 'constant',
                                         fill_value = 'no_headgear'))])
    print("✅ IMPUTED 'no_headgear' for NULLS IN 'f_headgear'")

    #--------------------------------------------------------------------------#

    ### MEAN IMPUTE CERTAIN FEATURES ###
    mean_impute_features = ["f_dob", "f_prb_avg"]
    mean_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy = 'mean'))])
    print("✅ IMPUTED MEAN FOR NULLS IN 'f_dob' & 'f_prb_avg'")

    #--------------------------------------------------------------------------#

    ### IMPUTE NULLS WITH 0's ###
    zero_impute_features = ["f_rating_rbd",
                            "trainer_runs_win_at",
                            "trainer_runs_win_l200r",
                            "trainer_runs_win_l50r",
                            "trainer_runs_win_l16r",
                            "trainer_runs_at",
                            "trainer_runs_l200r",
                            "trainer_runs_l50r",
                            "trainer_runs_l16r",
                            "jockey_runs_win_at",
                            "jockey_runs_win_l200r",
                            "jockey_runs_win_l50r",
                            "jockey_runs_win_l16r",
                            "jockey_runs_at",
                            "jockey_runs_l200r",
                            "jockey_runs_l50r",
                            "jockey_runs_l16r",
                            "horse_runs_win_at",
                            "horse_runs_win_l10r",
                            "horse_runs_win_l5r",
                            "horse_runs_win_l2r",
                            "horse_runs_at",
                            "horse_runs_l10r",
                            "horse_runs_l5r",
                            "horse_runs_l2r",
                            "iv_horse_at",
                            "iv_trainer_l200r",
                            "iv_trainer_l50r",
                            "iv_trainer_l16r",
                            "iv_trainer_at",
                            "iv_jockey_l200r",
                            "iv_jockey_l50r",
                            "iv_jockey_l16r",
                            "iv_jockey_at",
                            "ae_horse_l10r",
                            "ae_horse_l5r",
                            "ae_horse_l2r",
                            "ae_horse_at",
                            "ae_trainer_l200r",
                            "ae_trainer_l50r",
                            "ae_trainer_l16r",
                            "ae_trainer_at",
                            "ae_jockey_l200r",
                            "ae_jockey_l50r",
                            "ae_jockey_l16r",
                            "ae_jockey_at",
                            "rolling_avg_trainer_finish_at",
                            "rolling_avg_trainer_finish_l200r",
                            "rolling_avg_trainer_finish_l50r",
                            "rolling_avg_trainer_finish_l16r",
                            "rolling_avg_horse_finish_at",
                            "rolling_avg_horse_finish_l10r",
                            "rolling_avg_horse_finish_l5r",
                            "rolling_avg_horse_finish_l2r",
                            "rolling_avg_jockey_finish_at",
                            "rolling_avg_jockey_finish_l200r",
                            "rolling_avg_jockey_finish_l50r",
                            "rolling_avg_jockey_finish_l16r"]
    zero_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy = 'constant',
                                         fill_value = 0))])
    print("✅ IMPUTED '0' FOR NULLS IN 68 x FEATURES")

    #--------------------------------------------------------------------------#

    ### O.H.E. CATEGORICAL FEATURES ###
    categorical_features = ["f_track", "f_headgear", "country"]
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False))])
    print("✅ CAT. FEATURES OH-ENCODED (Track, Headgear, Country)")

    #--------------------------------------------------------------------------#

    ### COLUMN TRANSFORMER ###
    ct1 = ColumnTransformer(
        transformers=[("zero_imputer", zero_imputer, zero_impute_features),
                      ("headgear_imputer", headgear_imputer, headgear_feature),
                      ("mean_imputer", mean_imputer, mean_impute_features),
                      ],
        verbose_feature_names_out = False,
        remainder = 'passthrough')

    ct1_processed = ct1.fit_transform(data)

    print('number of columns: ', len(ct1_processed.columns))

    ct2 = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        verbose_feature_names_out = False,
        remainder = 'passthrough')

    ct2_processed = ct2.fit_transform(ct1_processed)

    ct3 = ColumnTransformer(
        transformers=[("scale", numeric_transformer, numeric_features),],
        verbose_feature_names_out = False,
        remainder = 'passthrough')

    print("✅ COLUMN TRANSFORMER ASSEMBLED")

    #--------------------------------------------------------------------------#

    ### FIT_TRANSFORM FEATURES ###
    print("⏳ FIT_TRANSFORMING THE PREPROCESSING PIPE...")
    data_processed = ct3.fit_transform(ct2_processed)
    print('number of columns: ', len(data_processed.columns))
    print("✅ DATA PROCESSED WITH SHAPE:", data_processed.shape)

    return data_processed
