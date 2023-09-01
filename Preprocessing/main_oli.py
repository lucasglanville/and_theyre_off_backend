from data_clean import remove_221_rows, dropping_no_betting_data, josh_features, class_or_rating_average, oli_features
from v2_preprocessor_for_olis_pipeline import preprocess_features_v2
from Models.latest_model import train_test_val, model, backtesting_func
import pandas as pd

data  = pd.read_csv("raw_data/raw_data_v2.2.csv")

data_cleaned = remove_221_rows(data)
data_cleaned = dropping_no_betting_data(data_cleaned)
data_cleaned = josh_features(data_cleaned)
data_cleaned = class_or_rating_average(data_cleaned)
data_cleaned = oli_features(data_cleaned)



preprocessed_data = preprocess_features_v2(data_cleaned)

preprocessed_data.to_csv("data_cleaned_and_preprocessed_v3.csv", index=False)

X_test_pca, X_train_pca, X_val_pca, y_test, y_val, y_train, backtesting = train_test_val()

model_ = model(X_train_pca, X_val_pca, y_train, y_val)

y_pred = model_.predict(X_test_pca)

print(y_pred)

backtesting_df = backtesting_func(backtesting, y_pred)

print(backtesting_df['cumulative_profit_0.05'])


# data = pd.read_csv("raw_data/data_cleaned_and_preprocessed_v3.csv")
# data.sort_values(by='f_ko')
# data = data.reset_index(drop=True)

# backtest = data[['f_ko','f_id', 'id','f_horse','f_pm_01m', 'linear_target', 'f_place']]

# X = data.drop(columns=['f_pm_01m', 'f_pm_01m_p_back' , 'f_place', 'f_id', 'id', 'f_horse',
#                        'trainer_runs_l200r', 'trainer_runs_l50r', 'trainer_runs_l16r',
#                        'jockey_runs_l200r', 'jockey_runs_l50r', 'jockey_runs_l16r',
#                       'horse_runs_l10r', 'horse_runs_l5r', 'horse_runs_l2r', 'linear_target'])
# y = data["f_place"] #OR 'linear_target'

# print(data.shape)
# print(X.shape)
# print(y.shape)
# print(backtest.shape)

# #Train = Year 1
# #Val = Year 2
# #Test = Year 3 (6 months)


# X_train = X.iloc[:45648]
# X_val = X.iloc[45648:91429]
# X_test = X.iloc[91429:]
# y_train = y.iloc[:45648]
# y_val = y.iloc[45648:91429]
# y_test = y.iloc[91429:]
# backtest_train = backtest.iloc[:45648]
# backtest_val = backtest.iloc[45648:91429]
# backtest_test = backtest.iloc[91429:]

#backtesting=data.iloc[91432:]

#X_train=X.iloc[:73753]
#X_val=X.iloc[73753:91432]
#X_test=X.iloc[91432:]
#y_train=y.iloc[:73753]
#y_val=y.iloc[73753:91432]
#y_test=y.iloc[91432:]
