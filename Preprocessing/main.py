from data_clean import remove_221_rows, dropping_no_betting_data, josh_features, class_or_rating_average, oli_features
from v2_preprocessor import preprocess_features_v2
import pandas as pd

data  = pd.read_csv("raw_data/raw_data_v2.2.csv")

data_cleaned = remove_221_rows(data)
data_cleaned = dropping_no_betting_data(data_cleaned)
data_cleaned = josh_features(data_cleaned)
data_cleaned = class_or_rating_average(data_cleaned)
data_cleaned = oli_features(data_cleaned)



preprocessed_data = preprocess_features_v2(data_cleaned)

preprocessed_data.to_csv("raw_data/data_cleaned_and_preprocessed.csv", index=False)

data = pd.read_csv("raw_data/data_cleaned_and_preprocessed.csv")
data.sort_values(by='f_ko')

X = data.drop(columns=["f_place", "f_ko", "f_pm_15m", "f_pm_05m", "f_pm_01m", "f_pm_15m_p_back", "f_id"])
y = data["f_place"]

backtesting=data.iloc[91432:]

X_train=X.iloc[:73753]
X_val=X.iloc[73753:91432]
X_test=X.iloc[91432:]
y_train=y.iloc[:73753]
y_val=y.iloc[73753:91432]
y_test=y.iloc[91432:]
