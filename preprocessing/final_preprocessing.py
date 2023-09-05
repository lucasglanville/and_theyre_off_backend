from preprocessing.data_clean import get_data, dropping_no_betting_data
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from keras.utils import get_custom_objects
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')



#######################################################

"""1)  Getting the data and first round of preprocessing"""
###receiving all columns of data for user selected rows (by track and time)

#Get the data from the bucket
data = get_data("raw_data/hr_data_0409_221rem.csv") ### CHANGE THIS PATH to get data from the bucket
print("data loaded")
#Fill in the missing odds
data = dropping_no_betting_data(data)
#Drop stall_position NAs
data = data[data['stall_position'].notna()]
#Remove horses with odds over 50 at 5m before the race
data = data[(data['f_pm_05m'] <= 50)]
#Reset index
data = data.reset_index(drop=True)

#######################################################

"""2)  Filling the null L16 columns with 0s"""

X_preproc = data[[
          'iv_trainer_l16r', 'iv_jockey_l16r',
          'ae_trainer_l16r' ,'ae_jockey_l16r']]
X_preproc = X_preproc.fillna(0)

#######################################################

"""3)  Scaling the numerical values and defining X"""

#Adding f_runners and stall_position to X_preproc pre-scaling
X_preproc['f_runners'] = data['f_runners']
X_preproc['stall_position'] = data['stall_position']


#Loading scaler values and scaling 5 features
set_config(transform_output = "pandas")
with open('Models/weights-JStone2609/scaler_updated2.pkl', 'rb') as f: # CHANGE THIS PATH to get the saved scalar
    loaded_scaler = pickle.load(f)
X = loaded_scaler.transform(X_preproc)

#Adding final 2 features that don't need scaling
X['pred_isp_prob'] = 1 / data['pred_isp']

#Matching the column order to the order of the original saved weights
X = X[['stall_position', 'iv_trainer_l16r', 'iv_jockey_l16r', 'ae_trainer_l16r', 'ae_jockey_l16r', 'pred_isp_prob', 'f_runners']]

#######################################################

#"""4)  Defining y"""

# NOT NEEDED IN CURRENT DESIGN (but keeping incase it becomes useful)

# def winner(x):
#         if x == 1:
#             return 1
#         else:
#             return 0
# data['f_place'] = data['f_place'].apply(winner)
# y = data[['f_place', 'pred_isp']]

# y = data[['f_place', 'pred_isp']]

#######################################################

"""5) Defining backtest and changing commision to 2%"""

# Define a function to create a new profit column with 2% commision
def fill_01m_profit(data):
    def fill_nan(row):
        if row['f_place'] == 0:
            return -1
        elif row['f_place'] == 1:
            return (row['f_pm_01m'] - 1) * 0.98
        else:
            return row['f_pm_01m_p_back']

# Apply the lambda function to create 01m_profit column
    data['01m_profit'] = data.apply(fill_nan, axis=1)
    return data

data = fill_01m_profit(data)

backtest = data[['f_ko', 'f_track', 'f_id', 'id','f_horse','f_pm_01m', '01m_profit', 'f_place']]

#######################################################

"""6)  Model Architecture"""

NN = Sequential()
NN.add(InputLayer(input_shape=(7, ))) # input layer
NN.add(Dense(32, activation='relu')) # hidden layer 1
NN.add(Dense(2, activation='softmax')) # output layer

#######################################################

"""7)  Loading Weights"""

NN.load_weights("Models/weights-JStone2609/custom_scorer0.05_7input_l16_05mfilter_01mplace") ##CHANGE PATH TO LOAD MODEL WEIGHTS

#######################################################

"""8)  Creating preds"""

y_pred = NN.predict(X)

#######################################################

"""9) Creating backtest table"""

backtest['model_preds'] = y_pred[:, 0:1]
backtest['model_preds'] = round(backtest['model_preds'],2)
backtest = backtest.sort_values(['model_preds'], ascending = False)
backtest_live = backtest.drop(columns=['f_id', 'id','f_place'])
def bet_or_nobet(f_pm_01m, model_preds):
    if f_pm_01m < 50 and model_preds > 0.9:
        return "BET"
    else:
        return "NO BET"

backtest_live['bet'] = backtest_live.apply(lambda row: bet_or_nobet(row['f_pm_01m'], row['model_preds']), axis=1)

backtest_live
