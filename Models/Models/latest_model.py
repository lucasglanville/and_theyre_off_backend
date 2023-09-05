import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.decomposition import PCA

def train_test_val():
    data = pd.read_csv("data_cleaned_and_preprocessed_v3.csv")
    data = data.sort_values(by='f_ko')
    data=data.query("f_pm_01m < 15 ")
    data = data.reset_index(drop=True)

    test_start_date = '2022-08-20 18:45:00'  # max_date - 1 yr
    val_start_date = '2022-02-20 18:45:00'  # max_date - 1.5 yrs
    # Split the data into test and train
    train = data[data['f_ko'] <= val_start_date]
    val = data[(data['f_ko'] > val_start_date) & (data['f_ko'] <= test_start_date)]
    test = data[data['f_ko'] > test_start_date]
    # Split the data into X and y
    X_train = train.drop(columns=["f_place", "f_pm_15m","f_pm_05m", "f_pm_01m","f_pm_15m_p_back", "f_id", "f_ko"])
    X_val = val.drop(columns = ["f_place","f_pm_15m", "f_pm_05m", "f_pm_01m", "f_pm_15m_p_back", "f_id", "f_ko"])
    X_test = test.drop(columns = ["f_place", "f_pm_15m", "f_pm_05m", "f_pm_01m", "f_pm_15m_p_back", "f_id", "f_ko"])
    y_train = train["f_place"]
    y_val = val["f_place"]
    y_test = test["f_place"]

    backtesting= data[data['f_ko'] > test_start_date]


    pca = PCA(n_components=70)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_test_pca, X_train_pca, X_val_pca, y_test, y_val, y_train, backtesting

def model(X_train_pca, X_val_pca, y_train, y_val):

    def build_model(input_dim):
        model = Sequential()

        # Input layer
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.15))

        # Hidden layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.01))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        return model

    model = build_model(X_train_pca.shape[1])

    model.compile(optimizer=Adam(learning_rate=0.00005), loss="binary_crossentropy", metrics=[Precision()])
    early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(X_train_pca, y_train, batch_size=16, epochs=1000, validation_data=(X_val_pca, y_val), callbacks=[early_stopping])

    return model



def backtesting_func(backtesting, y_pred):
    backtesting["y_pred"]=y_pred
    backtesting["probabilities"]=1/backtesting["f_pm_15m"]
    backtesting["diff"] = backtesting["y_pred"] - backtesting["probabilities"]
    backtesting["bet_0.1"] = backtesting.apply(lambda row: 1 if row['y_pred'] - row['probabilities'] > 0.1 else 0, axis=1)
    backtesting["bet_0.05"] = backtesting.apply(lambda row: 1 if row['y_pred'] - row['probabilities'] > 0.05 else 0, axis=1)
    backtesting["bet_0.15"] = backtesting.apply(lambda row: 1 if row['y_pred'] - row['probabilities'] > 0.15 else 0, axis=1)
    backtesting['bets_placed_0.15'] = backtesting['bet_0.15'].cumsum()
    backtesting['bets_placed_0.1'] = backtesting["bet_0.1"].cumsum()
    backtesting['bets_placed_0.05'] = backtesting['bet_0.05'].cumsum()
    backtesting['profit_0.05']=backtesting['bet_0.05'] * backtesting["f_pm_15m_p_back"]
    backtesting['profit_0.1']=backtesting['bet_0.1'] * backtesting["f_pm_15m_p_back"]
    backtesting['profit_0.15']=backtesting['bet_0.15'] * backtesting["f_pm_15m_p_back"]
    backtesting['cumulative_profit_0.05'] = backtesting['profit_0.05'].cumsum()
    backtesting['cumulative_profit_0.1'] = backtesting['profit_0.1'].cumsum()
    backtesting['cumulative_profit_0.15'] = backtesting['profit_0.15'].cumsum()
    return backtesting
