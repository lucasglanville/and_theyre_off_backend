import pandas as pd

## Dropping rows if we have no betting data, and imputing if we have some betting data
def dropping_no_betting_data(data):

    data = data.dropna(subset=['f_pm_15m', 'f_pm_10m', 'f_pm_05m' , 'f_pm_03m', 'f_pm_02m', 'f_pm_01m'], how='all')

    def impute_row(row):
        row = row.fillna(method='ffill').fillna(method='bfill')
        return row

    columns_to_impute = ['f_pm_15m', 'f_pm_10m', 'f_pm_05m', 'f_pm_03m', 'f_pm_02m', 'f_pm_01m']

    data[columns_to_impute] = data[columns_to_impute].apply(impute_row, axis=1)
    return data
