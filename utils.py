# Utilities file to carry out simple operations.
# K SAI DINESH  ;  CS20B1122.

# Import necessary operations
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Provide path to data directory.
DATA_DIR = os.path.join(os.getcwd().replace("/notebooks", ""), "data")

# Fantasy point Scoring System.
MULTIPLIERS = {
    "PTS": 1,
    "3P": 0.5,
    "TRB": 1.25,
    "AST": 1.5,
    "STL": 2,
    "BLK": 2,
    "TOV": -0.5,
}

# Mean absolute error function.
def calculate_MAE(pred, true):
    n = len(pred)
    abs_error = 0
    for i in range(n):
        abs_error += abs(pred[i] - true[i])
    mae = abs_error / n
    return mae

# Root Mean Squared Error function.
def calculate_RMSE(pred, true):
    return np.sqrt(mean_squared_error(pred, true))

# Concatenate all csv files in a given folder path.
def csv_concatenate(folder_path, nested=False):
    
    if nested == True:
        files = glob.glob(folder_path + "/*/*.csv")
    else:
        files = glob.glob(folder_path + "/*.csv")

    df_list = []

    for file in files:
        df_list.append(pd.read_csv(file, parse_dates=True, infer_datetime_format=True))

    # Fill NaN with 0s.
    df = pd.concat(df_list).fillna(0).reset_index(drop=True)

    return df

# Function to calculate fantasy points based on the multipliers provided above.
def calculate_FPTS(df):
    indices = len(df)
    fpts_list = []

    for i in tqdm(range(indices)):
        fpts = 0
        doubles = 0
        for stat, multiplier in MULTIPLIERS.items():
            if stat in ["PTS", "TRB", "AST", "STL", "BLK"]:
                if df.loc[i, stat] >= 10:
                    doubles += 1
            fpts += df.loc[i, stat] * multiplier
        if doubles >= 2:
            fpts += 1.5
        if doubles >= 3:
            fpts += 3
        fpts_list.append(fpts)

    return fpts_list
