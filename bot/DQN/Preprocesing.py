import pandas as pd
import numpy as np
from datetime import datetime

def get_dataset(path, test_data=False):
    df = pd.read_csv(path)
    df = preproces_dataset(df)
    df_train = df.iloc[:-500]
    df_test = df.iloc[-500:]
    return df_train if test_data is False else df_test

def preproces_dataset(df):
    df.drop('station_id', axis=1, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64)
    # print(len(df))
    df.dropna(inplace=True)
    # print(len(df))
    # df = df[:2]
    # print(df['timestamp'])
    return df


