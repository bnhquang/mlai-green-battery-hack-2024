import pandas as pd
import numpy as np
from datetime import datetime

def get_dataset(path, test_data=False):
    df = pd.read_csv(path)
    df = preproces_dataset(df)
    return df

def preproces_dataset(df):
    df.dropna(inplace=True)
    return df


