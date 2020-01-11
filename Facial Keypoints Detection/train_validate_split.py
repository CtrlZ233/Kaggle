import numpy as np
import pandas as pd


def split(path, ratio):
    df = pd.read_csv(path, index_col=0)
    train_set = df.iloc[:int(df.shape[0] * ratio)]
    validation_set = df.iloc[int(df.shape[0] * ratio):]
    return train_set, validation_set

if __name__ == '__main__':
    complete_path = './train_data/complete_data.csv'
    half_path = './train_data/half_data.csv'
    ratio = 0.7
    train_set, validation_set = split(complete_path, ratio)
    train_set.to_csv("./train_data/train_complete_data.csv")
    validation_set.to_csv('./train_data/validate_complete_data.csv')

    train_set, validation_set = split(half_path, ratio)
    train_set.to_csv("./train_data/train_half_data.csv")
    validation_set.to_csv('./train_data/validate_half_data.csv')
