import pandas as pd
import numpy as np
label_csv_file_path = './train_data/train_label.csv'
df = pd.read_csv(label_csv_file_path,index_col=0)

df.dropna(how='any', inplace=True)
df.to_csv('./train_data/complete_data.csv')
df = pd.read_csv(label_csv_file_path, index_col=0)
df = df.iloc[2284:][['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x','right_eye_center_y'
                     , 'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']]
df.dropna(how='any', inplace=True)
df.to_csv('./train_data/half_data.csv')