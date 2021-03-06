# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:13:00 2021

@author: HUSTAR15
"""

import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

data_frame = pd.read_csv(input_file)

important_dates = ['1/20/14', '1/30/14']
data_frame_value_in_set = data_frame.loc[data_frame['Purchase Date'].isin(important_dates), :]
data_frame_value_in_set.to_csv(output_file, index=False)