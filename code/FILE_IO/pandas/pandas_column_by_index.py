# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:07:32 2021

@author: HUSTAR15
"""

import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

data_frame = pd.read_csv(input_file)
data_frame_column_by_index = data_frame.iloc[:, [0, 3]] #iloc -> 
data_frame_column_by_index.to_csv(output_file, index=False)