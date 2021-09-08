# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:09:47 2021

@author: HUSTAR15
"""

import sys
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]

data_frame = pd.read_csv(input_file)
print(data_frame)
data_frame.to_csv(output_file, index=False)