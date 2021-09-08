# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:10:25 2021

@author: HUSTAR15
"""

import pandas as pd
import sys

input_file = 'supplier_data_unnecessary_header_footer.csv' #sys.argv[1]
output_file = 'output_pan_3.csv' #sys.argv[2]

data_frame = pd.read_csv(input_file, header=None)

data_frame = data_frame.drop([0,1,2,16,17,18]) #필요없는 ROW 버림
data_frame.columns = data_frame.iloc[0]
data_frame = data_frame.reindex(data_frame.index.drop(3))

data_frame.to_csv(output_file, index=False)