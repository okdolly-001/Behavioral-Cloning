import pandas as pd

import csv
log = pd.read_csv('driving_log.csv')
a = pd.read_csv('a_out.csv')
b= pd.read_csv('b_out.csv')

#
result = log.append([b,a])
#
result.to_csv("output.csv",index=False)

