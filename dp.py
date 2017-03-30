import pandas as pd

#read response data

data = pd.read_csv('data.txt',delim_whitespace =True)
column_id = ['s','e','r']
data.columns = column_id


#read expert label


expert_label = pd.read_csv('expert_label.txt',delim_whitespace = True,header = None)
ex_columns = ['ex_lb']
expert_label.columns = ex_columns
