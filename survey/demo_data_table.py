import pandas as pd
import json
import glob

def create_data_table():
    file_names = glob.glob("data-all/*.json")
    data_table = pd.DataFrame()                         # create empty dataframe.
    for i, file_name in enumerate(file_names):          # enumerate files > to set columns with keys of the first file accordingly.
        data = json.load(open(file_name))
        main_key = ['demographicsData']
        mini_dict = [data.get(k) for k in main_key][0]  # get first value(=dict) of the demographic key
        if i == 0:                                      # when the first file is processed
            cols = list(mini_dict.keys())               # obtain list of its keys
            data_table = pd.DataFrame(columns = cols)   # set these keys as the column names of data table. this ensures that when concatenating dataframes together, the column will pair up correctly
        df = pd.DataFrame([mini_dict])                  # if df is created in this way, mini_dict.keys will be the df column names, and mini_dict.values will be the first (and only) row.
        data_table = pd.concat([data_table, df])        # concat both data frames. because they both have column names defined, they will concatenate correctly.
    data_table.index = range(len(file_names))           # set index for each row in the range of length of files
    print(data_table)
    return data_table

create_data_table()