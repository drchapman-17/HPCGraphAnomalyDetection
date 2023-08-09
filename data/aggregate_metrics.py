import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm
import glob 
from datetime import datetime
from datetime import timezone 
import sys

SPATH = 'C:/Users/tomma/Documents/Uni/Magistrale/AI_Industry/hpc-project/data'
DPATH = 'C:/Users/tomma/Documents/Uni/Magistrale/AI_Industry/hpc-project/data/aggregated'
START_TIME = datetime(2022,3,31,2,30,12,0000,tzinfo=timezone.utc)
END_TIME = datetime(2022,4,30,23,56,57,0000,tzinfo=timezone.utc)


def main():
    if len(sys.argv)!=2:
        raise Exception('Please provide the rack number [0-48]')
    
    rack = sys.argv[1]
    nodes = []
    df_list = []
    data_path = os.path.join(SPATH,str(rack))
    print(f'Reading data from {data_path}.. ', end='')
    filenames = glob.glob('*.parquet',root_dir=data_path)
    for filename in tqdm(filenames,position=0):
        df = pd.read_parquet(os.path.join(data_path,filename))
        df.dropna(inplace = True)
        df = df[(df.timestamp>=START_TIME)&(df.timestamp<=END_TIME)]
        df.reset_index(drop=True, inplace = True)
        df_list.append(df)
    nodes = nodes + sorted([int(f[:-8]) for f in filenames])
    # Data processing (fill null values, normalization, transformations, ...) should be applied here
    print('done!')


    # Aggregate all nodes features into a single dataframe where each column represents all the features of a given node at time t
        
    new_df_list = [] 
    all_columns = np.load(os.path.join(SPATH,'columns.npy'),allow_pickle=True)

    def uniform_and_compress(df,node):
        columns = df.columns[1:-1]
        for column in all_columns:
            if column not in columns:
                df[column] = np.zeros(len(df))
        df[str(node)] = df[all_columns].apply(np.array,axis=1)
        df = df.rename(columns={'value':f'value_{node}'})
        return df.drop(all_columns,axis=1)
        
    print(f'Aggregating values.. ', end='')
    for node,df in tqdm(zip(nodes,df_list)): 
        new_df_list.append(uniform_and_compress(df,node))
    print('done!')

    # We then join all the dataframes using the timestamp as the key value.
    # We are applying an **outer** join, so samples that do not match still 
    # appear in the final result which may contain null values for nodes 
    # without a sample at certain timestamps.

    print(f'Joining data.. ', end='')
    dfone = new_df_list[0]
    for df in tqdm(new_df_list[1:]):
        dfone = pd.merge(dfone,df,how='outer',on='timestamp')

    value_columns = [f'value_{i}' for i in nodes]
    dfone['values'] = dfone[value_columns].apply(np.array,axis=1)
    dfone.drop(value_columns,axis=1,inplace=True)

    print('done!')
    dfone.to_parquet(os.path.join(DPATH,f'rack_{rack}.parquet'))
    print(f'Aggregated data has been saved at {os.path.join(DPATH,f"rack_{rack}.parquet")}')

if __name__ == '__main__':
    main()