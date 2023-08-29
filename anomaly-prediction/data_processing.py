import itertools
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DROP_TH = 100
KEEPNODES = None 

class MinMaxScaler():
    def __init__(self,min,max,range_min = 0, range_max=1):
        self.min = min
        self.max = max
        if range_max-range_min <=0:
            raise ValueError("Invalid output range encountered. range_max - range_min =",range_max-range_min)
        self.range_min = range_min
        self.range_max = range_max
        self.mask = np.where(self.max != self.min)
        
    def __call__(self, x):
        x_std = (x -self.min)
        x_std[self.mask] /= (self.max[self.mask]-self.min[self.mask])
        return x_std * (self.range_max-self.range_min) + self.range_min
    

def load_metrics(datapath):
    # Read all dataframes
    df_list = []
    for i in tqdm(range(49)):
        df_list.append(pd.read_parquet(os.path.join(datapath,f'rack_{i}.parquet')).rename(columns={'values':f'values_{i}'}))

    # Aggregate all dataframes into a single one
    data = df_list[0]
    for df in tqdm(df_list[1:]):
        data = pd.merge(data,df,how='outer',on='timestamp')

    # Aggregate values
    values_columns = [f'values_{i}' for i in range(49)]
    data['values']= data[values_columns].agg(np.concatenate,axis=1)
    data = data.drop(values_columns,axis=1)

    return data

def process_metrics(data,keep_nodes = KEEPNODES,drop_th = DROP_TH,verbose=False):

    node_columns = data.columns[1:-1]
    # Drop columns with more than DROP_TH null values
    if verbose: print(f"Dropping columns with more than {drop_th} null values.. ",end='')
    to_drop = data.columns[data.isnull().sum()>DROP_TH].to_list()
    if keep_nodes is not None:
        if verbose: print(f"Adding {drop_th} null values.. ",end='')
        to_drop += [n for n in node_columns if n not in keep_nodes]
    data = data.drop(to_drop,axis=1).dropna()


    # Remove elements relative to dropped columns from the 'values' column and build the 'anomalies' column
    mask = np.ones(len(node_columns),dtype=bool)
    for n in to_drop:
        mask[node_columns==n] = 0
    data['values'] = data['values'].apply(lambda x : x[mask].astype(int))
    data['anomalies'] = data['values'].apply(lambda x : (x>0)|(np.isnan(x)).astype(int))
    
    
    # Rescale features in the [0,1] range
    node_columns = data.columns[1:-3]
    maximum = np.max(np.stack([np.max(np.stack(data[column].values),axis=0) for column in node_columns]),axis=0)
    minimum = np.min(np.stack([np.min(np.stack(data[column].values),axis=0) for column in node_columns]),axis=0)
    scaler = MinMaxScaler(minimum,maximum)
    data[node_columns] = data[node_columns].applymap(scaler)
    
    return data 

def add_window_labels(data,window_size):
    # Apply a sliding window to compute for each node if it raised an anomaly within the window 
    labels = []
    for i in tqdm(range(len(data))):
        if i+window_size<len(data):
            anomalies = np.stack(data['anomalies'].iloc[i:i+window_size].values)   
            labels.append(anomalies.any(axis=0).astype(int))
        else: # Put them to zero when the number of samples is less than the window size, no relevant data come from here
            labels.append(np.zeros(data['anomalies'][0].shape))
    data['labels'] = labels
    return data

def get_edgelist_racks(racks,nodes):
    racks['nodes'] = racks.nodes.map(lambda a : [e for e in a if e in nodes])
    racks=racks[racks['nodes'].map(len)>0]   
    edge_list = racks.nodes.map(lambda x: list(itertools.permutations(x,2))) # build edgelist for each node
    return torch.from_numpy(np.concatenate(edge_list.values)).long().T
    

def get_edgelist_jobs(jobs,nodes):
    heatmap = np.zeros((980,980))
    if len(jobs)>0:
        edge_list = jobs.nodes.map(lambda x: list(itertools.permutations(x, 2))) # nodes with a single job
        edge_list = edge_list[edge_list.map(len) > 0]
        if len(edge_list) > 0:
            heatmap = build_heatmap(np.concatenate(edge_list.values))
    
    edge_list = torch.nonzero(torch.from_numpy(heatmap))
    eweights = torch.tensor(heatmap[heatmap.nonzero()],dtype=torch.float32)

    edge_list = torch.cat((edge_list,torch.tensor(nodes).repeat((2,1)).T)).apply_(lambda x: nodes.index(x))
    eweights = torch.cat((eweights,torch.full((len(nodes),),1)))

    return edge_list.T,eweights

def build_heatmap(edges):
    # Find the maximum node index to determine the dimensions of the heatmap matrix
    max_node = np.max(edges)
    # Initialize the heatmap matrix with zeros
    heatmap = np.zeros((max_node + 1, max_node + 1), dtype=int)

    # Increment the corresponding cells in the heatmap matrix
    np.add.at(heatmap, (edges[:, 0], edges[:, 1]), 1)
    return heatmap

def compute_labels(data,window_size):
    labels = []
    for i in tqdm(range(len(data))):
        if i+window_size<len(data):
            anomalies = np.stack(data['anomalies'].iloc[i:i+window_size].values)
            labels.append(anomalies.any(axis=0).astype(int))
        else: # Put them to zero when the number of samples is less than the window size, no relevant data come from here
            labels.append(np.zeros(data['anomalies'][0].shape))
    return torch.tensor(labels)

def train_test_split(data,train_split):
  train_size = int(len(data)*train_split)
  train_data = data[:train_size]
  test_data = data[train_size:]

  return train_data, test_data