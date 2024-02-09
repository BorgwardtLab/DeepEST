import os
from glob import glob

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose       import ColumnTransformer

from utils import load_pickle
from IPython import embed

class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, config, splits_dir, split, expr_loc, structures, matrix_label, conv_dict):
        super().__init__()
        self.config = config
        self.exp_data = pd.read_csv(expr_loc, index_col = 0)
        self.struc_data = load_pickle(structures)
        self.label = pd.read_csv(matrix_label, index_col = 0)
        self.conversion_dict = pd.read_csv(conv_dict,
            names=["Locus_tag", "Protein_id", "No"], sep = " ")
        self.split = split
        self.splitdir = splits_dir


    @staticmethod
    def __check_order(exp_data, label, conversion_dict):
        assert np.all(exp_data.index == label.index)
        assert np.all(exp_data.index == conversion_dict.Locus_tag)


    @staticmethod
    def __normalize_exp(exp_data,train_idx,val_idx,test_idx):
        # Splitting data
        exp_train = exp_data.iloc[train_idx]
        exp_val   = exp_data.iloc[val_idx]
        exp_test  = exp_data.iloc[test_idx]

        # Normalizing the data
        scaler = MinMaxScaler().fit(exp_train.values)
        exp_train = scaler.transform(exp_train.values)
        exp_val = scaler.transform(exp_val.values)
        exp_test  = scaler.transform(exp_test.values)

        return exp_train,exp_val,exp_test

    def setup(self,stage=None):
        train_idx,val_idx,test_idx = self._get_split()

        self.__check_order(self.exp_data, self.label, self.conversion_dict)

        exp_train,exp_val,exp_test = self.__normalize_exp(self.exp_data,train_idx,val_idx,test_idx)

        self.train_set = CombinedDataset(self.config, exp_train, self.struc_data, 
            self.conversion_dict.iloc[train_idx], self.label.iloc[train_idx])
        self.val_set   = CombinedDataset(self.config, exp_val,   self.struc_data, 
            self.conversion_dict.iloc[val_idx], self.label.iloc[val_idx])
        self.test_set  = CombinedDataset(self.config, exp_test,  self.struc_data, 
            self.conversion_dict.iloc[test_idx], self.label.iloc[test_idx])
    
    def _get_split(self):
        train_idx = np.load(os.path.join(self.splitdir,f'train_index_{self.split}.npy'))
        val_idx   = np.load(os.path.join(self.splitdir,f'val_index_{self.split}.npy'))
        test_idx  = np.load(os.path.join(self.splitdir,f'test_index_{self.split}.npy'))

        return train_idx,val_idx,test_idx

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.config['batch_size'],
            num_workers = self.config['num_workers'], shuffle = False)
    def val_dataloader(self):
        return DataLoader(self.val_set,  batch_size = self.config['batch_size'],
            num_workers = self.config['num_workers'], shuffle = False)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.config['batch_size'],
            num_workers = self.config['num_workers'], shuffle = False)

class CombinedDataset(Dataset):
    def __init__(self, config, exp, struc, sample_conversion, label=None):
        super().__init__()
        self.exp = exp
        self.struc = struc
        self.sample_conversion = sample_conversion
        if label is None:
            self.label = None
        else:
            self.label = label.values
        self.config = config
    
    def __getitem__(self,idx):
        protein_id = self.sample_conversion.iloc[idx]['Protein_id']

        exp_features   = torch.FloatTensor(self.exp[idx])
        struc_features = torch.FloatTensor(self.struc[protein_id])

        if self.label is not None:
            label =  torch.FloatTensor(self.label[idx])
            return (exp_features,struc_features),label
        else:
            return (exp_features,struc_features)

    def __len__(self):
        return len(self.sample_conversion)
