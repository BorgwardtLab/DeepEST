'''
This is the script for traning DeepEST
'''

import os
import sys
import yaml
import time
import torch
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from IPython import embed 

from datamodule import CombinedDataModule, CombinedDataset
from model import CombinedModel
from utils import save_preds



def main(args):

    if (not os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # Seed everything
    pl.seed_everything(47, workers=True)

    # Load config
    with open(args.config,'r') as f: # this was "config_combined.yaml"
        config = yaml.safe_load(f)


    # Load data
    data = CombinedDataModule(config['data_params'], args.splitdir, 
        args.split, args.expr_loc, args.structures,args.label,
        args.conversion_dict)
    
    data.setup()


    # Find the GO terms for the species in question
    transfer_terms = data.label.columns.values.tolist()

    # Load model
    model = CombinedModel(config['model_params'], config['training_params'], transfer_terms)

    # Checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = 'trained_models/',
                                                    filename = "{}_{}".format(args.species, args.split),
                                                    monitor = 'val_loss',
                                                    mode = 'min',
                                                    save_top_k = 1,
                                                    every_n_epochs = 10,
                                                    verbose = True)
    
    # Early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor = 'val_loss',
                                                        mode = 'min',
                                                        min_delta = 0.0,
                                                        patience = 4)

    # Trainer
    trainer = pl.Trainer(max_epochs = config['training_params']['max_epochs'],
                        enable_progress_bar = True,
                        callbacks=[checkpoint_callback,early_stopping_callback],
                        log_every_n_steps=1,
                        check_val_every_n_epoch = 5,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        devices=1 if torch.cuda.is_available() else None,
                        deterministic=True)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Fit
    print(f"Training started at {time.asctime(time.localtime(time.time()))}.") 
    trainer.fit(model,datamodule=data)
    print(f"Training finished at {time.asctime(time.localtime(time.time()))}.") 

    # Uncomment if you want to test here.
    trainer.test(datamodule=data,ckpt_path='best')
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    save_preds(model, data, args.outdir, "DeepEST_{}_{}_{}.csv".format(args.species, {}, args.split),
        mode='val')
    save_preds(model,data,args.outdir, "DeepEST_{}_{}_{}.csv".format(args.species, {}, args.split),
        mode='test')

    

def parse_arguments():
    '''
    Definition of the command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required = True, 
        type=int, help = 'which split to use')
    parser.add_argument('--splitdir', required = True, 
        help = 'directory of the split')
    parser.add_argument("--config", required = True, 
        help = "config file, specific per fold and species and split")
    parser.add_argument('--expr-loc', required = True, 
        help = 'file with expression-location data')
    parser.add_argument('--structures', required = True, 
        help = 'file with the structures')
    parser.add_argument('--label', required = True, 
        help = 'file with the matrix label')
    parser.add_argument('--conversion_dict', required = True, 
        help = 'conversion dictionary between locus tag and protein id')
    parser.add_argument('--outdir', required = True, 
        help = 'output directory')
    parser.add_argument('--species', required = True, 
        help = 'output directory')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
