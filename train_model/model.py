import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import h5py as h5
import pickle

import pandas as pd
import numpy as np

from utils import performance_assessment
from utils_deepgoplus import update_scores
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from IPython import embed

#######################################
########## EXPRESSION-LOCATION ########
#######################################
class ExpNetwork(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.model = self.__build_model()

    def __build_model(self):
        # Set the layer sizes according to the config file
        layer_sizes = [self.config['input_dim']]+self.config['hidden_sizes']
        layers = []

        # Set the activation function
        if 'activation' in self.config.keys():
            activation = self.config['activation']
            activation = eval(f'nn.{activation}()')
        else:
            activation = nn.ReLU()

        # Build the first n-1 non-linear layers
        for i in range(len(layer_sizes)-2):
            layers.extend([nn.Linear(layer_sizes[i],layer_sizes[i+1]),
                               activation,
                               nn.Dropout(self.config['p'])])

        # Output layer
        layers.append(nn.Linear(layer_sizes[-2],layer_sizes[-1]))

        return nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)


#####################
##### STRUCTURE #####
#####################
class ExtractedStructureNetwork(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(3*512,1024)
        self.activation = eval(f"nn.{self.config['activation']}()")
        self.dropout = nn.Dropout(p=config['p'])
        self.fc2 = nn.Linear(1024,config['output_dim'])

        # Load the weights for transfer learning
        self.__set_weights()

    def __set_weights(self):
        weights = h5.File(self.config['model_file'])['model_weights']  
        self.fc1.weight = nn.Parameter(torch.FloatTensor(weights['dense/dense/kernel:0'][:]).permute(1,0))
        self.fc1.bias = nn.Parameter(torch.FloatTensor(weights['dense/dense/bias:0'][:]))

        self.fc2.weight = nn.Parameter(torch.FloatTensor(weights['labels/labels/dense_1/kernel:0'][:,::2]).permute(1,0))
        self.fc2.bias = nn.Parameter(torch.FloatTensor(weights['labels/labels/dense_1/bias:0'][::2]))

    def forward(self,x):
        z = self.fc1(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

#####################
##### COMBINED ######
#####################
class CombinedModel(pl.LightningModule):
    def __init__(self,config, config_train, transfer_terms):
        super().__init__()
        self.config = config
        self.config_train = config_train

        self.experimental = ExpNetwork(self.config['expression'])
        self.structure = ExtractedStructureNetwork(self.config['structure'])
        
        self.sigmoid = nn.Sigmoid()
        self.__set_mask(transfer_terms)
        
        self.combination_parameter = nn.Parameter(torch.ones(2))
        self.criterion = nn.BCELoss()
        self.save_hyperparameters()
    
    def __set_mask(self,transfer_terms):
        with open(self.config['structure']['deepfri_terms'],'rb') as f:
            deepf_terms = pickle.load(f).terms.values.tolist()

        self.all_terms     = list(set(deepf_terms).union(transfer_terms))
        self.deepf_mask    = np.array([self.all_terms.index(i) for i in deepf_terms])
        self.transfer_mask = np.array([self.all_terms.index(i) for i in transfer_terms])
        self.total_gos     = len(self.all_terms)
        self.all_terms     = np.array(self.all_terms)
        self.intersection, self.idx1, self.idx2  = np.intersect1d(self.transfer_mask, self.deepf_mask, return_indices=True)
        assert (self.all_terms[self.transfer_mask]==transfer_terms).all()
        assert (self.all_terms[self.deepf_mask]   ==deepf_terms).all()
    
    def forward(self,x):
        e, struc = x
        out = torch.zeros(e.shape[0],self.total_gos,device=self.device)

        z_1 = self.experimental(e)
        z_3 = self.structure(struc)

        z_1 *= self.combination_parameter[0]
        z_3 *= self.combination_parameter[1]
        
        out[:,self.transfer_mask] += z_1
        out[:,self.deepf_mask]    += z_3

        return self.sigmoid(out)
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat[:,self.transfer_mask],y)
        self.log('train_loss',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat[:,self.transfer_mask],y)
        self.log('val_loss',loss,on_epoch=True)

    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.forward(x)
        return {'y_hat':y_hat,'y':y}
    
    def test_epoch_end(self, outputs):
        y     = torch.concat([x['y'] for x in outputs],dim=0).detach().cpu().numpy()
        y_hat = torch.concat([x['y_hat'] for x in outputs],dim=0).detach().cpu().numpy()

        y_hat_hier = update_scores(pd.DataFrame(data=y_hat,columns=self.all_terms), self.config['go_fn'])
        performance_assessment(y_hat_hier.loc[:,self.all_terms[self.transfer_mask]].values,y)



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'],weight_decay=self.config['weight_decay'])
        print(self.config)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.config_train['warm_up'], self.config_train['max_epochs'])
        return ([optimizer], [lr_scheduler])
        

