import os
import torch
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import auc, recall_score, precision_recall_fscore_support, roc_auc_score


def load_pickle(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


def save_preds(model, datamodule, outdir, out_name, mode):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    term_names = model.all_terms
    if mode=='train':
        gene_names,_,_ = datamodule._get_split()
    elif mode=='val':
        _,gene_names,_ = datamodule._get_split()
    elif mode=='test':
        _,_,gene_names = datamodule._get_split()
    else:
        raise ValueError('"mode" needs to be either "train", "val", or "test"!')
    
    gene_names = datamodule.conversion_dict.loc[gene_names,'Locus_tag']
    y_hat_list = []
    for x,_ in tqdm(eval(f'datamodule.{mode}_dataloader()'),leave=False,desc=f'Saving {mode} predictions'):
        x = tuple([s.to(device) for s in x])
        y_hat_list.append(model.forward(x).detach().cpu().numpy())
    
    y_hat = np.vstack(y_hat_list)
    y_hat_df = pd.DataFrame(y_hat,index=gene_names,columns=term_names)
    y_hat_df.to_csv(os.path.join(outdir, out_name.format(mode) + '.csv'),index=True)
    model.train()


def performance_assessment(Y_pred, Y_test, return_vals=False):
    print('\nPerformance assessment')

    sum_terms = Y_test.sum(axis = 0)
    index_0 = sum_terms == 0
    index_1 = sum_terms == Y_pred.shape[0]
    index_keep = np.logical_not(np.logical_or(index_0, index_1))

    ################################## GENE CENTRIC MEASURES ####################################
    # we obtain the recall, precision and f1-score per each protein (column) per threshold = t
    # we threshold the label matrix
    
    thresholds = np.linspace(0, 1, 100)
    rec_prot, prec_prot, f_prot = [], [], []

    precisions, recalls = np.array([]), np.array([])
    precisionsm, recallsm = [], []
    for idx, t in tqdm(enumerate(thresholds),leave=False):
        Y_pred_binary = (Y_pred > t).astype(int)

        p, r, f, s = precision_recall_fscore_support( Y_test.T, Y_pred_binary.T, average=None, zero_division=1)
        if ((Y_pred_binary.sum(axis = 1) > 0).sum() > 0):
            p_av = np.mean(p[ Y_pred_binary.sum(axis = 1) > 0]) # following  https://doi.org/10.1038/nmeth.2340,
            # the precision per each threshold is obtained as the average
            # across proteins having at least one prediction above threshold
        else:
            p_av = 0

        r_av = np.mean(r) # the recall, instead, across all proteins.
        
        # Saving results for threshold t
        if((p_av * r_av) == 0):
            f_prot.append(0)
        else:
            f_prot.append(2*p_av * r_av/(p_av + r_av))

        # micro
        pm, rm, fm, sm = precision_recall_fscore_support(Y_test.flatten(), Y_pred_binary.flatten(), 
            average='binary', zero_division=1)
        precisionsm.append(pm)
        recallsm.append(rm)
        

    F_max = np.max(f_prot)
    print('-------------------------')
    print('gene-centric F-max: {}'.format(F_max))

    
    micro = auc(recallsm, precisionsm)
    print('Micro Term-centric average AUPRC: {}'.format(micro))
    print('-------------------------\n')

    if return_vals:
        return (F_max, micro)

