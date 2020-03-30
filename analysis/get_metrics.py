import pandas as pd
import numpy as np
import sys

sys.path.append('../../../metrics/')

from moses.metrics import get_all_metrics

idata_prefix = '../../ReLeaSE/data/smi/'
odata_prefix = '../../ReLeaSE/data/csv/'

train_data = '../../ReLeaSE/data/chembl_22_clean_150000_sorted_std_final_sample.smi'
gen_data = idata_prefix+'logp/step/10000/1/RL_gen_logp_biased.smi'

train_out = odata_prefix+'training/'
gen_out = odata_prefix+'logp/step/10000/1/'

training = []
f1 = open(train_data, 'r')
for i in f1.readlines():
    training.append(i.split()[0])

generated = []
f2 = open(gen_data, 'r')
for i in f2.readlines():
    generated.append(i.strip())

print (training[:5])
print (generated[:5])

metrics = get_all_metrics(generated, n_jobs=4, device='cpu', train=training)

print ('metrics: ', metrics)