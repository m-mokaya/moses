import pandas as pd
import numpy as numpy
import sys

sys.path.append('../moses/metrics')

import moses.metrics

data_prefix = '../../ReLeaSE/data/smi/'

data1 = data_prefix+'logp/step/2/RL_gen_logp_unbiased.smi'
data2 = data_prefix+'logp/step/2/RL_gen_logp_biased.smi'

unbiased = []
f1 = open(data1, 'r')
for i in f1.readlines():
    unbiased.append(i)

biased = []
f2 = open(data2, 'r')
for i in f2.readlines():
    biased.append()

metrics = moses.metrics.get_all_metrics(unbiased,device='cpu')

print ('Metrics: ', metrics)