from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parent
p=pd.read_csv(ROOT/'outputs/test_exploratory_scores.csv')
y=p.label.to_numpy(int)
rng=np.random.default_rng(20261009)
pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
pairs=[('raw_mean','raw_max_original'),('raw_power_mean_p4','raw_max_original'),('raw_power_mean_p2','raw_max_original'),('raw_mean','raw_power_mean_p4')]
B=500
rows=[]
for a,b in pairs:
 diffs=[]
 for _ in range(B):
  ix=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
  diffs.append(average_precision_score(y[ix],p[a].to_numpy()[ix])-average_precision_score(y[ix],p[b].to_numpy()[ix]))
 diffs=np.array(diffs)
 rows.append({'comparison':f'{a} minus {b}','bootstrap_replicates':B,'mean_ap_difference':float(diffs.mean()),'ci95_low':float(np.quantile(diffs,.025)),'ci95_high':float(np.quantile(diffs,.975)),'prob_difference_gt_zero':float(np.mean(diffs>0))})
out=pd.DataFrame(rows)
out.to_csv(ROOT/'outputs/paired_bootstrap_test_exploratory.csv',index=False)
print(out.to_string(index=False))
