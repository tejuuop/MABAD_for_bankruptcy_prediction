#!/usr/bin/env python3
"""Leakage-aware extension of Phase 2 candidate aggregation experiment.
Uses only saved modality-distance CSV; no model retraining or imputer.pkl.
Validation 5-fold CV is the primary comparison. Existing test split is exploratory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'/'active_finbert_mabad_valtest_modality_distances.csv'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MODS=['d_financial','d_mda','d_auditor']

def pct(reference, frame):
    arr=[]
    for c in MODS:
        x=np.sort(reference[c].to_numpy(float)); v=frame[c].to_numpy(float)
        arr.append(np.searchsorted(x,v,side='right')/(len(x)+1.0))
    return np.column_stack(arr)

def iqr_weights(reference):
    q=np.array([reference[c].quantile(.75)-reference[c].quantile(.25) for c in MODS],float)
    return q/q.sum() if np.isfinite(q).all() and q.sum()>0 else np.ones(3)/3

def scores(frame, rank_matrix, weights):
    x=frame[MODS].to_numpy(float)
    return {
      'original_raw_max':x.max(axis=1),
      'raw_mean':x.mean(axis=1),
      'raw_power_mean_p4':np.mean(x**4,axis=1)**0.25,
      'rank_equal_mean':rank_matrix.mean(axis=1),
      'rank_iqr_weighted':rank_matrix@weights,
      'rank_financial_damped_0.1_1_1':rank_matrix@np.array([.1,1,1])/2.1,
      'rank_financial_only':rank_matrix[:,0],
      'rank_mda_only':rank_matrix[:,1],
      'rank_auditor_only':rank_matrix[:,2],
      'rank_text_equal_mean':rank_matrix[:,1:].mean(axis=1),
    }

def metrics(method, y, s, split, fold=None):
    order=np.argsort(-s,kind='mergesort'); out={'method':method,'split':split,'n':len(y),'positives':int(np.sum(y)),
      'average_precision':float(average_precision_score(y,s)),
      'roc_auc':float(roc_auc_score(y,s)) if len(np.unique(y))>1 else np.nan}
    if fold is not None: out['fold']=fold
    for k in [20,50,100]:
      kk=min(k,len(y)); out[f'precision_at_{k}']=float(np.sum(y[order[:kk]])/kk)
      out[f'recall_at_{k}']=float(np.sum(y[order[:kk]])/max(1,int(np.sum(y))))
    return out

def main():
    df=pd.read_csv(INPUT).dropna(subset=['split','label',*MODS]).copy()
    df['label']=df.label.astype(int)
    val=df[df.split.astype(str).str.lower().isin(['val','validation'])].reset_index(drop=True)
    test=df[df.split.astype(str).str.lower().eq('test')].reset_index(drop=True)
    y=val.label.to_numpy(); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fold_rows=[]; oof={}
    for fold,(tr_i,va_i) in enumerate(cv.split(val,y),1):
      tr,va=val.iloc[tr_i],val.iloc[va_i]; ranks=pct(tr,va); w=iqr_weights(tr)
      cand=scores(va,ranks,w)
      for name,sc in cand.items():
        row=metrics(name,va.label.to_numpy(),sc,f'validation_fold_{fold}',fold)
        row.update({'weight_financial':w[0],'weight_mda':w[1],'weight_auditor':w[2]})
        fold_rows.append(row)
        oof.setdefault(name,np.full(len(val),np.nan))[va_i]=sc
    fold_df=pd.DataFrame(fold_rows); fold_df.to_csv(OUT/'extended_validation_fold_metrics.csv',index=False)
    summary=fold_df.groupby('method').agg(mean_ap=('average_precision','mean'),sd_ap=('average_precision','std'),mean_auc=('roc_auc','mean'),sd_auc=('roc_auc','std'),mean_precision_at_100=('precision_at_100','mean'),mean_recall_at_100=('recall_at_100','mean')).reset_index()
    pooled=[]
    for name,sc in oof.items(): pooled.append(metrics(name,y,sc,'validation_oof_pooled'))
    pooled_df=pd.DataFrame(pooled).rename(columns={'average_precision':'pooled_oof_ap','roc_auc':'pooled_oof_auc'})
    summary=summary.merge(pooled_df[['method','pooled_oof_ap','pooled_oof_auc']],on='method',how='left').sort_values('mean_ap',ascending=False)
    summary.to_csv(OUT/'extended_validation_summary.csv',index=False)
    # Test results only for transparency; never use to choose the method.
    ranks_val=pct(val,test); wt=iqr_weights(val); test_scores=scores(test,ranks_val,wt)
    test_rows=[metrics(n,test.label.to_numpy(),s,'test_exploratory') for n,s in test_scores.items()]
    pd.DataFrame(test_rows).to_csv(OUT/'extended_test_exploratory.csv',index=False)
    diag={'n_rows':len(df),'validation_n':len(val),'validation_positives':int(y.sum()),'test_n':len(test),'test_positives':int(test.label.sum()),
      'years_by_split':{str(k):{str(yr):{'n':int(len(g)),'positives':int(g.label.sum())} for yr,g in z.groupby('year')} for k,z in df.groupby('split')},
      'validation_iqr':{c:float(val[c].quantile(.75)-val[c].quantile(.25)) for c in MODS},
      'validation_iqr_weights':dict(zip(MODS,map(float,wt))),
      'availability_columns_constant_one':{c:bool(c in df and df[c].nunique(dropna=False)==1 and float(df[c].iloc[0])==1.0) for c in ['financial_available','mda_available','auditor_available']},
      'warning':'Five-fold validation has only 19 positives overall. Existing test has been examined in prior work; all test metrics are exploratory, not confirmatory.'}
    (OUT/'extended_run_metadata.json').write_text(json.dumps(diag,indent=2))
    print(summary.to_string(index=False))
    print('\nDiagnostics:',json.dumps(diag,indent=2))
if __name__=='__main__': main()
