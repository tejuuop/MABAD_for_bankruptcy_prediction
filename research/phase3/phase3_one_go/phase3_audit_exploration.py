#!/usr/bin/env python3
"""Phase 3 one-go audit of post-hoc MABAD distance aggregation.

Uses the saved Phase 0.5 distance CSV only; no MABAD retraining or imputer.pkl.
Primary analysis: 5-fold out-of-fold predictions on the 2018 validation split.
Later-period test results are emitted for audit/context only and MUST NOT be used
for choosing a method because the test split has already been inspected.

All learned percentile references/IQR weights are fitted on each CV training fold.
The label is used only for evaluation and stratification, never to fit weights.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs' / 'active_finbert_mabad_valtest_modality_distances.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
MODS = ['d_financial', 'd_mda', 'd_auditor']
SEED = 20261011
N_SPLITS = 5


def percentile_matrix(reference, frame):
    """Map each modality to its empirical upper-tail percentile, train-reference only."""
    cols=[]
    for c in MODS:
        ref=np.sort(reference[c].to_numpy(dtype=float))
        vals=frame[c].to_numpy(dtype=float)
        cols.append(np.searchsorted(ref, vals, side='right') / (len(ref)+1.0))
    return np.column_stack(cols)


def iqr_weights(reference):
    q=np.array([reference[c].quantile(.75)-reference[c].quantile(.25) for c in MODS], dtype=float)
    if (not np.isfinite(q).all()) or q.sum() <= 0:
        return np.ones(3)/3
    return q/q.sum()


def score_candidates(train, held):
    raw=held[MODS].to_numpy(dtype=float)
    ranks=percentile_matrix(train, held)
    w_iqr=iqr_weights(train)
    # Fixed, label-independent sensitivity values specified before evaluation.
    candidates={
      'raw_max_original': raw.max(axis=1),
      'raw_mean': raw.mean(axis=1),
      'raw_median': np.median(raw, axis=1),
      'raw_power_mean_p2': np.mean(np.maximum(raw,0)**2,axis=1)**0.5,
      'raw_power_mean_p4': np.mean(np.maximum(raw,0)**4,axis=1)**0.25,
      'rank_equal_mean': ranks.mean(axis=1),
      'rank_iqr_weighted': ranks @ w_iqr,
      'rank_financial_weight_0': ranks @ np.array([0,.5,.5]),
      'rank_financial_weight_005': ranks @ np.array([.05,.475,.475]),
      'rank_financial_weight_01': ranks @ np.array([.1,.45,.45]),
      'rank_financial_weight_02': ranks @ np.array([.2,.4,.4]),
      'rank_equal_weight': ranks @ np.array([1/3,1/3,1/3]),
    }
    return candidates, w_iqr


def safe_auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y)) > 1 else float('nan')


def metric_dict(name, y, s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    order=np.argsort(-s,kind='mergesort')
    d={'method':name,'n':int(len(y)),'positives':int(y.sum()),
       'prevalence':float(y.mean()),'average_precision':float(average_precision_score(y,s)),
       'roc_auc':safe_auc(y,s)}
    for k in [20,50,100]:
        kk=min(k,len(y))
        d[f'precision_at_{k}']=float(y[order[:kk]].sum()/kk)
        d[f'recall_at_{k}']=float(y[order[:kk]].sum()/max(1,int(y.sum())))
    return d


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Input not found: {INPUT}\nPlace the provided CSV in inputs/.')
    df=pd.read_csv(INPUT)
    required={'split','label',*MODS}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df=df.dropna(subset=['split','label',*MODS]).copy()
    df['label']=df['label'].astype(int)
    split=df['split'].astype(str).str.lower()
    val=df[split.isin(['val','validation'])].reset_index(drop=True)
    test=df[split.eq('test')].reset_index(drop=True)
    if val.empty or test.empty: raise ValueError("Expected split names 'val'/'validation' and 'test'.")
    y=val.label.to_numpy(dtype=int)
    if val.label.nunique()!=2: raise ValueError('Validation must contain both classes.')

    # Input and distribution audit.
    audit={
      'input_file':str(INPUT.name),'total_rows_after_required_column_dropna':int(len(df)),
      'validation_rows':int(len(val)),'validation_positives':int(y.sum()),
      'validation_prevalence':float(y.mean()),'test_rows':int(len(test)),
      'test_positives':int(test.label.sum()),'test_prevalence':float(test.label.mean()),
      'seed':SEED,'n_splits':N_SPLITS,
      'test_usage_warning':'Previously inspected: exploratory only, do not select/tune using test metrics.',
      'label_used_to_fit_aggregation_weights':False,
    }
    if 'company_id' in val:
      audit['validation_unique_companies']=int(val.company_id.nunique())
      audit['validation_rows_with_repeated_company_id']=int(val.company_id.duplicated(keep=False).sum())
    if 'date' in val:
      audit['validation_date_min']=str(val.date.min()); audit['validation_date_max']=str(val.date.max())
    if 'year' in val:
      audit['validation_years']=[int(x) for x in sorted(pd.to_numeric(val.year,errors='coerce').dropna().unique())]

    # Score distributions and positive-vs-negative diagnostics; descriptive only.
    dist=[]
    for c in MODS:
      for subset_name, sub in [('validation_all',val),('validation_negative',val[val.label==0]),('validation_positive',val[val.label==1]),('test_exploratory_all',test)]:
        x=pd.to_numeric(sub[c],errors='coerce').dropna()
        dist.append({'modality':c,'subset':subset_name,'n':int(len(x)),
          'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),
          'q25':float(x.quantile(.25)),'median':float(x.median()),'q75':float(x.quantile(.75)),
          'iqr':float(x.quantile(.75)-x.quantile(.25)),'max':float(x.max()),
          'unique_values':int(x.nunique())})
    pd.DataFrame(dist).to_csv(OUT/'modality_distance_diagnostics.csv',index=False)
    corr=val[MODS].corr(method='spearman').reset_index(names='modality')
    corr.to_csv(OUT/'validation_spearman_correlations.csv',index=False)

    # Use grouped folds where possible to reduce company-level leakage; validation
    # currently appears to have nearly one row/company, but audit this explicitly.
    use_group=bool(StratifiedGroupKFold is not None and 'company_id' in val and val.company_id.nunique()<len(val))
    if use_group:
      splitter=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
      folds=splitter.split(val,y,groups=val.company_id.astype(str).to_numpy())
      audit['cv_splitter']='StratifiedGroupKFold(company_id)'
    else:
      splitter=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
      folds=splitter.split(val,y)
      audit['cv_splitter']='StratifiedKFold(rows; no repeated company groups available or sklearn unsupported)'

    # OOF prediction table: each validation row gets one score from a model whose
    # percentile reference/weights were fit without that row.
    methods=None
    oof={}
    fold_rows=[]; fold_weight_rows=[]
    for fold,(tr_idx,va_idx) in enumerate(folds,start=1):
      tr=val.iloc[tr_idx].reset_index(drop=True); va=val.iloc[va_idx].reset_index(drop=True)
      cand, wi=score_candidates(tr,va)
      if methods is None:
        methods=list(cand.keys()); oof={m:np.full(len(val),np.nan) for m in methods}
      for m,s in cand.items():
        oof[m][va_idx]=s
        row=metric_dict(m,va.label.to_numpy(),s); row['fold']=fold
        fold_rows.append(row)
      fold_weight_rows.append({'fold':fold,'train_rows':len(tr),'heldout_rows':len(va),
        'train_positives':int(tr.label.sum()),'heldout_positives':int(va.label.sum()),
        'iqr_weight_financial':float(wi[0]),'iqr_weight_mda':float(wi[1]),'iqr_weight_auditor':float(wi[2]),
        'train_company_count':int(tr.company_id.nunique()) if 'company_id' in tr else None,
        'heldout_company_count':int(va.company_id.nunique()) if 'company_id' in va else None})
    if any(np.isnan(v).any() for v in oof.values()): raise RuntimeError('OOF predictions missing for some rows.')
    oof_df=val.copy()
    for m,s in oof.items(): oof_df['oof__'+m]=s
    oof_df.to_csv(OUT/'validation_oof_predictions.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'validation_fold_metrics.csv',index=False)
    pd.DataFrame(fold_weight_rows).to_csv(OUT/'fold_fit_diagnostics.csv',index=False)

    # Pooled OOF metrics are primary; mean fold AP is reported as secondary due
    # to only 3-4 positives per fold and high variance in fold-wise AP.
    pooled=[]
    for m,s in oof.items():
      d=metric_dict(m,y,s); d['evaluation']='pooled_out_of_fold_validation'; pooled.append(d)
    pooled_df=pd.DataFrame(pooled).sort_values('average_precision',ascending=False)
    fold_df=pd.DataFrame(fold_rows)
    fold_summary=fold_df.groupby('method').agg(fold_mean_ap=('average_precision','mean'),fold_sd_ap=('average_precision','std'),
      fold_mean_auc=('roc_auc','mean'),fold_sd_auc=('roc_auc','std')).reset_index()
    summary=pooled_df.merge(fold_summary,on='method',how='left').sort_values('average_precision',ascending=False)
    summary.to_csv(OUT/'validation_oof_summary.csv',index=False)

    # Paired bootstrap uncertainty on pooled OOF predictions, stratified by label.
    # This is descriptive uncertainty on validation, not an independent test.
    rng=np.random.default_rng(SEED); B=500
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    base='raw_max_original'
    boot_rows=[]
    for m in methods:
      if m==base or m not in {'raw_mean','raw_power_mean_p4','rank_equal_mean','rank_iqr_weighted','rank_financial_weight_01'}: continue
      diffs=[]
      for _ in range(B):
        ix=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
        rng.shuffle(ix)
        diffs.append(average_precision_score(y[ix],oof[m][ix])-average_precision_score(y[ix],oof[base][ix]))
      boot_rows.append({'comparison':f'{m} minus {base}','method':m,'baseline':base,
        'observed_oof_ap_difference':float(average_precision_score(y,oof[m])-average_precision_score(y,oof[base])),
        'bootstrap_replicates':B,'bootstrap_ci_2_5':float(np.quantile(diffs,.025)),
        'bootstrap_ci_97_5':float(np.quantile(diffs,.975)),
        'interpretation':'Exploratory paired bootstrap on validation OOF rows; not independent confirmation; multiple candidates were compared.'})
    pd.DataFrame(boot_rows).to_csv(OUT/'validation_oof_paired_bootstrap.csv',index=False)

    # Pre-specified weights fitted on full validation only for test description.
    # No test result is used to tune or select weights.
    test_rows=[]
    full_candidates,_=score_candidates(val,test)
    for m,s in full_candidates.items():
      d=metric_dict(m,test.label.to_numpy(),s); d['evaluation']='test_exploratory_do_not_select'; test_rows.append(d)
    pd.DataFrame(test_rows).to_csv(OUT/'test_exploratory_all_methods.csv',index=False)

    # Missingness/availability subgroup counts (no new model selection).
    subgroup=[]
    for split_name,part in [('validation',val),('test_exploratory',test)]:
      for flag in ['financial_available','mda_available','auditor_available']:
        if flag in part:
          for value,sub in part.groupby(flag,dropna=False):
            subgroup.append({'split':split_name,'grouping_variable':flag,'group_value':str(value),
              'n':int(len(sub)),'positives':int(sub.label.sum()),'prevalence':float(sub.label.mean()) if len(sub) else np.nan})
    pd.DataFrame(subgroup).to_csv(OUT/'availability_subgroups.csv',index=False)

    audit['primary_metric']='average precision (AP), appropriate to rare positive labels; ROC-AUC and precision/recall at top-k also reported'
    audit['cv_notes']='Pooled OOF AP is primary; per-fold AP is unstable because each fold contains very few positives.'
    audit['candidate_count']=len(methods)
    audit['pooled_oof_top_method_by_ap']=str(pooled_df.iloc[0]['method'])
    audit['pooled_oof_top_ap']=float(pooled_df.iloc[0]['average_precision'])
    audit['pooled_oof_baseline_ap']=float(pooled_df.loc[pooled_df.method==base,'average_precision'].iloc[0])
    audit['pooled_oof_top_ap_difference_vs_baseline']=audit['pooled_oof_top_ap']-audit['pooled_oof_baseline_ap']
    audit['caution']='Top method among a set of exploratory candidates is not proof of novelty/generalization; use as hypothesis generation only.'
    (OUT/'audit_run_metadata.json').write_text(json.dumps(audit,indent=2))

    print('\nPHASE 3 COMPLETE')
    print('Primary result: pooled out-of-fold validation metrics (2018 validation only).')
    print(summary[['method','average_precision','roc_auc','precision_at_20','recall_at_20','fold_mean_ap','fold_sd_ap']].to_string(index=False))
    print('\nCV splitter:',audit['cv_splitter'])
    print('Rows/positives:',len(val),int(y.sum()),'| test rows/positives:',len(test),int(test.label.sum()))
    print('\nCreated files in outputs/:')
    for p in sorted(OUT.iterdir()): print(' -',p.name)
    print('\nIMPORTANT: Do not select a method based on test_exploratory metrics.')

if __name__=='__main__':
    main()
