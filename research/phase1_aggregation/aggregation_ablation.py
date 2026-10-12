#!/usr/bin/env python3
"""Exploratory MABAD modality-distance aggregation ablation.

Uses reconstructed Phase 0 val/test distances. Test results are exploratory because
prior project iterations already examined this test split. No test labels are used
for fitting or hyperparameter selection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs' / 'active_finbert_mabad_valtest_modality_distances.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
MODS = ['d_financial', 'd_mda', 'd_auditor']


def metrics(y, s):
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind='mergesort')
    out = {'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
           'ap': float(average_precision_score(y, s)),
           'auc': float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None}
    for k in (10, 20, 50, 100):
        kk = min(k, len(y)); hits = int(y[order[:kk]].sum())
        out[f'precision_at_{k}'] = hits / kk
        out[f'recall_at_{k}'] = hits / max(1, int(y.sum()))
    return out


def score_methods(frame, mu=None, sd=None, rank_refs=None):
    x = frame[MODS].to_numpy(float)
    z = (x - mu) / sd if mu is not None else None
    ranks = None
    if rank_refs is not None:
        ranks = np.column_stack([np.searchsorted(rank_refs[j], x[:, j], side='right') / (len(rank_refs[j]) + 1.0) for j in range(x.shape[1])])
    dmax = x.max(axis=1)
    sorted_x = np.sort(x, axis=1)
    sorted_z = np.sort(z, axis=1) if z is not None else None
    scores = {
        'raw_max_original': dmax,
        'raw_mean': x.mean(axis=1),
        'raw_median': np.median(x, axis=1),
        'raw_top2_mean': sorted_x[:, 1:].mean(axis=1),
        'raw_power_mean_p2': np.mean(np.maximum(x, 0)**2, axis=1)**0.5,
        'raw_power_mean_p4': np.mean(np.maximum(x, 0)**4, axis=1)**0.25,
    }
    if ranks is not None:
        sr = np.sort(ranks, axis=1)
        scores.update({'rank_max': ranks.max(axis=1), 'rank_mean': ranks.mean(axis=1),
                       'rank_median': np.median(ranks, axis=1), 'rank_top2_mean': sr[:, 1:].mean(axis=1)})
    if z is not None:
        scores.update({
            'z_max': z.max(axis=1),
            'z_mean': z.mean(axis=1),
            'z_median': np.median(z, axis=1),
            'z_top2_mean': sorted_z[:, 1:].mean(axis=1),
        })
        # Stable log-mean-exp; tau controls how close it is to max (tau -> 0).
        for tau in (0.1, 0.25, 0.5, 1.0, 2.0):
            a = z / tau
            mx = a.max(axis=1)
            scores[f'z_logmeanexp_tau_{tau:g}'] = tau * (mx + np.log(np.exp(a - mx[:, None]).mean(axis=1)))
    return scores


def main():
    df = pd.read_csv(INPUT)
    required = {'split', 'label', *MODS, 'uid', 'date'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df.dropna(subset=['split', 'label', *MODS]).copy()
    df['label'] = df['label'].astype(int)
    val = df[df.split == 'val'].copy().reset_index(drop=True)
    test = df[df.split == 'test'].copy().reset_index(drop=True)
    if len(val) != 2908 or int(val.label.sum()) != 19: print('WARNING: validation count differs from expected')
    if len(test) != 6381 or int(test.label.sum()) != 47: print('WARNING: test count differs from expected')

    # Raw aggregation candidates are parameter-free. Z-score aggregation uses only
    # validation moments; no test statistics are used to standardize scores.
    mu = val[MODS].mean().to_numpy()
    sd = val[MODS].std(ddof=0).replace(0, 1).to_numpy() if isinstance(val[MODS].std(ddof=0), pd.Series) else np.std(val[MODS].to_numpy(), axis=0)
    sd = np.where(sd == 0, 1, sd)
    rank_refs = [np.sort(val[c].to_numpy(float)) for c in MODS]
    val_scores = score_methods(val, mu, sd, rank_refs)
    test_scores = score_methods(test, mu, sd, rank_refs)

    # Tune softmax temperature using validation AP only; all test results are still
    # marked exploratory because this test set was previously inspected.
    temp_names = [f'z_logmeanexp_tau_{t:g}' for t in (0.1, 0.25, 0.5, 1.0, 2.0)]
    chosen_temp = max(temp_names, key=lambda name: average_precision_score(val.label, val_scores[name]))
    fixed_names = [n for n in val_scores if not n.startswith('z_logmeanexp_')]
    selected_names = fixed_names + temp_names

    rows=[]
    for name in selected_names:
        for split, part, scores in [('val', val, val_scores), ('test_exploratory', test, test_scores)]:
            rows.append({'model': name, 'split': split, **metrics(part.label, scores[name])})
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'aggregation_metrics.csv', index=False)

    # Validation 5-fold stability for fixed methods. For standardized methods, fit
    # means/stds on each fold's training partition, then score its held-out fold.
    cv_rows=[]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (itr, iva) in enumerate(skf.split(val[MODS], val.label), start=1):
        tr = val.iloc[itr]; va = val.iloc[iva]
        fmu = tr[MODS].mean().to_numpy()
        fsd = np.std(tr[MODS].to_numpy(float), axis=0, ddof=0); fsd=np.where(fsd==0,1,fsd)
        frefs = [np.sort(tr[c].to_numpy(float)) for c in MODS]
        ss = score_methods(va, fmu, fsd, frefs)
        for name in fixed_names + temp_names:
            cv_rows.append({'fold': fold, 'model': name, **metrics(va.label, ss[name])})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT/'validation_5fold_results.csv', index=False)
    summary = cv.groupby('model').agg(ap_mean=('ap','mean'), ap_sd=('ap','std'), auc_mean=('auc','mean'), auc_sd=('auc','std')).reset_index()
    summary.to_csv(OUT/'validation_5fold_summary.csv', index=False)

    # Frozen scores for auditing/reproduction.
    pred = test[['uid','company_id','date','year','label']].copy()
    for name in selected_names: pred[name] = test_scores[name]
    pred.to_csv(OUT/'test_exploratory_scores.csv', index=False)

    val_metric = result[result.split=='val'].set_index('model')
    test_metric = result[result.split=='test_exploratory'].set_index('model')
    baseline_ap = float(test_metric.loc['raw_max_original','ap'])
    rank = test_metric.sort_values('ap', ascending=False)
    summary_json = {
        'status':'EXPLORATORY — not confirmatory; test split was previously inspected in earlier project work',
        'input_file':str(INPUT),
        'dataset': {'validation_rows':len(val),'validation_positives':int(val.label.sum()),'test_rows':len(test),'test_positives':int(test.label.sum()),'test_prevalence':float(test.label.mean())},
        'standardization':'per-modality mean/std fitted on validation only; applied unchanged to test',
        'selected_softmax_temperature_by_validation_ap':chosen_temp,
        'validation_best_fixed_or_candidate':str(val_metric['ap'].idxmax()),
        'test_best_ap_exploratory':str(rank.index[0]),
        'baseline_test_ap':baseline_ap,
        'test_best_ap_value_exploratory':float(rank.iloc[0]['ap']),
        'test_candidates_sorted_by_ap_exploratory':[{ 'model':str(n), 'ap':float(r['ap']), 'auc':float(r['auc']), 'precision_at_20':float(r['precision_at_20']), 'recall_at_100':float(r['recall_at_100']) } for n,r in rank.iterrows()],
        'warning':'Do not report the best test candidate as a confirmed winner. Use validation for selection and seek an untouched later holdout or an independent dataset for confirmation.'
    }
    (OUT/'aggregation_summary.json').write_text(json.dumps(summary_json, indent=2))

    report = ['# Phase 1 — Direct Aggregation-Function Ablation','',
    '## Status and purpose','',
    '**Exploratory only.** This phase tests whether alternative mathematical aggregations of the three already-reconstructed MABAD modality distances rank bankruptcy filings better than the original maximum. The current test split was examined in earlier work, so these test results are not a pristine confirmatory evaluation.','',
    '## Data and protocol','',
    f'- Validation: {len(val):,} filings, {int(val.label.sum())} positives ({val.label.mean():.3%}).',
    f'- Test: {len(test):,} filings, {int(test.label.sum())} positives ({test.label.mean():.3%}).',
    '- Original baseline: raw `max(d_financial, d_mda, d_auditor)`, checked against saved `d_max`.',
    '- Raw aggregations require no fitted parameters.',
    '- Standardized aggregations use each modality mean and standard deviation fitted on validation only; the same transform is applied to test.',
    f'- Softmax/log-mean-exp temperature selected by validation AP only: `{chosen_temp}`.',
    '- Five-fold stratified validation CV is included as a stability check; with only 19 validation positives, fold-level AP is highly variable.',
    '- No test labels are used for fitting or selecting temperature, but prior exploration of this test split means test comparisons remain exploratory.','',
    '## Candidates','',
    '- `raw_max_original`: original maximum aggregation.',
    '- `raw_mean`, `raw_median`, `raw_top2_mean`, `raw_power_mean_p2`, `raw_power_mean_p4`.',
    '- `z_max`, `z_mean`, `z_median`, `z_top2_mean`: validation-standardized modality scores.',
    '- `rank_max`, `rank_mean`, `rank_median`, `rank_top2_mean`: modality scores transformed to empirical percentiles using validation-only reference distributions (fold-training references inside CV).',
    '- `z_logmeanexp_tau_*`: validation-standardized log-mean-exp with a pre-declared temperature grid.','',
    '## Results','',
    '| Candidate | Validation AP | Test AP (exploratory) | Test AUC | P@20 | R@100 |',
    '|---|---:|---:|---:|---:|---:|']
    for name in selected_names:
        vr=val_metric.loc[name]; tr=test_metric.loc[name]
        report.append(f"| `{name}` | {vr['ap']:.4f} | {tr['ap']:.4f} | {tr['auc']:.4f} | {tr['precision_at_20']:.1%} | {tr['recall_at_100']:.1%} |")
    report += ['', '## Validation cross-validation stability','', '| Candidate | Mean AP | AP SD | Mean AUC | AUC SD |','|---|---:|---:|---:|---:|']
    for _,r in summary.sort_values('ap_mean', ascending=False).iterrows():
        report.append(f"| `{r['model']}` | {r['ap_mean']:.4f} | {r['ap_sd']:.4f} | {r['auc_mean']:.4f} | {r['auc_sd']:.4f} |")
    report += ['', '## Interpretation cautions','',
    '1. Financial distance has a very different distribution from the text distances and is close to constant for most filings. Raw mean/median/power means may therefore be dominated by scale rather than meaningful modality evidence.',
    '2. Standardization and empirical-percentile normalization reduce scale dominance but change the meaning of the original max and may be sensitive to the validation distribution. They are comparisons, not automatically better methods.',
    '3. AP is the primary ranking metric because bankruptcy prevalence is below 1%; AUC is secondary.',
    '4. The validation split has only 19 positives and test has 47. Apparent metric gains can be unstable.',
    '5. These experiments cannot establish novelty. A targeted literature review and independent confirmation are still required.', '',
    '## Decision rule','',
    'Do not pick a method based on test AP. Use validation CV and predeclared criteria to shortlist at most one candidate, then seek an untouched later temporal holdout or independent dataset. If that is unavailable, report this phase as exploratory and avoid confirmatory language.', '',
    '## Output files','',
    '- `aggregation_metrics.csv`: validation and exploratory test metrics for all candidates.',
    '- `validation_5fold_results.csv`: fold-level results.',
    '- `validation_5fold_summary.csv`: mean and standard deviation across folds.',
    '- `test_exploratory_scores.csv`: frozen per-filing candidate scores.',
    '- `aggregation_summary.json`: machine-readable protocol and summary.',
    '- `aggregation_ablation.py`: reproducible script.','']
    (OUT/'aggregation_report.md').write_text('\n'.join(report))
    print('VALIDATION')
    print(result[result.split=='val'][['model','ap','auc','precision_at_20','recall_at_100']].sort_values('ap',ascending=False).to_string(index=False))
    print('\nTEST EXPLORATORY')
    print(result[result.split=='test_exploratory'][['model','ap','auc','precision_at_20','recall_at_100']].sort_values('ap',ascending=False).to_string(index=False))
    print('\nCV')
    print(summary.sort_values('ap_mean',ascending=False).to_string(index=False))
    print('\nchosen temp',chosen_temp)

if __name__=='__main__': main()
