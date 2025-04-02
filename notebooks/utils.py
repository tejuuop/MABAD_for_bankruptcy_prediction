import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
from rapidfuzz import process, distance


def load_data():
    fins_sparse = pd.read_csv('../data/SEC/sec_fins_sparse.csv')
    labels = fins_sparse['label']
    fins_sparse.drop(columns=['label'], inplace=True)
    fins_sparse.set_index('uid', inplace=True)
    labels.index = fins_sparse.index
    return fins_sparse, labels


def get_dense_submatrix(fins, labels, threshold=0.5):
    """
    Returns a dense submatrix (dropnans) after selecting all columns with at least {threshold*100} % non nan rows  
    """
    cols_density = fins.notna().sum(axis=0)
    dense_cols = cols_density[cols_density > threshold * fins.shape[0]].index
    result = fins.loc[:, dense_cols].dropna()
    print(f"Num of numerical columns: {dense_cols.__len__()}")
    print(result.label.value_counts())
    return result, dense_cols


def get_sliced_perc(fins, labels, cols_perc=0.5, false_rows_perc=1, true_rows_perc=0.9):

    print(
        f"Select only columns that have more than {cols_perc*100}% dense rows on the original dataset")
    print("Then:")
    print(
        f"Select only false rows that have more than {false_rows_perc*100}% dense columns ")
    print(
        f"Select only true rows that have more than {true_rows_perc*100}% dense columns ")

    cols_density = fins.notna().sum(axis=0)
    dense_cols_idx = cols_density[cols_density >=
                                  cols_perc * fins.shape[0]].index
    print()
    print(f"There are {dense_cols_idx.__len__()} columns")

    fins_sliced = fins.loc[:, dense_cols_idx]

    true_rows_density = fins_sliced.loc[labels == True].notna().sum(axis=1)
    false_rows_density = fins_sliced.loc[labels == False].notna().sum(axis=1)

    dense_true_rows_idx = true_rows_density.loc[true_rows_density >=
                                                true_rows_perc * fins_sliced.shape[1]].index
    dense_false_rows_idx = false_rows_density.loc[false_rows_density >=
                                                  false_rows_perc * fins_sliced.shape[1]].index

    fins_sliced = fins_sliced.loc[dense_true_rows_idx.tolist(
    ) + dense_false_rows_idx.tolist()]
    print()
    print(labels[fins_sliced.index].value_counts())
    return fins_sliced.index, fins_sliced.columns


def test_predictors(X, y, num_runs=5, random_state=42):

    f1s, APs, RPs = [], [], []

    cv = StratifiedKFold(random_state=random_state,
                         n_splits=num_runs, shuffle=True)

    for train_inds, test_inds in cv.split(X, y):
        X_train, y_train = X.iloc[train_inds], y.iloc[train_inds]
        X_test,  y_test = X.iloc[test_inds], y.iloc[test_inds]

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=i, stratify=y)

        clf = Pipeline(
            [

                # ('imp', KNNImputer(n_neighbors=3)),
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        random_state=random_state, class_weight="balanced", max_iter=1000
                    ),
                ),
            ]
        )

        clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:, 1]
        y_test_pred = clf.predict(X_test)

        f1s.append(f1_score(y_true=y_test, y_pred=y_test_pred))
        APs.append(average_precision_score(
            y_true=y_test, y_score=y_test_proba))

        y_test_proba_sorted_idx = np.flip(np.argsort(y_test_proba))
        y_test_sorted = y_test[y_test_proba_sorted_idx]

        total_positives = y_test.sum()
        hits = y_test_sorted[:total_positives].sum()
        recall_at_precision = hits/total_positives
        RPs.append(recall_at_precision)

    print("\nBaseline - Logistic Regression:")
    results_df = pd.DataFrame({
        "Mean": [np.mean(f1s), np.mean(APs), np.mean(RPs)],
        "Std": [np.std(f1s), np.std(APs), np.std(RPs)]
    }, index=["F1 Score", "Average Precision", "Recall at Precision"])

    return results_df


def get_string_similarity_dict(sec_cols, finsim_cols, top):

    # from sentence_transformers import SentenceTransformer

    # fin_cols_lower = [sec.lower() for sec in finsim_cols]

    # sec_cols_lower = [sec.lower() for sec in sec_cols]

    # # 1. Load a pretrained Sentence Transformer model
    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # # 2. Calculate embeddings by calling model.encode()
    # embeddings_fin = model.encode(fin_cols_lower)
    # embeddings_sec = model.encode(sec_cols_lower)
    # similarities = model.similarity(embeddings_fin, embeddings_sec)
    # top_per_fin = np.fliplr(np.argsort(similarities, axis=1))[:, top]
    # similarity_dict = {}
    # for fin_index, fin_col in enumerate(finsim_cols):
    #     similarity_dict[fin_col] = [sec_cols[similar_index_sec]
    #                                 for similar_index_sec in top_per_fin[fin_index]]

    similarity_dict = {}
    sec_cols_lower = [sec.lower() for sec in sec_cols]
    for fin_col in finsim_cols:
        matches = process.extract(
            fin_col.lower(), sec_cols_lower, scorer=distance.Levenshtein.distance, limit=top)
        similarity_dict[fin_col] = [match[0]
                                    for match in matches]
    return similarity_dict


def expand_with_finsim(sec, match_percentage=0.05, top=10):
    finsim = pd.read_csv("../data/other/finsim_feats_overlap_ecl_ao.csv")
    finsim.index = finsim['cik'].astype(
        str) + "__" + finsim['period_of_report']
    finsim.drop(columns=['cik', 'period_of_report'], inplace=True)

    sec_cols = list(sec.columns)
    finsim_cols = list(finsim.columns)

    similarity_dict = get_string_similarity_dict(
        sec_cols=sec_cols, finsim_cols=finsim_cols, top=top)

    match_counts = {}
    sec2finsim = {}

    merged = sec.merge(finsim, how='inner', left_index=True, right_index=True)

    for col1 in finsim_cols:
        for col2 in sec_cols:
            match_count = (merged[col1] == merged[col2]
                           ).sum()
            if match_count > match_percentage * merged.shape[0]:
                if col2 in similarity_dict[col1]:
                    match_counts[(col1, col2)] = match_count

                    if col1 in sec2finsim:
                        print("Multiple Match!")
                        print(
                            f"Previous match for column {col1} was {sec2finsim[col1]}, new match is {col2}")

                    sec2finsim[col2] = col1
                else:
                    # sec2finsim[col2] = col1
                    # print(f"Finsim col: {col1} --> Sec col {col2}") #just value match
                    pass

    print(f"Number of column pairs with matching values: {len(match_counts)}")
    for pair, count in match_counts.items():
        print(f"Columns {pair} have {count} matching rows")

    sec_filled = sec.copy()
    valid_indices = sec_filled.index.intersection(finsim.index)

    mapped_cols = {tag: sec2finsim[tag]
                   for tag in sec2finsim if tag in sec_filled.columns}
    for sec_col, finsim_col in mapped_cols.items():
        if finsim_col in finsim.columns and sec_col in sec_filled.columns:
            sec_filled.loc[valid_indices, sec_col] = sec_filled.loc[valid_indices, sec_col].fillna(
                finsim.loc[valid_indices, finsim_col]
            )
    print(f"{sec.isna().sum().sum() - sec_filled.isna().sum().sum()} new values from finsim!")

    return sec2finsim, sec_filled
