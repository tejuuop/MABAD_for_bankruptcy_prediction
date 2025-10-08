import os
import sys
import pandas as pd

# Set a global random seed for reproducibility
RANDOM_SEED = 42


def time_split(data: pd.DataFrame):
    """
    Perform a chronological (time-based) split of the dataset into train, validation, and test sets.

    This function assumes the dataframe includes:
      - a 'uid' column, where the year is embedded as the second element in a string separated by '__'
        e.g., 'COMPANY123__2018__XYZ'
      - a 'label' column with binary labels (1 for positive, 0 for negative)

    The function:
      1. Extracts the year from 'uid'
      2. Sorts all rows chronologically by year
      3. Uses the positive samples to determine time cutoffs:
         - 65% of positives for training
         - next 10% for validation
         - remaining 25% for testing
      4. Applies these temporal cutoffs to the full dataset (including negatives)

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing at least columns ['uid', 'label'].

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : tuple of pd.DataFrames / pd.Series
        Feature and label splits for train, validation, and test sets.
    """

    # Extract year information from UID
    data['year'] = data['uid'].apply(lambda x: x.split('__')[1])
    data['year'] = pd.to_datetime(data['year'], errors='coerce')
    data = data.sort_values('year').reset_index(drop=True)

    # Use only positive samples to define the chronological boundaries
    positives = data[data['label'] == 1].copy()
    n_pos = len(positives)

    # Compute temporal boundaries
    train_end = int(0.65 * n_pos)
    val_end = int(0.75 * n_pos)

    train_max_date = positives.iloc[train_end]['year']
    val_max_date = positives.iloc[val_end]['year']

    # Apply time-based splits to the entire dataset
    train_df = data[data['year'] <= train_max_date]
    val_df = data[(data['year'] > train_max_date) & (data['year'] <= val_max_date)]
    test_df = data[data['year'] > val_max_date]

    # Display statistics for sanity check
    print(f"Train positives: {train_df['label'].sum()}")
    print(f"Val positives: {val_df['label'].sum()}")
    print(f"Test positives: {test_df['label'].sum()}")

    # Index by UID for easier merging downstream
    train_df.set_index('uid', inplace=True)
    val_df.set_index('uid', inplace=True)
    test_df.set_index('uid', inplace=True)

    # Separate features and labels
    y_train, X_train = train_df['label'], train_df.drop(columns=['label', 'year'])
    y_val, X_val = val_df['label'], val_df.drop(columns=['label', 'year'])
    y_test, X_test = test_df['label'], test_df.drop(columns=['label', 'year'])

    # Print shapes
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_splits(path_dir: str):
    """
    Load and split the dataset into train/val/test sets, replicating the setup used in the paper.

    This function expects two CSV files in `path_dir`:
      - 'numerical.csv': containing tabular (financial ratio) data
      - 'mda_auditor.csv': containing text data with columns for 'mda' and 'auditor' opinion

    The time-based splitting is performed separately on each dataset to ensure temporal consistency.

    Parameters
    ----------
    path_dir : str
        Path to the directory containing the input CSV files.

    Returns
    -------
    (X_train, X_train_text), y_train,
    (X_val, X_val_text), y_val,
    (X_test, X_test_text), y_test : tuple
        Each X_* is a tuple of (financial_features, text_features),
        and y_* are Series with binary labels.
    """

    financial = pd.read_csv(os.path.join(path_dir, 'numerical.csv'))
    mda_auditor = pd.read_csv(os.path.join(path_dir, 'mda_auditor.csv'))

    financial_split = time_split(financial)
    text_split = time_split(mda_auditor)

    return (
        (financial_split[0], text_split[0]), financial_split[1],
        (financial_split[2], text_split[2]), financial_split[3],
        (financial_split[4], text_split[4]), financial_split[5]
    )

