
import pickle
import numpy as np
import pandas as pd

def describe_value(value, name="value"):
    print(f"  {name} type: {type(value).__name__}")

    if isinstance(value, np.ndarray):
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)
        if value.size and np.issubdtype(value.dtype, np.number):
            print("  finite values:", int(np.isfinite(value).sum()))
    elif isinstance(value, dict):
        print("  nested keys:", list(value.keys())[:8])
    elif isinstance(value, (list, tuple)):
        print("  length:", len(value))
        if value:
            first = value[0]
            print("  first element type:", type(first).__name__)
            if isinstance(first, (int, float, np.number)):
                print("  first 8 values:", value[:8])
    elif hasattr(value, "shape"):
        print("  shape:", value.shape)
        if hasattr(value, "dtype"):
            print("  dtype:", value.dtype)

with open("finbert_mda.pkl", "rb") as f:
    mda = pickle.load(f)

with open("finbert_auditor.pkl", "rb") as f:
    auditor = pickle.load(f)

print("=== DICTIONARY KEY CHECK ===")
print("MDA keys:", len(mda))
print("Auditor keys:", len(auditor))
print("Same key sets:", set(mda.keys()) == set(auditor.keys()))
print("Common keys:", len(set(mda.keys()) & set(auditor.keys())))
print("MDA-only keys:", len(set(mda.keys()) - set(auditor.keys())))
print("Auditor-only keys:", len(set(auditor.keys()) - set(mda.keys())))

print("\n=== SAMPLE MDA VALUE ===")
mda_key = next(iter(mda))
print("Key:", repr(mda_key))
describe_value(mda[mda_key])

print("\n=== SAMPLE AUDITOR VALUE ===")
auditor_key = next(iter(auditor))
print("Key:", repr(auditor_key))
describe_value(auditor[auditor_key])

print("\n=== NUMERICAL CSV SAMPLE ===")
df = pd.read_csv("numerical.csv", nrows=5)
print("Has uid:", "uid" in df.columns)
print("Has label:", "label" in df.columns)
if "uid" in df.columns:
    print("Sample uids:", df["uid"].tolist())
if "label" in df.columns:
    print("Sample labels:", df["label"].tolist())

print("\n=== TEXT CSV SAMPLE ===")
text = pd.read_csv("mda_auditor.csv", nrows=3)
print("Columns:", text.columns.tolist())
print("Sample uids:", text["uid"].tolist())
print("Sample labels:", text["label"].tolist())
print("MDA text lengths:",
      text["item_7"].fillna("").str.len().tolist())
print("Auditor text lengths:",
      text["opinion_text"].fillna("").str.len().tolist())

print("\nInspection complete.")

