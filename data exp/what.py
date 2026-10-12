
import os
import pickle
import pandas as pd

for filename in [
    "finbert_mda.pkl",
    "finbert_auditor.pkl",
    "numerical.csv",
    "mda_auditor.csv",
]:
    print(f"\n--- {filename} ---")

    if not os.path.exists(filename):
        print("File not found")
        continue

    print("Size (GB):", round(os.path.getsize(filename) / 1e9, 3))

    if filename.endswith(".csv"):
        # Read only a tiny sample; do not print all column names.
        df = pd.read_csv(filename, nrows=2)
        print("Column count:", len(df.columns))
        print("First 12 columns:", df.columns.tolist()[:12])
        print("Sample shape:", df.shape)
        continue

    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        print("Object type:", type(obj).__name__)

        if isinstance(obj, dict):
            print("Key count:", len(obj))
            print("First 10 keys:", list(obj.keys())[:10])
        elif hasattr(obj, "shape"):
            print("Shape:", obj.shape)
        elif isinstance(obj, (list, tuple)):
            print("Length:", len(obj))
            if obj:
                print("First element type:", type(obj[0]).__name__)
        else:
            print("Object loaded; details not printed")
    except Exception as e:
        print("Inspection error:", type(e).__name__, str(e)[:200])
