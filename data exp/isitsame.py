
import pickle
import numpy as np
import pandas as pd

with open("finbert_mda.pkl", "rb") as f:
    mda = pickle.load(f)

with open("finbert_auditor.pkl", "rb") as f:
    auditor = pickle.load(f)

df = pd.read_csv("numerical.csv", usecols=["uid", "label"])

print("=== IDENTIFIER ALIGNMENT ===")
print("Numerical rows:", len(df))
print("Unique numerical IDs:", df["uid"].nunique())
print("MDA IDs missing from numerical:",
      len(set(mda) - set(df["uid"])))
print("Auditor IDs missing from numerical:",
      len(set(auditor) - set(df["uid"])))
print("Numerical IDs missing from MDA:",
      len(set(df["uid"]) - set(mda)))
print("Numerical IDs missing from auditor:",
      len(set(df["uid"]) - set(auditor)))

print("\n=== LABEL DISTRIBUTION ===")
print(df["label"].value_counts(dropna=False).to_string())
print("Positive rate:", round(df["label"].mean(), 6))

print("\n=== EMBEDDING QUALITY ===")
for name, embeddings in [("MDA", mda), ("Auditor", auditor)]:
    sample = np.stack(list(embeddings.values())[:1000])
    print(
        name,
        "| dimensions:", sample.shape[1],
        "| finite:", bool(np.isfinite(sample).all()),
        "| mean vector norm:", round(
            float(np.linalg.norm(sample, axis=1).mean()), 4
        ),
    )

print("\n=== DUPLICATE IDS ===")
print("Duplicate numerical IDs:", int(df["uid"].duplicated().sum()))
print("\nCheck complete.")
