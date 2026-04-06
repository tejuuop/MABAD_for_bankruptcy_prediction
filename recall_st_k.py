import matplotlib.pyplot as plt

k_values = [10, 20, 30, 40, 50, 100]
xgboost_recall = [0.16, 0.37, 0.42, 0.53, 0.58, 0.74]
mabad_recall = [0.11, 0.21, 0.26, 0.26, 0.26, 0.63]

plt.plot(k_values, xgboost_recall, marker='o', label='XGBoost', color='blue')
plt.plot(k_values, mabad_recall, marker='s', label='MABAD', color='orange')

plt.xlabel('Top K Samples Checked')
plt.ylabel('Recall (Percentage of Bankruptcies Caught)')
plt.title('Recall@K: How many "Needles" did we find in the Haystack?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('recall_curve.png')
plt.show()