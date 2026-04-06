import matplotlib.pyplot as plt
import seaborn as sns

models = ['XGBoost (TF-IDF)', 'MABAD (TF-IDF)', 'Target (Paper)']
auc = [0.97, 0.93, 0.94]
ap = [0.24, 0.13, 0.34]

x = range(len(models))
plt.figure(figsize=(10, 6))

plt.bar(x, auc, width=0.4, label='AUC (Ranking Quality)', color='skyblue', align='center')
plt.bar([i + 0.4 for i in x], ap, width=0.4, label='AP (Real Precision)', color='salmon', align='center')

plt.xticks([i + 0.2 for i in x], models)
plt.ylabel('Score')
plt.title('Performance Comparison: Baseline vs. Replication')
plt.legend()
plt.savefig('performance_comparison.png')
plt.show()