import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

lw = 2

fpr = dict()
tpr = dict()
roc_auc = dict()
class_name = []

test_labels_enc = np.array([[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            ])

basic_cnn_preds = np.array([[0, 0, 0.1],
                            [0, 0, 0.2],
                            [0, 0, 0.3],
                            [0, 0, 0.4],
                            [0, 0, 0.5],
                            [0, 0, 0.6],
                            [0, 0, 0.7],
                            [0, 0, 0.8],
                            [0, 0, 0.9],
                            [0, 0, 0.9],
                            [0, 0, 0.9],
                            [0, 0.9, 0],
                            ])

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(test_labels_enc[:, i], basic_cnn_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# %%
colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'aqua', 'darkorange', 'cornflowerblue']
color_counter = 0
for key in range(3):
    plt.plot(list(fpr[key]), list(tpr[key]), color=colors[color_counter], lw=lw,
             label='ROC curve of Class {0} (area = {1:0.2f})'
                   ''.format(key, roc_auc[key]))
    color_counter += 1

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Single Stage Classifiers')
plt.legend(loc="lower right")
plt.show()
