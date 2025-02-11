import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 假设y_true是真实的标签，y_pred是预测的标签
y_true = [1, 2, 1, 2, 2, 1, 1, 1, 3, 2, 2, 1, 0, 2, 2, 2]
y_pred = [2, 0, 0, 3, 3, 2, 2, 0, 0, 3, 0, 2, 1, 0, 3, 3]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 设置混淆矩阵的类别名称
class_names = ['0', '1', '2', '3']

# 使用seaborn绘制混淆矩阵
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('predict')
plt.ylabel('label')
plt.title('conffusion_matrix')
plt.show()