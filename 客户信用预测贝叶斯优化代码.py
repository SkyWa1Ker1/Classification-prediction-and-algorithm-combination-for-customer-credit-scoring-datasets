# 导入所需的库
#scikit-klearn机器学习库
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']   
data = pd.read_csv(url, names=names)


# 指定要导出的Excel文件路径
excel_file_path = "C:/Users/97795/Desktop/贝叶斯优化/贝叶斯优化/客户信用预测.xlsx"


# 数据探索
# 统计类别变量的分布
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=data)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
print(data)
# 绘制直方图和散点图
plt.figure(figsize=(12, 6))

# 绘制A2的直方图
plt.subplot(1, 2, 1)
sns.histplot(data['A2'], bins=20, kde=True, color='blue')
plt.title('Histogram of A2')
plt.xlabel('A2')
plt.ylabel('Frequency')

# 绘制A3与A8的散点图
plt.subplot(1, 2, 2)
sns.scatterplot(x='A3', y='A8', data=data, hue='class')
plt.title('Scatter Plot of A3 and A8')
plt.xlabel('A3')
plt.ylabel('A8')

plt.tight_layout()
plt.show()
# 将DataFrame导出到Excel文件
data.to_excel(excel_file_path, index=False)



# 数据预处理
data.replace('?', np.nan, inplace=True)  # 将'?'替换为NaN
data.dropna(inplace=True)  # 删除含有缺失值的样本
data = pd.get_dummies(data, columns=['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'])  # 将分类变量转换为哑变量
X = data.drop('class', axis=1)
y = data['class']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 初始化和训练贝叶斯分类器
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 进行预测
y_pred = nb_classifier.predict(X_test)
# 计算优化前的评估指标
accuracy_before = accuracy_score(y_test, y_pred)
recall_before = recall_score(y_test, y_pred,average='macro')
precision_before=precision_score(y_test,y_pred,average='macro')
print("优化前的评估指标：")
print("准确率：", accuracy_before)
print("召回率：", recall_before)
print("精确度",precision_before)
# 可视化优化前的性能
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(['Accuracy', 'Precision', 'Recall',], [accuracy_before, precision_before, recall_before], color='blue', alpha=0.5)

plt.title('Performance Metrics Before Optimization')
plt.ylabel('Score')
plt.ylim(0, 1)  # 设置y轴范围为0到1
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# 初始化多个贝叶斯分类器
nb_classifier1 = GaussianNB()
nb_classifier2 = GaussianNB()
nb_classifier3 = GaussianNB()

# 将多个贝叶斯分类器放入元分类器中
rf_base_estimators = [
    ('nb1', nb_classifier1),
    ('nb2', nb_classifier2)
]

# 初始化随机森林作为元分类器
random_forest = RandomForestClassifier(n_estimators=200)
random_forest2 = RandomForestClassifier(n_estimators=200)
random_forest3 = RandomForestClassifier(n_estimators=200)
# 使用随机森林作为元分类器的投票法组合多个贝叶斯分类器
voting_classifier = VotingClassifier(estimators=[
    ('rf', random_forest),('rf2', random_forest2),('rf3', random_forest3)
] + rf_base_estimators, voting='soft')

# 训练集成分类器
voting_classifier.fit(X_train, y_train)

# 进行预测
y_pred_ensemble = voting_classifier.predict(X_test)


# 计算优化后的评估指标
accuracy_after = accuracy_score(y_test, y_pred_ensemble)
recall_after = recall_score(y_test, y_pred_ensemble, average='macro')
precision_after=precision_score(y_test,y_pred_ensemble,average='macro')
print("优化后的评估指标：")
print("准确率：", accuracy_after)
print("召回率：", recall_after)
print("精确度",precision_after)
# 可视化优化后的性能
plt.figure(figsize=(10, 6))
# 绘制柱状图
plt.bar(['Accuracy', 'precision', 'Recall'], [accuracy_after,precision_before, recall_after], color='green', alpha=0.5)
plt.title('Performance Metrics After Optimization')
plt.ylabel('Score')
plt.ylim(0, 1)  # 设置y轴范围为0到1
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_ensemble)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt

# 定义指标名称和值
metrics = ['Accuracy', 'Recall', 'Precision']
before_optimization = [accuracy_before, recall_before, precision_before]
after_optimization = [accuracy_after, recall_after, precision_after]

# 绘制柱状图
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(metrics))
plt.bar(index, before_optimization, bar_width, label='Before Optimization', color='b')
plt.bar([i + bar_width for i in index], after_optimization, bar_width, label='After Optimization', color='r')

# 添加标签和标题
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparison of Metrics Before and After Optimization')
plt.xticks([i + bar_width / 2 for i in index], metrics)
plt.legend()
plt.show()







