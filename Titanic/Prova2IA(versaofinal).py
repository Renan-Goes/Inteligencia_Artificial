import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm as cm
from warnings import simplefilter

from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC

import seaborn as sns

titanic_file = 'train.csv'
titanic_data = pd.read_csv(titanic_file)

features = ['Age', 'Sex', 'SibSp', 'Parch', 'Fare']

filtered_titanic_data = titanic_data.replace('male', 1).replace('female', 0)

n_filtered_titanic_data = filtered_titanic_data.dropna(axis=0, how='all', subset=['Age'])

df = pd.DataFrame(filtered_titanic_data, columns=features)
df[features].replace('', np.nan, inplace=True)
df.dropna(subset=features, inplace=True)

for column in df:
    plt.figure()
    df.boxplot([column])


n_filtered_titanic_data.plot.scatter(x='Age', y='Fare')

hist = df.hist(bins=5)

f, ax = plt.subplots(figsize=(5, 5))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, vmin=-1, vmax=1, annot=True)

X = n_filtered_titanic_data[features]
y = n_filtered_titanic_data.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier2 = SVC(kernel='rbf')

svclassifier.fit(X_train, y_train)
svclassifier2.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
y_pred2 = svclassifier2.predict((X_test))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

conf = confusion_matrix(y_test, y_pred)
print('Sensitivity Linear: ' + "%.3f" % (conf[0][0]/(conf[0][0] + conf[0][1])))
print('Specificity Linear: ' + "%.3f" % (conf[1][1]/(conf[1][0] + conf[1][1])))
print('Precision Linear: ' + "%.3f" % (conf[0][0]/(conf[0][0] + conf[1][0])))
print('Accuracy Linear: ' + "%.3f" % ((conf[0][0] + conf[1][1])/(conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1])))
print('F1-Score Linear: ' + "%.3f" % f1_score(y_test, y_pred))
print('Mean absolute error Linear: ' + "%.3f" % mean_absolute_error(y_test, y_pred))

fig = plt.figure()
ax2 = fig.add_subplot(111)
sns.heatmap(conf, annot=True, ax = ax2)
print('\\' + '-'*20 + '/')
conf2 = confusion_matrix(y_test, y_pred2)
f1g = str(f1_score(y_test, y_pred2))
print('Sensitivity RBF: ' + "%.3f" % (conf2[0][0]/(conf2[0][0] + conf2[0][1])))
print('Specificity RBF: ' + "%.3f" % (conf2[1][1]/(conf2[1][0] + conf2[1][1])))
print('Precision RBF: ' + "%.3f" % (conf2[0][0]/(conf2[0][0] + conf2[1][0])))
print('Accuracy RBF: ' + "%.3f" % ((conf2[0][0] + conf2[1][1])/(conf2[0][0] + conf2[0][1] + conf2[1][0] + conf2[1][1])))
print('F1-Score RBF: ' + "%.3f" % f1_score(y_test, y_pred2))
print('Mean absolute error RBF: ' + "%.3f" % mean_absolute_error(y_test, y_pred2))

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)
sns.heatmap(conf2, annot=True, ax = ax3)