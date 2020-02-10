import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import sys
from sklearn import neighbors, preprocessing
from sklearn.metrics import r2_score, log_loss, mean_absolute_error, classification_report, confusion_matrix, accuracy_score, f1_score, confusion_matrix
from sklearn import svm

house_file = 'train2.csv'
house_data = pd.read_csv(house_file)

new_data = house_data.fillna(0.0)

num = ['int64', 'float64']
ndata = new_data[[c for c,v in new_data.dtypes.items() if v in num]]
categorical = new_data[[c for c,v in new_data.dtypes.items() if v not in num]]

for col in ndata.columns:
    ndata.fillna(ndata.loc[:, col].mean(), inplace=True, axis=1)

features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
            'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal', 'MoSold', 'YrSold']

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(ndata[features])
y = ndata.SalePrice

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

print("KNN:")
menor_mae_knn = sys.maxsize
for j in [1, 3, 5, 10, 20, 50, 100, 500]:
    knn = neighbors.KNeighborsRegressor(j)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    valor_mae = mean_absolute_error(y_test, y_pred)
    precision = r2_score(y_test, y_pred)

    #Gráficos da predição mostrando a relação entre valores preditos e valores verdadeiros#

    #     df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #     df1 = df.head(25)
    #     df1.plot(kind='bar',figsize=(6,2))
    #     plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    #     plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #     plt.show()

    if (valor_mae < menor_mae_knn):
        maior_p_knn = precision
        maior_k = j
        menor_mae_knn = valor_mae

    print("-"*80)
    print("Mean absolute error for " + "%d neighbours with " % j +
          "%.3f precision was " % precision + "%.3f" % valor_mae)
    print("-"*80)

print("Best mean absolute error for KNN was: " "%.3f " % menor_mae_knn + "with %d" % maior_k +
      " neighbours and %.3f precision." % maior_p_knn)


menor_mae_rf = sys.maxsize
print("Random Forest:")
for i in [10, 100, 1000]:
    for max_leaf_nodes in [10, 50, 100, 500, 1000]:
        regressor = RandomForestRegressor(i, random_state = 0, max_leaf_nodes=max_leaf_nodes)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        #Gráficos da predição mostrando a relação entre valores preditos e valores verdadeiros#

        #         df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        #         df1 = df.head(25)
        #         df1.plot(kind='bar',figsize=(6,2))
        #         plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        #         plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        #         plt.show()

        valor_mae = mean_absolute_error(y_test, y_pred)
        precision = r2_score(y_test, y_pred)


        if (valor_mae < menor_mae_rf):
            maior_p_rf = precision
            maior_v = i
            menor_mae_rf = valor_mae
            maior_l = max_leaf_nodes

        print("-"*80)
        print("Mean absolute error for " + "%d" % i + " trees with " + "%d " % max_leaf_nodes +
              "nodes and %.3f precision " % precision + "is %.3f" % valor_mae)
        print("-"*80)

print("Best mean absolute error for Random Forest was: " "%.3f" % menor_mae_rf + " with %d" % maior_v +
      " trees, " + "%d" % maior_l +" nodes and %.3f precision." % maior_p_rf)