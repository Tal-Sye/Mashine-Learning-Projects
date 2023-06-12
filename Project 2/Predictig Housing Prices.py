import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from google.colab import drive
drive.mount('/content/drive')
path = '/content/boston.csv'
data = pd.read_csv(path)
boston_X, boston_y = datasets.load_boston(return_X_y=True)
boston_X = boston_X[:, np.newaxis, 2]
boston_X_train = boston_X[:-20]
boston_X_test = boston_X[-20:]
boston_y_train = boston_y[:-20]
boston_y_test = boston_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(boston_X_train, boston_y_train)
boston_y_pred = regr.predict(boston_X_test)
plt.scatter(boston_X_test, boston_y_test)
plt.plot(boston_X_test, boston_y_pred)
plt.xticks(())
plt.yticks(())
plt.show()

import statsmodels.api as sm
regr = linear_model.LinearRegression()
regr.fit(boston_X, boston_y)
sorted_features = data.drop(['CHAS'], 1)
sorted_result = data['CHAS']
X_train, X_test, y_train, y_test = train_test_split(sorted_features, sorted_result,
model = RFE(estimator = LinearRegression(), n_features_to_select = 1)
model.fit(X_train, y_train)
model.score(X_test, y_test)
print(model.score)