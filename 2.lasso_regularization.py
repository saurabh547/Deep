# Write a Program to implement regularization to prevent the model from overfitting
#pip install numpy pandas scikit-learn

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Lasso

# Load and prepare the data
df_train = pd.read_csv(r'C:\Users\kaniya12\Desktop\practicals\train.csv')
df_test = pd.read_csv(r'C:\Users\kaniya12\Desktop\practicals\test.csv')
df_train = df_train.dropna()
df_test = df_test.dropna()

x_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)
x_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

# Lasso Regularization
lasso = Lasso()
lasso.fit(x_train, y_train.ravel())
print("Lasso Train RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_train, lasso.predict(x_train))), 5))
print("Lasso Test RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(x_test))), 5))



import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Ridge

# Load and prepare the data
df_train = pd.read_csv(r'C:\Users\kaniya12\Desktop\practicals\train.csv')
df_test = pd.read_csv(r'C:\Users\kaniya12\Desktop\practicals\test.csv')
df_train = df_train.dropna()
df_test = df_test.dropna()

x_train = df_train['x'].values.reshape(-1, 1)
y_train = df_train['y'].values.reshape(-1, 1)
x_test = df_test['x'].values.reshape(-1, 1)
y_test = df_test['y'].values.reshape(-1, 1)

# Ridge Regularization
ridge = Ridge()
ridge.fit(x_train, y_train)
print("Ridge Train RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_train, ridge.predict(x_train))), 5))
print("Ridge Test RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_test, ridge.predict(x_test))), 5))
