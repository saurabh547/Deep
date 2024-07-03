import numpy as np
import pandas as pd

np.random.seed(0)

x_train = np.random.rand(100, 1) * 10
y_train = 2 * x_train + np.random.randn(100, 1) * 2

np.random.seed(1)
x_test = np.random.rand(20, 1) * 10
y_test = 2 * x_test + np.random.randn(20, 1) * 2

df_train = pd.DataFrame({'x': x_train.flatten(), 'y': y_train.flatten()})
df_test = pd.DataFrame({'x': x_test.flatten(), 'y': y_test.flatten()})

df_train.to_csv(r'C:\Users\kaniya12\Desktop\train.csv', index=False)
df_test.to_csv(r'C:\Users\kaniya12\Desktop\test.csv', index=False)
