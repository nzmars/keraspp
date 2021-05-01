#%%
from tensorflow import keras
import numpy as np

x = np.array([0, 1, 2, 3, 4])
# x = np.arange(10)
# y = x * 2 + 1 + np.random.randn(x.size)
y = x * 2 + 1
n_test = 2

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:n_test], y[:n_test], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())