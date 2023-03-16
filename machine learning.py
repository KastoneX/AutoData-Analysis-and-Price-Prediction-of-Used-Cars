# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
data = pd.read_csv('vehicles.csv')

# Data cleaning and preprocessing
data.drop(['url', 'region_url', 'VIN'], axis=1, inplace=True)
data.dropna(inplace=True)
data = data[data['price'] >= 100]
data = data[data['price'] <= 100000]
data = data[data['year'] >= 1950]
data = data[data['odometer'] <= 500000]
data = data[data['odometer'] >= 1000]
data['age'] = 2023 - data['year']
data['brand'] = data['manufacturer']
data['model'] = data['model'].apply(lambda x: x.lower())
data['brand'] = data['brand'].apply(lambda x: x.lower())
le = LabelEncoder()
data['brand'] = le.fit_transform(data['brand'])
data['model'] = le.fit_transform(data['model'])
data.drop(['manufacturer', 'model'], axis=1, inplace=True)

# Data normalization
X = data.drop(['price'], axis=1)
y = data['price']
X = (X - X.mean()) / X.std()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build CNN model
model = keras.Sequential()
model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64)

# Evaluate model
y_pred = model.predict(X_test)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
