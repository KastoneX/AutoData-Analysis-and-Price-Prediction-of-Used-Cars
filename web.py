# Import necessary libraries
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data and preprocess
data = pd.read_csv('vehicles.csv')
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

# Reshape data for CNN
X = X.values.reshape((X.shape[0], X.shape[1], 1))

# Load trained model
model = keras.models.load_model('car_price_prediction_model.h5')

# Define Flask app
app = Flask(__name__)

# Define Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    # Convert data to numpy array
    input_data = np.array(data['data']).reshape((1, data['data'].shape[0], 1))
    # Make prediction
    prediction = model.predict(input_data)[0][0]
    # Return prediction
    return jsonify({'prediction': prediction})

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
