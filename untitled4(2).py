# -*- coding: utf-8 -*-


import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt




df = pd.read_csv('Heart_Disease_Prediction.csv')

df.head()

df.info()

sns.countplot(x=df['Heart Disease'],hue='Sex',data=df)

sns.barplot(x=df['Heart Disease'],y=df['Exercise angina'],data=df)

sns.barplot(x=df['Sex'],y=df['Exercise angina'],hue='Sex',data=df)

sns.barplot(x=df['Sex'],y=df['Cholesterol'],hue='Sex',data=df)

sns.barplot(x=df['Heart Disease'],y=df['Cholesterol'],hue='Heart Disease',data=df)

sns.lineplot(x=df['Age'],y=df['Cholesterol'],data=df)

sns.lineplot(x=df['Age'],y=df['ST depression'],data=df)

sns.barplot(x=df['Heart Disease'],y=df['ST depression'],data=df)

sns.heatmap(df.corr())

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Heart Disease'])
y = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")

df.head()

#data baru
import numpy as np

new_data = np.array([[32, 1, 2, 115, 260, 1, 0, 170, 1, 1.6, 1, 0, 3]])  # Presence/absence tidak dimasukkan (target)
new_data_scaled = scaler.transform(new_data)

# Prediksi dengan menggunakan madel yang telah dilatih
predictions = model.predict(new_data_scaled)

# konversi ke yes atau no dengan  threshold (0.5)
binary_predictions = (predictions > 0.5).astype(int)

print(f"Predicted Probability: {predictions[0][0]}")
print(f"Binary Prediction: {binary_predictions[0][0]}")
print()

if (binary_predictions==0):
    print("Absence...")
else:
    print("Presence")

print()
