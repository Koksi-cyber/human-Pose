import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
import os
import cv2
import imghdr
import glob

# [previous code for loading images and preprocessing]

# Extract features using the CNN model
get_feature_layer_output = tf.keras.backend.function([model.layers[0].input], [model.layers[6].output])
features_train = get_feature_layer_output([X_train])[0]
features_test = get_feature_layer_output([X_test])[0]

# Feature selection using mutual information
mi_selector = SelectKBest(mutual_info_classif, k=100)
features_train_mi = mi_selector.fit_transform(features_train, y_train)
features_test_mi = mi_selector.transform(features_test)

# Feature selection using PCA
pca = PCA(n_components=64)
features_train_pca = pca.fit_transform(features_train)
features_test_pca = pca.transform(features_test)

# Function to create an ANN model
def create_ann_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Model selection with GridSearchCV
model_params = {
    'feature_selector__k': [50, 100],
    'classifier__input_shape': [(50,), (100,)]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selector', SelectKBest(mutual_info_classif)),
    ('classifier', create_ann_model())
])

grid = GridSearchCV(pipeline, model_params, cv=5, n_jobs=-1)
grid.fit(features_train_mi, y_train)
print("Best parameters found: ", grid.best_params_)

# Train the ANN model with the best parameters
best_k = grid.best_params_['feature_selector__k']
best_input_shape = grid.best_params_['classifier__input_shape']

# Select the top k features based on the best parameters found
mi_selector = SelectKBest(mutual_info_classif, k=best_k)
features_train_mi = mi_selector.fit_transform(features_train, y_train)
features_test_mi = mi_selector.transform(features_test)

# Train the ANN model with the selected features
ann_model = create_ann_model(best_input_shape)
ann_model.fit(features_train_mi, y_train, epochs=50, batch_size=32, validation_data=(features_test_mi, y_test))

# Evaluate the performance of the trained ANN model on the test dataset
loss, accuracy = ann_model.evaluate(features_test_mi, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)