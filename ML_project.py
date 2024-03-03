# -*- coding: utf-8 -*-
"""ML_ASS3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17qp9W3EWvewheT8z9XjDdzSX8uk9PCEd
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('/content/mnist_train.csv')

num_classes = train_data['label'].nunique()
print(f"\nNumber of Unique Classes (Digits): {num_classes}")
print(f"---------------------------------------------------------------------------")

num_features = train_data.shape[1] - 1
print(f"Number of Features (Pixels): {num_features}")
print(f"---------------------------------------------------------------------------")

missing_values = train_data.isnull().sum()
print(f"\nMissing Values: {missing_values}")
print(f"---------------------------------------------------------------------------")
TotalMissingValues = missing_values.sum()
print(f"Total Missing Values in the Training Dataset: {TotalMissingValues}")

train_data.iloc[:, 1:] = train_data.iloc[:, 1:] / 255.0

image_size = (28, 28)
train_images = train_data.iloc[:, 1:].values.reshape(-1, *image_size)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_data['label'][i]}")
    plt.axis('off')
plt.show()


X_train, X_val, y_train, y_val = train_test_split(
    train_data.iloc[:, 1:], train_data['label'], test_size=0.2, random_state=0
)

# KNN
knn_classifier = KNeighborsClassifier()

param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

grid_search = GridSearchCV(knn_classifier, param_grid, cv=3, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters (K-NN):", grid_search.best_params_)

best_knn = grid_search.best_estimator_
y_val_pred_knn = best_knn.predict(X_val)
val_accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
print(f"Validation Accuracy with Best Hyperparameters (K-NN): {val_accuracy_knn}")
print(f"---------------------------------------------------------------------------")

#  ANN models
def create_ann_model(neurons=128, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Experiment 1
hidden_neurons_list_exp1 = [64, 128]
val_accuracies_exp1 = []

model_exp1 = create_ann_model(neurons=128, learning_rate=0.001)

for neurons in hidden_neurons_list_exp1:
    batch_size = 32

    X_train_flatten = X_train.values.reshape(-1, 28, 28)
    X_val_flatten = X_val.values.reshape(-1, 28, 28)

    model_exp1.fit(
        X_train_flatten, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=(X_val_flatten, y_val),
        verbose=0
    )

    val_accuracy_exp1 = model_exp1.evaluate(X_val_flatten, y_val, verbose=0)[1]
    val_accuracies_exp1.append(val_accuracy_exp1)

for neurons, accuracy_exp1 in zip(hidden_neurons_list_exp1, val_accuracies_exp1):
    print(f"Validation Accuracy (Hidden Neurons={neurons}) - Experiment 1: {accuracy_exp1}")
    print(f"---------------------------------------------------------------------------")

best_neurons_exp1 = hidden_neurons_list_exp1[np.argmax(val_accuracies_exp1)]
best_model_exp1 = create_ann_model(neurons=best_neurons_exp1, learning_rate=0.001)

best_model_exp1.fit(
    X_train_flatten, y_train,
    epochs=30,
    batch_size=batch_size,
    validation_data=(X_val_flatten, y_val),
    verbose=0
)

print(f"Best Hidden Neurons (Experiment 1): {best_neurons_exp1}")
print(f"Validation Accuracy (Best Model - Experiment 1): {max(val_accuracies_exp1)}")
print(f"---------------------------------------------------------------------------")

# Experiment 2
learning_rates_exp2_model2 = [0.001, 0.01]
val_accuracies_exp2_model2 = []

model_exp2_model2 = create_ann_model(neurons=256, learning_rate=0.001)

for learning_rate in learning_rates_exp2_model2:

    model_exp2_model2.fit(
        X_train_flatten, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=(X_val_flatten, y_val),
        verbose=0
    )


    val_accuracy_exp2_model2 = model_exp2_model2.evaluate(X_val_flatten, y_val, verbose=0)[1]
    val_accuracies_exp2_model2.append(val_accuracy_exp2_model2)
    print(f"Validation Accuracy (Learning Rate={learning_rate}) - Experiment 2 (Model 2): {val_accuracy_exp2_model2}")
    print(f"---------------------------------------------------------------------------")

best_learning_rate_exp2_model2 = learning_rates_exp2_model2[np.argmax(val_accuracies_exp2_model2)]
best_model_exp2_model2 = create_ann_model(neurons=256, learning_rate=best_learning_rate_exp2_model2)


best_model_exp2_model2.fit(
    X_train_flatten, y_train,
    epochs=30,
    batch_size=batch_size,
    validation_data=(X_val_flatten, y_val),
    verbose=0
)

print(f"Best Learning Rate (Experiment 2 - Model 2): {best_learning_rate_exp2_model2}")
print(f"Validation Accuracy (Best Model - Experiment 2 - Model 2): {max(val_accuracies_exp2_model2)}")
print(f"---------------------------------------------------------------------------")
print(f"---------------------------------------------------------------------------")
print(f"Validation Accuracy (K-NN): {val_accuracy_knn}")
print(f"Validation Accuracy (Best Model - Experiment 1): {max(val_accuracies_exp1)}")
print(f"Validation Accuracy (Best Model - Experiment 2 - Model 2): {max(val_accuracies_exp2_model2)}")

# Compare validation accuracies and identify the best approach
best_validation_approach = ""

max_val_accuracy = max(val_accuracy_knn, max(val_accuracies_exp1), max(val_accuracies_exp2_model2))

if max_val_accuracy == val_accuracy_knn:
    best_validation_approach = "K-NN"
    best_model = knn_classifier  # Adjust this line based on how you saved your K-NN model
elif max_val_accuracy == max(val_accuracies_exp1):
    best_validation_approach = "Best Model - Experiment 1"
    best_model = best_model_exp1
else:
    best_validation_approach = "Best Model - Experiment 2 - Model 2"
    best_model = best_model_exp2_model2
print(f"---------------------------------------------------------------------------")
print(f"Best Validation Approach: {best_validation_approach}")
print(f"---------------------------------------------------------------------------")

# Get predictions on the validation set using the best model
y_val_pred_prob_best_model = best_model.predict(X_val_flatten)
y_val_pred_best_model = np.argmax(y_val_pred_prob_best_model, axis=1)

# Calculate and print the confusion matrix
conf_matrix_best_model = confusion_matrix(y_val, y_val_pred_best_model)
print("Confusion Matrix (Best Model):")
print(conf_matrix_best_model)
print(f"---------------------------------------------------------------------------")
# Save the best model
best_model.save('best_model.h5')

# Reload the best model from the file
loaded_best_model = models.load_model('best_model.h5')
#  testing data
test_data = pd.read_csv('/content/mnist_test.csv')

num_classes_test = test_data['label'].nunique()
print(f"\nNumber of Unique Classes (Digits): {num_classes_test}")
print(f"---------------------------------------------------------------------------")

num_features_test = test_data.shape[1] - 1
print(f"Number of Features (Pixels): {num_features_test}")
print(f"---------------------------------------------------------------------------")

missing_values_test = test_data.isnull().sum()
print(f"\nMissing Values: {missing_values_test}")
print(f"---------------------------------------------------------------------------")

test_data.iloc[:, 1:] = test_data.iloc[:, 1:] / 255.0
test_images = test_data.iloc[:, 1:].values.reshape(-1, *image_size)

X_test_flatten = test_images.reshape(-1, 28, 28)

y_test_pred_prob_loaded_model = loaded_best_model.predict(X_test_flatten)
y_test_pred_loaded_model = np.argmax(y_test_pred_prob_loaded_model, axis=1)

test_accuracy = accuracy_score(test_data['label'], y_test_pred_loaded_model)

print(f"---------------------------------------------------------------------------")
print(f"Testing Accuracy (Best Model ): {test_accuracy}")

conf_matrix_test = confusion_matrix(test_data['label'], y_test_pred_loaded_model)
print("\nConfusion Matrix (Testing Data - Best Model ):")
print(conf_matrix_test)