from random import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

drug_data = pd.read_csv("drug.csv")


missing_values = drug_data.isnull().sum()
print("Missing Values:")
print(missing_values)


# Filling numerical columns with mean
drug_data['Age'].fillna(drug_data['Age'].mean(), inplace=True)
drug_data['Na_to_K'].fillna(drug_data['Na_to_K'].mean(), inplace=True)
# Filling categorical columns with the most frequent value
drug_data['Sex'].fillna(drug_data['Sex'].mode()[0], inplace=True)
drug_data['BP'].fillna(drug_data['BP'].mode()[0], inplace=True)
drug_data['Cholesterol'].fillna(drug_data['Cholesterol'].mode()[0], inplace=True)
drug_data['Drug'].fillna(drug_data['Drug'].mode()[0], inplace=True)


features = drug_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
target_drug_type = drug_data['Drug']


le = LabelEncoder()
drug_data['Sex'] = le.fit_transform(drug_data['Sex'])
drug_data['BP'] = le.fit_transform(drug_data['BP'])
drug_data['Cholesterol'] = le.fit_transform(drug_data['Cholesterol'])
drug_data['Drug'] = le.fit_transform(drug_data['Drug'])


print("\nEncoded Data:")
print(drug_data)


drug_data_encoded = pd.get_dummies(drug_data, columns=['Sex', 'BP', 'Cholesterol', 'Drug'], prefix=['Sex', 'BP', 'Cholesterol', 'Drug'])


print("Columns after one-hot encoding:")
print(drug_data_encoded.columns)


features = drug_data_encoded[['Age', 'Sex_0', 'Sex_1', 'BP_0', 'BP_1', 'BP_2', 'Cholesterol_0', 'Cholesterol_1', 'Na_to_K']]
target_drug_type = drug_data_encoded['Drug_0']

model_accuracies = []
model_accuracies_2 = []

for i in range(5):
    random_state = None
    if i > 0:
        random_state = i *10


    x_train, x_test, y_train, y_test = train_test_split(features, target_drug_type, test_size=0.3, random_state=random_state)


    clf = DecisionTreeClassifier()

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    tree_size = clf.tree_.node_count


    print(f"\nExperiment {i + 1} - Train-Test Split Ratio: 70-30, Random State: {random_state}")
    print(f"Training Set Size: {len(x_train)} samples")
    print(f"Testing Set Size: {len(x_test)} samples")
    print(f"Decision Tree Size: {tree_size} nodes")
    print(f"Accuracy: {accuracy:.4f}")


    model_accuracies.append(accuracy)


best_model_index = np.argmax(model_accuracies)
best_model_accuracy = model_accuracies[best_model_index]

print("\nModel Comparison:")
print(f"Best Model - Experiment {best_model_index + 1}")
print(f"Best Model Accuracy: {best_model_accuracy:.4f}")
print("############################################################################")
# Variables to store statistics
mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []


training_set_sizes = range(30, 80, 10)


for train_size in training_set_sizes:

    seed_accuracies = []
    seed_tree_sizes = []

    for seed in range(5):
        x_train, x_test, y_train, y_test = train_test_split(features, target_drug_type, test_size=(100 - train_size) / 100, random_state=seed)

        clf = DecisionTreeClassifier()

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        tree_size = clf.tree_.node_count

        seed_accuracies.append(accuracy)
        seed_tree_sizes.append(tree_size)

    mean_accuracy = np.mean(seed_accuracies)
    max_accuracy = np.max(seed_accuracies)
    min_accuracy = np.min(seed_accuracies)

    mean_tree_size = np.mean(seed_tree_sizes)
    max_tree_size = np.max(seed_tree_sizes)
    min_tree_size = np.min(seed_tree_sizes)

    mean_accuracies.append(mean_accuracy)
    max_accuracies.append(max_accuracy)
    min_accuracies.append(min_accuracy)
    mean_tree_sizes.append(mean_tree_size)
    max_tree_sizes.append(max_tree_size)
    min_tree_sizes.append(min_tree_size)


plt.figure(figsize=(10, 5))

# Plot 1: Accuracy against training set size
plt.subplot(1, 2, 1)
plt.plot(training_set_sizes, mean_accuracies, label='Mean Accuracy')
plt.fill_between(training_set_sizes, min_accuracies, max_accuracies, color='blue', alpha=0.2, label='Min-Max Range')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Training Set Size')
plt.legend()

# Plot 2: Number of nodes in the final tree against training set size
plt.subplot(1, 2, 2)
plt.plot(training_set_sizes, mean_tree_sizes, label='Mean Tree Size')
plt.fill_between(training_set_sizes, min_tree_sizes, max_tree_sizes, color='orange', alpha=0.2, label='Min-Max Range')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Number of Nodes in the Tree')
plt.title('Tree Size vs. Training Set Size')
plt.legend()


plt.tight_layout()
plt.show()


print("\nStatistics Report:")
print(f"{'Training Set Size (%)':<25} {'Mean Accuracy':<15} {'Min Accuracy':<15} {'Max Accuracy':<15} {'Mean Tree Size':<15} {'Min Tree Size':<15} {'Max Tree Size':<15}")
for i in range(len(training_set_sizes)):
    print(f"{training_set_sizes[i]:<25} {mean_accuracies[i]:<15.4f} {min_accuracies[i]:<15.4f} {max_accuracies[i]:<15.4f} {mean_tree_sizes[i]:<15.4f} {min_tree_sizes[i]:<15} {max_tree_sizes[i]:<15}")
