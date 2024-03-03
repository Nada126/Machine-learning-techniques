import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('diabetes.csv')


# Function to normalize the data using Min-Max Scaling
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())


# Function to split data into training and testing sets
def train_test_split(data, split_ratio=0.7):
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data



def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn_predict(train_data, test_instance, k):
    distances = []
    for index, row in train_data.iterrows():
        distance = euclidean_distance(row[:-1], test_instance)
        distances.append((index, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    neighbor_labels = train_data.iloc[[i[0] for i in neighbors]]['Outcome']


    unique_labels = neighbor_labels.unique()
    if len(unique_labels) == 1:
        return unique_labels[0]
    else:
        weights = 1 / np.array([distance for _, distance in neighbors])
        label_weights = {label: sum(weights[neighbor_labels == label]) for label in unique_labels}
        return max(label_weights, key=label_weights.get)


def evaluate_knn(train_data, test_data, k):
    correct_predictions = 0
    for index, test_instance in test_data.iterrows():
        predicted_label = knn_predict(train_data, test_instance[:-1], k)
        if predicted_label == test_instance['Outcome']:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy, correct_predictions, len(test_data)


# Normalize each feature
for column in data.columns[:-1]:
    data[column] = min_max_scaling(data[column])

 
train_data, test_data = train_test_split(data, split_ratio=0.7)


total_correct_instances = 0
total_instances = len(test_data)
total_accuracy = 0

for k_value in range(2, 7):
    accuracy, correct_instances, _ = evaluate_knn(train_data, test_data, k_value)
    total_correct_instances += correct_instances

    print(f'k value: {k_value}')
    print(f'Number of correctly classified instances: {correct_instances}')
    print(f'Total number of instances: {total_instances}')
    print(f'Accuracy: {accuracy * 100:.0f}%\n')
    total_accuracy += accuracy


average_accuracy = total_accuracy / 5
print(f'Average Accuracy across all iterations: {average_accuracy * 100:.0f}%')