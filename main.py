import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import numpy as np
# Load the "loan_old.csv" dataset.
loan_old = pd.read_csv("loan_old.csv")
#check whether there are missing values
missing_values = loan_old.isnull().sum()
print("Missing Values:")
print(missing_values)
features=loan_old.iloc[:,1:-2]
# #check the type of each feature and Target(categorical or numerical)
# print("\n) Data Types:")
# print(loan_old.dtypes)
#check the type of each feature (categorical or numerical)
print("\n) Data Types for features only:")
print(features.dtypes)
# #check whether numerical features and Targets have the same scale
# print("\n) Numerical Features and targets  Scale Check:")
# print(loan_old.describe())
#check whether numerical features have the same scale
print("\n) Numerical Features Scale Check:")
print(features.describe())
#visualize a pairplot between numercial columns
numerical_features = ['Income', 'Coapplicant_Income', 'Loan_Tenor']
sns.pairplot(loan_old[numerical_features])
plt.show()
#records containing missing values are removed
loan_old_cleaned = loan_old.dropna()

#the features and targets are separated
x = loan_old_cleaned.iloc[:, 1:-2]
target_loan_amount = loan_old_cleaned['Max_Loan_Amount']
target_loan_status = loan_old_cleaned['Loan_Status']

#the data is shuffled and split into training and testing sets
x_train, x_test, target_loan_amount_train, target_loan_amount_test, target_loan_status_train, target_loan_status_test = train_test_split(
    x, target_loan_amount, target_loan_status, test_size=0.1, random_state=0)
# categorical features are encoded
label_encoder = LabelEncoder()
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Property_Area','Credit_History']:
    x_train[col] = label_encoder.fit_transform(x_train[col])
    x_test[col] = label_encoder.transform(x_test[col])

#categorical targets are encoded
target_encoder = LabelEncoder()
target_loan_status_train = target_encoder.fit_transform(target_loan_status_train)
target_loan_status_test = target_encoder.transform(target_loan_status_test)

# numerical features are standardized
scaler = StandardScaler()
x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test[numerical_features] = scaler.transform(x_test[numerical_features])

# Display final processed data
print("\nProcessed Data:")
#Features
print(x_train.head())
#Targets
print(pd.DataFrame(target_loan_amount_train, columns=['Max_Loan_Amount']).head())
print(pd.DataFrame(target_loan_status_train, columns=['Loan_Status']).head())

#the linear regression model
linear_reg_model = LinearRegression()

# Fit a linear regression model to the data to predict the loan amount(test).-> Use sklearn's linear regression.
linear_reg_model.fit(x_train, target_loan_amount_train)

predictions = linear_reg_model.predict(x_test)

plt.scatter(target_loan_amount_test, predictions)
plt.xlabel("Actual Loan Amount")
plt.ylabel("Predicted Loan Amount")
plt.title("Linear Regression: Actual vs Predicted Loan Amount")
plt.show()

#Evaluate the linear regression model using sklearn's R2 score.
mae = mean_absolute_error(target_loan_amount_test, predictions)
mse = mean_squared_error(target_loan_amount_test, predictions)
r2 = r2_score(target_loan_amount_test, predictions)
# print("\nModel Evaluation Metrics:")
# print(f"Mean Absolute Error: {mae}")
#print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#class logisticRegression
class LogisticRegression:

    def _init_(self, lr=0.01, n_iters=3000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

#Fit a logistic regression model to the data to predict the loan status
# Logistic Regression
logistic_model = LogisticRegression(lr=0.01, n_iters=3000)

# Separate features and targets
X_train_logistic_np = x_train.values
y_train_logistic_np = target_loan_status_train

# Fit Logistic Regression model
logistic_model.fit(X_train_logistic_np, y_train_logistic_np)

#Write a function (from scratch) to calculate the accuracy of the model
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

X_test_logistic_np = x_test.values
y_test_logistic_np = target_loan_status_test

logistic_predictions = logistic_model.predict(X_test_logistic_np)

accuracy_logistic = calculate_accuracy(y_test_logistic_np, logistic_predictions)

print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_logistic}")

#Load the "loan_new.csv" dataset.
loan_new = pd.read_csv("loan_new.csv")

#Perform the same preprocessing on it (except shuffling and splitting).
#records containing missing values are removed
loan_new_cleaned = loan_new.dropna()

#the features and targets are separated
x_new = loan_new_cleaned.iloc[:, 1:]

#categorical features are encoded
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Property_Area','Credit_History']:
    x_new[col] = label_encoder.fit_transform(x_new[col])

# vi)numerical features are standardized
x_new[numerical_features] = scaler.fit_transform(x_new[numerical_features])

# Display final processed data for the new dataset
print("\nProcessed Data for loan_new:")
print(x_new.head())

#Use your models on this data to predict the loan amounts and status.
# Predict loan amounts using the linear regression model
loan_amount_predictions = linear_reg_model.predict(x_new)

# Predict loan status using the logistic regression model
logistic_predictions_new = logistic_model.predict(x_new.values)

# Create a new DataFrame with predictions
predictions_df = pd.DataFrame({
    'LinearReg_Predicted_Max_Loan_Amount': loan_amount_predictions,
    'LogisticReg_Predicted_Loan_Status': logistic_predictions_new
})

# Display the DataFrame with predictions
print("\nPredictions for the new data:")
print(predictions_df)