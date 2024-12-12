# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import jaccard_score
from sklearn import svm

# Load the dataset
df = pd.read_csv('heart.csv')

# Display basic statistics about the output variable
# print(df["output"].value_counts())

# Optional: Visualize the distribution of cholesterol levels
# df.hist(column="chol", bins=50)
# print(df.columns)

# Prepare the feature set (X) and target variable (y)
X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
         'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
y = df['output'].values.astype(int)

# Optional: Scatter plot to visualize relationships
# ax = df[df["output"]==0][0:50].plot(kind="scatter", x="age", y="chol", color="blue")
# df[df["output"]==1][0:50].plot(kind="scatter", x="age", y="exng", color="yellow")
# plt.show()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Optional: Display the shapes of the training and testing sets
# print("Train set=", X_train.shape, y_train.shape)
# print("Test set=", X_test.shape, y_test.shape)

# Initialize and train the Support Vector Machine (SVM) classifier
clf = svm.SVC(kernel="rbf").fit(X_train, y_train)

# Make predictions on the test set
y_hat = clf.predict(X_test)

# Calculate and print the Jaccard score
print("Jaccard score =", jaccard_score(y_test, y_hat, pos_label=1))

# Optional: Display the accuracy of the training and test sets
# print("Train set Accuracy =", metrics.accuracy_score(y_train, neig.predict(X_train)))
# print("Test set Accuracy =", metrics.accuracy_score(y_test, y_hat))

# Display the confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_hat, labels=[1, 0]))
print(classification_report(y_test, y_hat))
