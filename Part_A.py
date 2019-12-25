#
# Part_A.py
#
# __author__= Allen Telson, Nicolo Martina
#
# This file is in conjunction with Part A Word Embedding and Neural Networks of CAP 4770  Fall 2019 - Project
# This code is developed in order to provide different Text Classifiers used to rate a review based on positive negative
# or neutral. It tokenizes text converting it the words into frequencies in which they
# appear within documents in the corpus and is fed into different classifiers for predictions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split, validation_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# load csv file into variable
amazonData = pd.read_csv("reduced_amazon_ff_reviews.csv")

# initialize vectorizer
vectorizer = TfidfVectorizer()

# fit data into vectorizer...
X = vectorizer.fit_transform(amazonData['Text'])

# labels
y = amazonData['Rating']

# Classifiers
rf_classifier = RandomForestClassifier(criterion='entropy')
nb_classifier = MultinomialNB()
# knn_classifier = KNeighborsClassifier(n_neighbors=round(np.math.sqrt(len(X))), algorithm='ball_tree')
dt_classifier = DecisionTreeClassifier()
svc_classifier = SVC(kernel='linear')

# Code below was found via link: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
# this code is used to plot the validation curve of the random forest algorithm
param_range = np.arange(1, 250, 2)
train_scores, test_scores = validation_curve(rf_classifier,
                                             X,
                                             y,
                                             param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3,
                                             scoring="accuracy",
                                             n_jobs=-1)
# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
#
# # Fit the data into different classifiers
# knn_classifier.fit(X_train, y_train)
# # rf_classifier.fit(X_train, y_train)
# # nb_classifier.fit(X_train, y_train)
# # dt_classifier.fit(X_train, y_train)
# # svc_classifier.fit(X_train, y_train)
#
# y_predict = knn_classifier.predict(X_test)
# # y_predict = rf_classifier.predict(X_test)
# # y_predict = nb_classifier.predict(X_test)
# # y_predict = dt_classifier.predict(X_test)
# # y_predict = svc_classifier.predict(X_test)
#
# confusion = confusion_matrix(y_test, y_predict)
# report = classification_report(y_test, y_predict)
#
# print(confusion)
# print(report)

# # CROSS VALIDATION
# y_predict = cross_val_predict(rf_classifier, X, y, cv=10)
# confusion = confusion_matrix(y, y_predict)
# report = classification_report(y, y_predict)

# Print results
# print(confusion)
# print(report)
