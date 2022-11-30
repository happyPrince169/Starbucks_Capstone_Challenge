### Starbucks Capstone Challenge

### Description: Capstone project for Data Scientist Nanodegree in Udacity

This project is my capstone project for Data Scientist Nanodegree in Udacity.
In this project, I will use a machine learning model to predict whether or not someone will complete an offer based on demographics and offer portfolio.

- I build KNN and DecisionTree model as classification models to predict the ability of a customer to complete an offer. I'm not sure which would be the best model for this dataset. I tried and compared the results of the 2 models.
- To validate the results and compare 2 models, I compute Accuracy Score and Classification report. The classification report shows us the accuracy of the models based on the confusion matrix.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

  Link to my blog post with main observations to the code
  https://medium.com/@mia.nguyen.vu/starbucks-capstone-challenge-44f7d610ecd6

# Licensing, Authors, and Acknowledgements

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app.

# File structure of the project

* Starbucks_Capstone_notebook.ipynb # Jupyter Notebook
* data #Contains datasets
* README.md

# Predictive Modeling

- I use KNN and DecisionTree model as classification models to predict the ability of a customer to complete an offer. I'm not sure which would be the best model for this dataset. I tried and compared the results of the 2 models.
- To validate the results and compare 2 models, I compute Accuracy Score and Classification report. The classification report shows us the accuracy of the models based on the confusion matrix.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

# Conclusion

I just tried to build 2 models KNN and DecisionTree to predict whether or not a customer will fully complete an offer. The accuracy score of DecisionTree is nearly 76%, better than KNN, just about 62%.

However, to improve the result, I think I can try more models and there are still many ways to explore this dataset, which may give me more interesting information.

I still confused about how to choose the right model, and what makes a model generate a better results than the others on a dataset. I will continue to investigate and learn.

# Libraries I have imported

pandas as pd

import numpy as np

import math

import json

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt

### Below is the overview of the dataset from Udacity:

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**

* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**

* age (int) - age of the customer
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**

* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record
