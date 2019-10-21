# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html --> scikit-learn tutorial to train multiple models simultaneuously for experimentation
# https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn --> kaggle tutorial to train multiple models simultaneously for experimentation
# https://scikit-learn.org/stable/modules/multiclass.html --> scikit-learn list of viable multiclass classifiers (we want "inherently multiclass" or "one vs all")
# https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html --> outlines the "one vs all" fun ction in more detail, but the link above also includes the code

#"One vs All" method says that each article can only be assigned to one bias class. In order to do this, the sklearn.multiclass.OneVsRestClassifier function will create five bias models. the first model
# predicts liberal articles, the second predicts center-liberal, the third center, the fourth center conservative, and the fifth conservative. For example, the liberal model will look at everything that
# makes center articles different than all the rest and return 1 if a test article is center or 0 if it is anything else. It loops through the five models where each picks out articles that belong to that
# bias label

# AJ's Note --> Objective is to train, test, and plot several multiclass classification models to figure out which one works best with our data to predict bias of articles. I am most familiar with
# accuracy, precision, recall, and F1 measures to determine model performance. So can we plot all those for each model?


# Data processing / visualization]
import math
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

#Classification Models
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier

#Performance measurements
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import logging
import os.path
import logging.config

import pickle

#Import Data & Preparation
def main():

   logging.info("Inserting articles into news_corpus[]")
   X = pd.read_csv('../data/NewsCourpusRFECV.csv',encoding='utf-8')
   Y = pd.read_csv('../data/articles_trimmed.csv', error_bad_lines=False)

   Y = Y['political_bias']

   logging.info(X.shape)
   logging.info(Y.shape)

   parameters = {
         'n_estimators':
         [
            50, 75, 100, 250, 500
         ],
         'learning_rate':
         [
            1.0, 0.75, 0.50, 0.25, 0.10, 0.01
         ],
         'min_samples_split':
         [
            0.1, 1.0, 10
         ],
         'max_depth':
         [
            1, 32, 32
         ],
         'max_features':
         [
            'auto', 'log2', 'None'
         ]
   }

   name = 'Gradient Boosting'

   logging.info(f"Running {name} Classifier")

   logging.info("Un-pickling Classifier")
   gbc = pickle.load(open('../data/GradientBoosting.sav', 'rb'))

   logging.info("Splitting X and Y into x_train, x_test, y_train, y_test. test_size=0.22, random_state=42")
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, random_state=42, shuffle=True)

   logging.info("Preforming clf.fit")
   gbc.fit(x_train, y_train)

   clf = GridSearchCV(estimator=gbc, param_grid=parameters, cv=StratifiedKFold(5))

   logging.info("creating train_prefictions from clf.predict(X)")
   train_predictions = clf.predict(x_test)

   f1 = f1_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
   logging.info("Results (Accuracy): {:.4%}".format(f1))

   cm = confusion_matrix(y_test, train_predictions, labels=None, sample_weight=None)
   logging.info(f"Confusion Matrix: \n{cm}")

   logging.info("="*30)

if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO,
   format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
   handlers=[
      logging.FileHandler("../logs/ml.log"),
      logging.StreamHandler()
   ])

   logging.info('Bias_ml.py started')

   main()

   logging.info('Bias_ml.py Finished')

   logging.shutdown()


# Once we choose which models to use and refine the chosen models, we can do one of two things:
# 1) Choose the highest performing model to deploy
# 2) Keep all well-performing models and create a voting system. Ex: 6 models say article is center, 1 says it is liberal, and 1 says it is conservative.
#    Voting system would take the majority vote to label it center
