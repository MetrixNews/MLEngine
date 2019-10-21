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

import sys
sys.path.append('../')
import Common.Logging as log
import pickle

#Import Data & Preparation
def main():
   news_corpus = []

   logging.info("Inserting articles into news_corpus[]")
   X = pd.read_csv('../data/NewsCourpusRFECV.csv')
   Y = pd.read_csv('../data/articles_trimmed.csv')

   #Cross Validation Data Partition
   Y = Y['political_bias']

   logging.info("Preforming get_dummie on X")

   clf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=70, min_samples_split=500, min_samples_leaf=50, max_depth=20, max_features='sqrt', subsample=0.8)

   name = 'Gradient Boosting'

   #rfecv = RFECV(estimator=clf, step=50, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)

   logging.info(f"Running {name} Classifier")

   logging.info("Running RFECV")

   #rfecv.fit(X, Y)

   #logging.info('Optimal number of features: {}'.format(rfecv.n_features_))

   #logging.info(f'Dropping Features: {np.where(rfecv.support_ == False)[0]}')
   #X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

   #export_csv = X.to_csv(r'../Data/NewsCourpusRFECV.csv', index=None, header=True)

   logging.info("Splitting X and Y into x_train, x_test, y_train, y_test. test_size=0.22, random_state=42")
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, random_state=42, shuffle=True)

   logging.info("Preforming clf.fit")
   clf.fit(x_train, y_train)

   # logging.info("creating train_prefictions from clf.predict(x_train)")
   # train_predictions = clf.predict(x_test)

   # f1 = f1_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
   # logging.info("Results (Accuracy): {:.4%}".format(f1))

   # cm = confusion_matrix(y_test, train_predictions, labels=None, sample_weight=None)
   # logging.info(f"Confusion Matrix: \n{cm}")

   pickle.dump(clf, open('../data/GradientBoosting.sav', 'wb'))

   # #plotting
   # logging.info("Plotting rfecv results")
   # fig = plt.figure(figsize=(16,14))
   # plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
   # plt.xlabel('Number of features selected', fontsize=14, labelpad=1)
   # plt.ylabel('% Correct Classification', fontsize=14, labelpad=1)
   # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
   # plt.savefig('../data/Recursive Feature Elimination With Cross-Validation.png')
   # plt.close(fig)

   # dset = pd.DataFrame()
   # dset['attr'] = X.columns
   # dset['importance'] = rfecv.estimator_.feature_importances_

   # dset = dset.sort_values(by='importance', ascending=False)
   # fig2 = plt.figure(figsize=(30, 100))
   # plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
   # plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
   # plt.xlabel('Importance', fontsize=14, labelpad=20)
   # plt.savefig('../data/RFECV - Feature Importance.png')
   # plt.close(fig2)

   logging.info("="*30)


if __name__ == "__main__":
   logging = log.Logger('../logs/ml.log', 'ML Engine')

   logging.info('Bias_ml started')

   main()

   logging.info('Bias_ml Finished')

   logging.shutdown()


# Once we choose which models to use and refine the chosen models, we can do one of two things:
# 1) Choose the highest performing model to deploy
# 2) Keep all well-performing models and create a voting system. Ex: 6 models say article is center, 1 says it is liberal, and 1 says it is conservative.
#    Voting system would take the majority vote to label it center
