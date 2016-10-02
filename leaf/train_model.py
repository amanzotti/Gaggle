import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
import sklearn.linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error
import pickle as pk
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import sklearn.ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import norm
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)


import warnings
warnings.filterwarnings("ignore")

# name = 'log_reg'
# clf = sklearn.linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')
# # param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))
# param_dist = {"C": norm(loc=950.0, scale=200.),
#               "tol": uniform(loc=0.0001, scale=0.008)
#               }

# name = 'ada'
# clf = AdaBoostClassifier()
# # param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))
# param_dist = {"n_estimators": randint(low=100, high=600),
#               "learning_rate": uniform(loc=0.01, scale=0.99)
#               }

# name = 'LDA'
# clf = LinearDiscriminantAnalysis(solver = 'lsqr')
# # param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))
# param_dist = {
#     "shrinkage": uniform(loc=0.0001, scale=0.99)
# }

name = 'RF'
clf = RandomForestClassifier()
# param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))
param_dist = {"n_estimators": randint(low=100, high=600),
              "max_depth": [3, None],
              "max_features": randint(1, 11),
              "min_samples_split": randint(1, 11),
              "min_samples_leaf": randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]
              }


# name = 'Kneigh'
# clf = KNeighborsClassifier()
# # param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))
# param_dist = {"n_neighbors": randint(low=2, high=6),
#               }


random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=2000, scoring='log_loss', n_jobs=-1, cv=8, verbose=10)


random_search.fit(x_train, y_train)

pk.dump(random_search, open("{}_grid.p".format(name), "wb"))

test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)
y_test = random_search.best_estimator_.predict_proba(x_test)
solution = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
solution.to_csv("{}_sol.csv".format(name))
