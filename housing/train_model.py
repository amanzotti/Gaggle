import pandas as pd
import numpy as np
import sklearn
import utils
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

import warnings
warnings.filterwarnings("ignore")

pre = utils.preprocess()
data = pd.read_csv('./train.csv')
feat = data.drop('SalePrice', 1)
target = np.asarray(np.log(data.SalePrice))
clf = sklearn.linear_model.Ridge(alpha=4.)


# print(pre.transform(feat))

# sys.exit()

# model = sklearn.linear_model.Ridge()
# param_grid = dict(clf__alpha=np.logspace(-4, 1.2, 500))

# model = sklearn.svm.SVR(kernel='linear')
# param_grid = dict(clf__C=np.linspace(0.1, 4, 20))


name = 'kernel_ridge'

model = KernelRidge(kernel='rbf')
param_grid = {"clf__alpha": np.logspace(-4, 1.2, 40),
              "clf__gamma": np.logspace(-3, 1, 5)}



# name = 'svr_rbf'

# model = sklearn.svm.SVR(kernel='rbf')
# param_grid = {"clf__C": np.logspace(1, 3, 20),
#               "clf__gamma": np.logspace(-2, 2, 5), "clf__epsilon": np.logspace(-1, 1, 5)}


# name = 'lasso'
# model = sklearn.linear_model.Lasso()
# param_grid = dict(clf__alpha=np.logspace(np.log(0.00005), np.log(2.5), 400))


# model = DecisionTreeRegressor()

# param_grid = {
#     'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
#     'clf__min_samples_leaf': [1, 3, 5, 9, 11, 13, 17],
#     'clf__max_features': [1.0, 0.3, 0.1, 0.5, 0.7]  # not possible in our example (only 1 fx)
# }

# model = sklearn.ensemble.AdaBoostRegressor()


# param_grid = {'clf__n_estimators': (6,20,50,100,200,300,400,500,600,700,850),
#               'clf__learning_rate': (0.1, 0.3,0.7,1.)
#               }


# model = linear_model.BayesianRidge()
# param_grid = {'clf__n_iter': (6,20,50,100,200,300,400,500,600,700,850)
#               }


# model = neighbors.KNeighborsRegressor()
# param_grid = {'clf__n_neighbors': [100, 300, 500],
#               'clf__weights': ['uniform', 'distance']}


# param_grid = dict(clf__alpha=np.linspace(0,10,30))


pipe = Pipeline(steps=[
    # ('select_feat', RFECV(estimator=clf)),
    ('clf', model)  # classifier
])

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='mean_squared_error', verbose=10)

grid.fit(pre.transform(feat), target)

pk.dump(grid, open("{}_grid.p".format(name), "wb"))

test = pd.read_csv("./test.csv")
preds = np.exp(grid.best_estimator_.predict(pre.transform(test)))
solution = pd.DataFrame({"id": test.Id, "SalePrice": preds})
solution.to_csv("{}_sol.csv".format(name), index=False)
