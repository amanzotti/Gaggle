import pandas as pd
import numpy as np
from sklearn.pipeline import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
# import sklearn.metrics.mean_squared_error
import scipy.stats

df = pd.read_csv('./train.csv')
categorical = [key for key in df.keys() if df.dtypes[key] == np.dtype('O')]
numeric = [key for key in df.keys() if df.dtypes[key] != np.dtype('O')]
# correct naive expectations
actual_categoric = ['MSSubClass']
numeric = list(set(numeric) - set(actual_categoric))
categorical = list(set(categorical).union(set(actual_categoric)))

for cat in categorical:
    df[cat] = df[cat].astype('category')
_test_categories = {key: df[key].cat.categories for key in categorical}


class categorical_extractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts """

    def __init__(self):
        pass

    def find_categorical(self, df):
        """Helper code to compute average word length of a name"""
#         print(type(df),df.ndim)
        categorical = [key for key in df.keys() if df.dtypes[key] == np.dtype('O')]
        numeric = [key for key in df.keys() if df.dtypes[key] != np.dtype('O')]
        # correct naive expectations
        actual_categoric = ['MSSubClass']
        numeric = list(set(numeric) - set(actual_categoric))
        categorical = list(set(categorical).union(set(actual_categoric)))
        return categorical

    def transform(self, df):
        """The workhorse of this feature extractor"""
        _df = df.copy()
        categorical = self.find_categorical(df)
        for cat in categorical:
            _df[cat] = pd.Categorical(_df[cat], categories=_test_categories[cat])

        return pd.get_dummies(_df[categorical]).as_matrix()
#         return pd.get_dummies(_df[categorical])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
#         self.index_ = X.index
#         self.columns_ = X.columns
#         self.cat_columns_ = X.select_dtypes(include=['object']).columns
#         self.non_cat_columns_ = X.columns.drop(self.cat_columns_)
#         self.cat_map_ = {col: X[col].name for col in self.cat_columns_}
#         left = len(self.non_cat_columns_)
#         self.cat_blocks_ = {}
# #         for col in self.cat_columns_:
# #             right = left + len(X[col].cat.categories)
# #             self.cat_blocks_[col], left = slice(left, right), right
        return self


class numerical_extractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts """

    def __init__(self):
        pass

    def find_numerical(self, df):
        """Helper code to compute average word length of a name"""
        # categorical = [key for key in df.keys() if df.dtypes[key] == np.dtype('O')]
        numeric = [key for key in df.keys() if df.dtypes[key] != np.dtype('O')]
        # correct naive expectations
        actual_categoric = ['MSSubClass']
        numeric = list(set(numeric) - set(actual_categoric))
        return numeric

    def transform(self, df):
        """The workhorse of this feature extractor"""
        numerical = self.find_numerical(df)
        #         filna with median
        df_ = df.copy()
        for key in numerical:
            df_[key].fillna(df_[key].median(), inplace=True)
        return StandardScaler().fit_transform(np.asarray(df_[numerical]))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class preprocess(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts """

    def __init__(self):
        pass

    def find_categorical(self, df):
        """Helper code to compute average word length of a name"""
#         print(type(df),df.ndim)
        categorical = [key for key in df.keys() if df.dtypes[key] == np.dtype('O')]
        numeric = [key for key in df.keys() if df.dtypes[key] != np.dtype('O')]
        # correct naive expectations
        actual_categoric = ['MSSubClass']
        numeric = list(set(numeric) - set(actual_categoric))
        categorical = list(set(categorical).union(set(actual_categoric)))
        return categorical

    def find_numerical(self, df):
        """Helper code to compute average word length of a name"""
        # categorical = [key for key in df.keys() if df.dtypes[key] == np.dtype('O')]
        numeric = [key for key in df.keys() if df.dtypes[key] != np.dtype('O')]
        # correct naive expectations
        actual_categoric = ['MSSubClass']
        numeric = list(set(numeric) - set(actual_categoric))
        return numeric

    def transform(self, df):
        """The workhorse of this feature extractor"""
        _df = df.copy()
        _df['Age'] = _df['YrSold'] - _df['YearBuilt']
        _df['AgeRemod'] = _df['YrSold'] - _df['YearRemodAdd']
        _df['Baths'] = _df['FullBath'] + _df['HalfBath']
        _df['BsmtBaths'] = _df['BsmtFullBath'] + _df['BsmtHalfBath']
        _df['OverallQual_Square'] = _df['OverallQual'] * _df['OverallQual']
        _df['OverallQual_3'] = _df['OverallQual'] * _df['OverallQual'] * _df['OverallQual']
        _df['OverallQual_exp'] = np.exp(_df['OverallQual'])
        _df['GrLivArea_Square'] = _df['GrLivArea'] * _df['GrLivArea']
        _df['GrLivArea_3'] = _df['GrLivArea'] * _df['GrLivArea'] * _df['GrLivArea']
        _df['GrLivArea_exp'] = np.exp(_df['GrLivArea'])
        _df['GrLivArea_log'] = np.log(_df['GrLivArea'])
        _df['TotalBsmtSF_/GrLivArea'] = _df['TotalBsmtSF'] / _df['GrLivArea']
        _df['OverallCond_sqrt'] = np.sqrt(_df['OverallCond'])
        _df['OverallCond_square'] = _df['OverallCond'] * _df['OverallCond']
        _df['LotArea_sqrt'] = np.sqrt(_df['LotArea'])
        _df['1stFlrSF_log'] = np.log1p(_df['1stFlrSF'])
        _df['1stFlrSF'] = np.sqrt(_df['1stFlrSF'])
        _df['TotRmsAbvGrd_sqrt'] = np.sqrt(_df['TotRmsAbvGrd'])
        categorical = self.find_categorical(df)
        numerical = self.find_numerical(df)

        for cat in categorical:
            _df[cat] = pd.Categorical(_df[cat], categories=_test_categories[cat])

        for key in numerical:
            _df[key].fillna(_df[key].median(), inplace=True)

        # #if numerical feature are skewed apply log

        skewed_feats = _df[numerical].apply(lambda x: scipy.stats.skew(x.dropna()))  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index
        _df[skewed_feats] = np.log1p(_df[skewed_feats])
        cat_matrix = pd.get_dummies(_df[categorical]).as_matrix()
        num_matrix = StandardScaler().fit_transform(np.asarray(_df[numerical]))

        return np.hstack((num_matrix, cat_matrix))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
#         self.index_ = X.index
#         self.columns_ = X.columns
#         self.cat_columns_ = X.select_dtypes(include=['object']).columns
#         self.non_cat_columns_ = X.columns.drop(self.cat_columns_)
#         self.cat_map_ = {col: X[col].name for col in self.cat_columns_}
#         left = len(self.non_cat_columns_)
#         self.cat_blocks_ = {}
# #         for col in self.cat_columns_:
# #             right = left + len(X[col].cat.categories)
# #             self.cat_blocks_[col], left = slice(left, right), right
        return self
