import os

import pandas as pd
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Data downloading script
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if ('dataset.csv' not in os.listdir('../Data')):
    print('Dataset loading.')
    url = "https://www.dropbox.com/s/0sj7tz08sgcbxmh/large_movie_review_dataset.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    # The dataset is saved to `Data` directory
    open('../Data/dataset.csv', 'wb').write(r.content)
    print('Loaded.')

# write your code here
df_ = pd.read_csv("../Data/dataset.csv")


def stage1(df):
    # creating a df filtered and filtering by criteria
    dff = df.copy()
    index_rating = dff[(dff['rating'] > 5) & (dff['rating'] < 7)].index
    dff.drop(index_rating, inplace=True)
    dff['label'] = dff['rating'].apply(lambda x: 1 if x > 5 else 0)
    dff.drop(columns=["rating"], inplace=True)
    # print(df.shape[0], end=" ")
    # print(dff.shape[0], end=" ")
    # print(dff["label"][dff["label"] == 1].count() / dff.shape[0])
    return dff


def stage2():
    dff = stage1(df_)
    # splitting the dataframe into 2 parts
    X_train, X_test, y_train, y_test = train_test_split(dff["review"], dff["label"], random_state=23)
    vectorizer1 = TfidfVectorizer(sublinear_tf=True)
    # creating feature matrix with the training set
    X_train_feat_matrix = vectorizer1.fit_transform(X_train)
    # print(X_train_feat_matrix.shape)
    v = vectorizer1.get_feature_names_out()

    # creating feature matrix with the test set
    # alternative 1
    # fit is the calculation-training part, transform is applying calculations to data.
    X_test_feat_matrix = vectorizer1.transform(X_test)
    # alternative 2
    # vectorizer2 = TfidfVectorizer(vocabulary=vectorizer1.vocabulary_, sublinear_tf=True)
    # X_test_feat_matrix = vectorizer2.fit_transform(X_test)
    v = vectorizer1.get_feature_names_out()
    return X_train_feat_matrix, X_test_feat_matrix, y_train, y_test


def stage3():
    # https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
    X_train, X_test, y_train, y_test = stage2()
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_predict)
    # predict_proba returns probability for 0 and 1.
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(acc_score.__round__(4))
    print(auc_score.__round__(4))


def stage4():
    X_train, X_test, y_train, y_test = stage2()
    # the smaller the C the more strictly features are selected
    model = LogisticRegression(solver="liblinear", penalty="l1", C=0.15)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    # accuracy after lasso (L1-regularization). Setting coef of extra features to null.
    acc_score = accuracy_score(y_test, y_predict)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(acc_score.__round__(4))
    print(auc_score.__round__(4))
    # finding coefficients after lasso which is not null
    print(len([c for c in model.coef_[0] if abs(c) > 0.0001]))


def stage5():
    # getting the sets from stage2
    X_train, X_test, y_train, y_test = stage2()

    # n=100 rounded from '102' important features from stage4
    svd_model = TruncatedSVD(n_components=100)
    # calculating truncated arrays for X_train and X_test
    X_train_trunc = svd_model.fit_transform(X_train)
    X_test_trunc = svd_model.transform(X_test)

    # creating the model and fitting with truncated data
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train_trunc, y_train)
    # calculating the predictions
    y_predict = model.predict(X_test_trunc)
    acc_score = accuracy_score(y_test, y_predict)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test_trunc)[:, 1])
    print(acc_score.__round__(4))
    print(auc_score.__round__(4))


stage5()
