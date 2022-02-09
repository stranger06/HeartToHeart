import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import seaborn as sb
# from sklearn.ensemble import RandomForestClassifier : dumped during experimentation.
# from sklearn.neighbors import KNeighborsClassifier  : dumped during experimentation.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, plot_roc_curve, classification_report
def fit_and_score(models, X_train, X_test, Y_train, Y_test):
    """
    Fits and evaluates given machine learning models.
    models  : a dict of models to be fitted.
    X_train : training data (no labels).
    X_test  : testing data (no labels)..
    Y_train : training labels.
    Y_test  : testing labels.
    """
    np.random.seed(2)
    model_scores={}
    for name,model in models.items():
        model.fit(X_train, Y_train)
        model_scores[name] = model.score(X_test, Y_test)
    return model_scores
def plot_conf_mat(Y_test, Y_preds):
    """
    Accepts Y_test and Y_preds as paramters and plots a Confusion Matrix using seaborn's heatmap.
    """
    fig, ax = pt.subplots(figsize=(6,4))
    ax = sb.heatmap(confusion_matrix(Y_test, Y_preds), annot=True, cmap="PuRd")
    pt.xlabel("True Label")
    pt.ylabel("False Label");
plot_conf_mat(Y_test, Y_preds)
print(classification_report(Y_test, Y_preds))
def cv_score(clf, X, Y, n):
    """
    Accepts -
    classifier: the model to use.
    X: the data without labels.
    Y: the labels.
    n: number of cross validations to perform.
    
    Prints out (both textually and visually) the cross validation score for accuracy, precision, recall and f1 score. 
    """
    cv_acc = cross_val_score(clf, X, Y, cv=n, scoring="accuracy")
    cv_prec = cross_val_score(clf, X, Y, cv=n, scoring="precision")
    cv_rec = cross_val_score(clf, X, Y, cv=n, scoring="recall")
    cv_f1 = cross_val_score(clf, X, Y, cv=n, scoring="f1")
    cv_acc_mean = np.mean(cv_acc)
    cv_prec_mean = np.mean(cv_prec)
    cv_rec_mean = np.mean(cv_rec)
    cv_f1_mean = np.mean(cv_f1)
    print(f"Cross Validation Score Chart ( cv = {n} )")
    print("---------------------------------------")
    print(f"Accuracy:  {cv_acc} -> {(cv_acc_mean)*100:.2f}%")
    print(f"Precision: {cv_prec} -> {(cv_prec_mean)*100:.2f}%")
    print(f"Recall:    {cv_rec} -> {(cv_rec_mean)*100:.2f}%")
    print(f"F1 score:  {cv_f1} -> {(cv_f1_mean)*100:.2f}%")
    pd.DataFrame({"Accuracy": cv_acc_mean, "Precision": cv_prec_mean,
                  "Recall": cv_rec_mean, "F1 Score": cv_f1_mean}, index=[0]).T.plot.bar(color="lavender", legend=False, zorder=0, figsize=(6,4))
    pt.xticks(rotation=0)
df = pd.read_csv("data/heart-disease.csv")
corr_matrix = df.corr()
fig, ax = pt.subplots(figsize=(20,8))
ax = sb.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt='0.2f')
np.random.seed(6)
lr_grid = {"C": np.logspace(-4,4,20), "solver": ["liblinear"]}
lr_rscv = RandomizedSearchCV(LogisticRegression(), param_distributions=lr_grid, cv=5, n_iter=6)
lr_rscv.fit(X_train, Y_train)
Y_preds = lr_rscv.predict(X_test)
plot_roc_curve(lr_rscv, X_test, Y_test, color="lime");
cv_score(lr_rscv, X, Y, 6)
clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")
clf.fit(X_train, Y_train)
print(clf.coef_)
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
print(feature_dict)
