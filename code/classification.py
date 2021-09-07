import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [
    "Nearest Neighbors", 
    "Linear SVM", 
    "RBF SVM", 
    "Gaussian Process",
    "Decision Tree", 
    "Random Forest", 
    "Neural Net", 
    "AdaBoost", 
    "GradientBoosting",
    "Naive Bayes", 
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def train(path, standardize, cv):
    if path == "mnist10k":
        X = pd.read_csv("./data/mnist10k/features.csv").values
        y = pd.read_csv("./data/mnist10k/label.csv").values.ravel()
    else:
        raise ValueError("Path to data must be specified.")
    
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    
    kf = StratifiedKFold(n_splits=cv, shuffle=True)

    datasets = [
        (X[train_index], X[test_index], y[train_index], y[test_index]) for train_index, test_index in kf.split(X, y) 
    ]
    
    print("name, fold, train_acc, test_acc, test_mcc")
    for name, clf in zip(names, classifiers):
        for ds_cnt, ds in enumerate(datasets):
            X_train, X_test, y_train, y_test = ds
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            train_acc = clf.score(X_train, y_train)
            acc = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            print(f"{name}, {ds_cnt+1}, {train_acc}, {acc}, {mcc}")