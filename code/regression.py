import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, ElasticNet

REGRESSORS = {
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('LinearRegression', LinearRegression()),
    ('HuberRegressor', HuberRegressor()),
    ('ElasticNet', ElasticNet()),
    ('LinearSVR', LinearSVR()),
    ('SVR', SVR()),
    ('NuSVR', NuSVR()),
    ('GradientBoostingRegressor', GradientBoostingRegressor()),
    ('AdaBoostRegressor', AdaBoostRegressor()),
    ('GaussianProcessRegressor', GaussianProcessRegressor()),
    ('MLPRegressor', MLPRegressor()),
}

def train_cv(path, standardize, cv):
    if path == "solar":
        df = pd.read_csv("./data/solar/solar.csv")
        X = df[["Solar_rad","Temp","TempAmb"]].values
        y = df[['INV01.Ppv']].values.ravel()
    else:
        raise ValueError("Path to data must be specified.")
    
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    
    kf = KFold(n_splits=cv, shuffle=True)

    datasets = [
        (X[train_index], X[test_index], y[train_index], y[test_index]) for train_index, test_index in kf.split(X, y) 
    ]
    
    print("name, fold, Train_R2, R2, MSE, RMSE")
    for name, reg in REGRESSORS:
        for ds_cnt, ds in enumerate(datasets):
            X_train, X_test, y_train, y_test = ds
            reg.fit(X_train,y_train)
            self_rsq = reg.score(X_train, y_train)
            rsq = reg.score(X_test, y_test)
            mse = mean_squared_error(y_test, reg.predict(X_test))
            rmse = mean_squared_error(y_test, reg.predict(X_test), squared=False)
            print(f"{name}, {ds_cnt+1}, {self_rsq}, {rsq}, {mse}, {rmse}")