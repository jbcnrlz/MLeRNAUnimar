import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main():
    ddo = pd.read_csv("diamonds.csv")
    ddm = ddo.drop(["Unnamed: 0"],axis=1)
    ddm = ddm.drop(ddm[ddm["x"] == 0].index)    
    ddm = ddm.drop(ddm[ddm["y"] == 0].index)
    ddm = ddm.drop(ddm[ddm["z"] == 0].index)

    le = LabelEncoder()
    ddm['cut'] = le.fit_transform(ddm['cut'])
    ddm['color'] = le.fit_transform(ddm['color'])
    ddm['clarity'] = le.fit_transform(ddm['clarity'])
    x = ddm.drop(["price"],axis=1)
    y = ddm["price"]

    #x_treino, x_test, y_treino, y_test = train_test_split(x,y,test_size=0.2)

    lr = LinearRegression()
    scores = cross_val_score(estimator=lr,X=x,y=y,scoring='neg_root_mean_squared_error',cv=10)
    print(scores.mean()*-1)

    knnr = KNeighborsRegressor(n_neighbors=3)
    scores = cross_val_score(estimator=knnr,X=x,y=y,scoring='neg_root_mean_squared_error',cv=10)
    print(scores.mean()*-1)

    lal = Lasso()
    scores = cross_val_score(estimator=lal,X=x,y=y,scoring='neg_root_mean_squared_error',cv=10)
    print(scores.mean()*-1)

    rdr = Ridge()
    scores = cross_val_score(estimator=rdr,X=x,y=y,scoring='neg_root_mean_squared_error',cv=10)
    print(scores.mean()*-1)

    svmR = RandomForestRegressor(n_estimators=100)
    scores = cross_val_score(estimator=svmR,X=x,y=y,scoring='neg_root_mean_squared_error',cv=10)
    print(scores.mean()*-1)

if __name__ == '__main__':
    main()