from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

def main():
    data = load_iris()
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    rdf = RandomForestClassifier(n_estimators=100)
    rdf.fit(x_train,y_train)

    yPred = rdf.predict(x_test)
    print(classification_report(y_test,yPred))

    xboost = GradientBoostingClassifier()
    xboost.fit(x_train,y_train)
    yPred = xboost.predict(x_test)
    print(classification_report(y_test,yPred))

if __name__ == '__main__':
    main()