import pandas as pd
import numpy as np
import sys
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class LogisticRegressionProdicton(object):
    
    def ft_sigmoid(self, x):
        value = 1 / (1 + np.exp(-x))
        return value

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return [max((self.ft_sigmoid(i.dot(theta)), c) for theta, c in self.theta)[1] for i in X ]
    
    def score(self,X, y):
        score = sum(self.predict(X) == y) / len(y)
        return score
   
    # Set theta
    def setTheta(self, pathTheta='weight.csv'):
        theta_csv = pd.read_csv(pathTheta)
        self.theta = []
        for ilem in theta_csv.values:
            self.theta.append((ilem[1:], ilem[0]))
        return self.theta
    
    # preprocessing data_train
    def preprocessingDataTrain(self, pathDataTrain):
        data = pd.read_csv(pathDataTrain)
        self.transformer = LabelEncoder()
        self.y_train = data['Hogwarts House']
        X_train = data[[
                'Defense Against the Dark Arts',
                'Divination',
                'Charms',
                'Flying',
                'Herbology',
                'Muggle Studies',
                'Ancient Runes', 
            ]] 
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X_train.values)
        return self
    
       # preprocessing data_test
    def preprocessingDataTest(self, pathDataTest):
        data_test = pd.read_csv(pathDataTest)
        data_test = data_test.fillna(method='ffill')
        data_test = data_test[[
                'Defense Against the Dark Arts',
                'Divination',
                'Charms',
                'Flying',
                'Herbology',
                'Muggle Studies',
                'Ancient Runes', 
            ]]
        
        x_test = data_test.values
        self.X_test = self.scaler.transform(x_test)

        return self

        # Save houses.csv
    def saveHouses(self):
        pred = self.predict(self.X_test)
        pred = pd.DataFrame(pred, columns=['Hogwarts House'])
        pred.index.name = 'Index'
        pred.to_csv('houses.csv')

    def predictTest(self, graph=False):
        X_train,X_test,y_train,y_test = train_test_split(logi.X, logi.y_train, test_size = 0.2)
        X_test[np.isnan(X_test)] = 0
        pred = logi.predict(X_test)
        print(pd.crosstab(y_test, pred, margins=True, margins_name="Total"))
        score = logi.score(X_test, y_test)
        print(f"[{Color.GREEN}The accuracy of the model is{Color.END}] [score: {Color.GREEN}{round(score, 2)}{Color.END}]")
        self.visualisation(y_test, pred) if graph else False
    
    def visualisation(self, y_test, pred):
        ct = pd.crosstab(y_test, pred)
        plt = ct.plot(kind="bar", stacked=True, rot=0)
        plt.legend(title='Hogwarts House')
        plt.set_title('test data')
        plt.set_xlabel('House')
        plt.set_ylabel('count')
        for count, val in enumerate(ct.columns):
            plt.annotate(str(max(ct[val])), xy=(count, max(ct[val])), ha='center', va='bottom', weight="bold")
        plot.show()
    
# Main Function
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser(description='LogisticRegression @Paris 42 School - Made by @abbensid & @bhamdi')
    parser.add_argument("puTrain", type=str, help="The file containing dataset training")
    parser.add_argument("puTest", type=str, help="The file containing dataset testing")
    parser.add_argument("-t", "--test", action="store_true", help="test")
    parser.add_argument("-g", "--graph", action="store_true", help="graph test data")
    args = parser.parse_args()
    try:
        logi = LogisticRegressionProdicton()
        logi.preprocessingDataTrain(args.puTrain)
        logi.preprocessingDataTest(args.puTest)
        logi.setTheta()
        if (args.test):
            logi.predictTest(graph = True if args.graph else False)
        else:
            logi.saveHouses()
            print(f"[{Color.BLUE}File houses.csv create{Color.END}]")
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)