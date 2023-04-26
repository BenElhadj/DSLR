import numpy as np
import pandas as pd
import argparse
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

class LogisticRegressionTraining(object):
    
    def __init__(self, alpha=0.1, n_iteration=100):  
        self.alpha = alpha                            
        self.n_iter = n_iteration

    def ft_sigmoid(self, x): 
        value = 1 / (1 + np.exp(-x))
        return value
    
    def ft_cost(self,h, y):
        m = len(y)
        cost = (1 / m) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
        return cost
    
    def ft_gradient_descent(self, X, h, theta, y, m): 
        gradient_value = np.dot(X.T, (h - y)) / m
        theta -= self.alpha * gradient_value
        return theta


    def fit(self, X, y):
        self.theta = []
        self.cost = []
        X = np.insert(X, 0, 1, axis=1)
        m = len(y)
        for i in np.unique(y):
            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for j in range(self.n_iter):
                z = X.dot(theta)
                h = self.ft_sigmoid(z)
                theta = self.ft_gradient_descent(X, h, theta, y_onevsall, m)
                cost.append(self.ft_cost(h, y_onevsall))
                print(f'Gradient descent: {Color.WARNING}{int((j / self.n_iter) * 100)}%{Color.END}\r', end="", flush=True)
            print(f"Trainning class: {Color.GREEN} {i:{' '}<11} ✔ {Color.END}")
            self.theta.append((theta, i))
            self.cost.append((cost,i))
        return self.theta
    
    def preprocessing(self, pathData):
        data = pd.read_csv(pathData)
        data = data.dropna(subset=['Defense Against the Dark Arts', 'Charms', 'Herbology', 'Divination', 'Muggle Studies', 'Flying', 'Ancient Runes'])
        self.y_data = data['Hogwarts House']
        X_train = data[[
                'Defense Against the Dark Arts',
                'Divination',
                'Charms',
                'Flying',
                'Herbology',
                'Muggle Studies',
                'Ancient Runes', 
            ]]
        self.X = StandardScaler().fit_transform(X_train)
        return self

        # Save weight.csv
    def saveWeight(self, theta):
        theta = pd.DataFrame([ilem[0] for ilem in theta], index=[ilem[1] for ilem in theta])
        theta.to_csv('weight.csv')

    def visualisation(self):
        df = pd.DataFrame({'Gryffindor':self.cost[0][0], 'Hufflepuff' :self.cost[1][0], 'Ravenclaw':self.cost[2][0], 
        'Slytherin':self.cost[3][0]})
        sns.lineplot(df)
        plt.xlabel('n_iteration')
        plt.ylabel('Log_loss')
        plt.title('Evolution des erreurs')
        plt.show()


# Main Function
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser(description='LogisticRegression @Paris 42 School - Made by @abbensid & @bhamdi')
    parser.add_argument("input", type=str, help="The file containing datasets training")
    parser.add_argument("-s", "--stoch", action="store_true", help="gradient stochastic")
    parser.add_argument("-l", "--cost", action="store_true", help="graph loss cost")
    args = parser.parse_args()
    try:
        logi = LogisticRegressionTraining(n_iteration=1000, alpha=0.01).preprocessing(args.input)
        if (args.stoch):
            indexes = np.random.randint(0, len(logi.X), 250)
            logi.X = np.take(np.array(logi.X), indexes, axis=0)
            logi.y_data = np.take(np.array(logi.y_data), indexes)
        theta = logi.fit(logi.X, logi.y_data)
        logi.saveWeight(theta)
        print(f"[{Color.BLUE}Model Trained{Color.END}]")
        if (args.cost):
            logi.visualisation()
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)