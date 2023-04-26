import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

# Calcul variance
# data.drop(['Index'], axis=1, inplace=True)
# print(data.var(numeric_only=True).abs().sort_values(ascending=False))

def showOne(data):
    data = pd.read_csv(args.input)
    plt.figure('Histogram')
    ax = sns.histplot(data=data, x="Care of Magical Creatures", hue="Hogwarts House")
    ax.set_ylabel('Students')
    plt.show()   

def showAll(data):
    fig, axs = plt.subplots(3,4, figsize = (16,8), num='Histogram')
    sns.histplot(data=data, x="Arithmancy",  hue="Hogwarts House", ax=axs[0, 0]).set_ylabel('Students')
    sns.histplot(data=data, x="Muggle Studies",  hue="Hogwarts House",  ax=axs[0, 1]).set_ylabel('Students')
    sns.histplot(data=data, x="Astronomy",   hue="Hogwarts House", ax=axs[0, 2]).set_ylabel('Students')
    sns.histplot(data=data, x="Flying",   hue="Hogwarts House", ax=axs[0, 3]).set_ylabel('Students')
    sns.histplot(data=data, x="Ancient Runes",  hue="Hogwarts House", ax=axs[1, 0]).set_ylabel('Students')
    sns.histplot(data=data, x="Transfiguration", hue="Hogwarts House", ax=axs[1, 1]).set_ylabel('Students')
    sns.histplot(data=data, x="Charms",  hue="Hogwarts House", ax=axs[1, 2]).set_ylabel('Students')
    sns.histplot(data=data, x="Potions",  hue="Hogwarts House", ax=axs[1, 3]).set_ylabel('Students')
    sns.histplot(data=data, x="Herbology",  hue="Hogwarts House", ax=axs[2, 0]).set_ylabel('Students')
    sns.histplot(data=data, x="History of Magic",  hue="Hogwarts House", ax=axs[2, 1]).set_ylabel('Students')
    sns.histplot(data=data, x="Defense Against the Dark Arts",  hue="Hogwarts House", ax=axs[2, 2]).set_ylabel('Students')
    sns.histplot(data=data, x="Divination",  hue="Hogwarts House", ax=axs[2, 3]).set_ylabel('Students')
    fig.tight_layout()
    plt.show()

# Main Function
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser(description='GraphHistogram @Paris 42 School - Made by @abbensid & @bhamdi')
    parser.add_argument("input", type=str, help="The file containing dataset")
    parser.add_argument("-a", "--all", action="store_true", help="Show All Histogram")
    args = parser.parse_args()
    try:
        print(f"[{Color.BLUE}Display Histogram{Color.END}]")
        data = pd.read_csv(args.input)
        data.drop(['Index'], axis=1, inplace=True)
        if (args.all):
            showAll(data)
        else:
            showOne(data)
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)

