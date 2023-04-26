import sys
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def preprocessing(pathData):
    data = pd.read_csv(pathData)
    data.drop(['Index'], axis=1, inplace=True)
    return data

# Main Function
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser(description='GraphPairPlot @Paris 42 School - Made by @abbensid & @bhamdi')
    parser.add_argument("input", type=str, help="The file containing datasets")
    args = parser.parse_args()
    try:
        print(f"[{Color.BLUE}Display Pair Plot{Color.END}]")
        data = preprocessing(args.input)
        axis = sns.pairplot(data, hue="Hogwarts House", height=1.5)
        plt.show()
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)