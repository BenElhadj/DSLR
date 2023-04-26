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


# calcul correcltation
# data.drop(['Index'], axis=1, inplace=True)
# print(data.corr(numeric_only=True).abs().unstack().sort_values(ascending = False)[:20])


# Main Function
if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser(description='GraphScatterPlot @Paris 42 School - Made by @abbensid & @bhamdi')
    parser.add_argument("input", type=str, help="The file containing dataset")
    args = parser.parse_args()
    try:
        print(f"[{Color.BLUE}Scatter Plot{Color.END}]")
        data = pd.read_csv(args.input)
        plt.figure('Scatter Plot')
        axis = sns.scatterplot(data=data, x="Astronomy", y="Defense Against the Dark Arts", hue='Hogwarts House')
        axis.set_title('Correltation Enter Astronomy & Defense Against the Dark Arts')
        plt.show()
    except (Exception, BaseException) as e:
        print(f"{Color.WARNING} {e} {Color.END}")
        sys.exit(1)