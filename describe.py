import pandas as pd
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from IPython.display import display

# pd.set_option('display.max_columns', 500)

class Color:
    BLUE = '\x1b[96m'
    WARNING = '\x1b[93m'
    GREEN = '\x1b[92m'
    END = '\x1b[0m'

def ft_sum(Serie):
    result = 0.0
    for elem in Serie:
        result += elem
    return result

def ft_count(Serie):
	return Serie[Serie.notnull()].size

def ft_mean(Serie):
    result = ft_sum(Serie)
    size = Serie.size
    result = result/size if size else 'NaN'
    return result

def ft_min(Serie):
	result = Serie.iat[0] if Serie.size else 'NaN'
	for elem in Serie:
		result = elem if result > elem else result
	return result

def ft_max(Serie):
	result = Serie.iat[0] if Serie.size else 'NaN'
	for elem in Serie:
		result = elem if result < elem else result
	return result

def ft_std(Serie):
    mean = ft_mean(Serie)
    result = 0.0
    for elem in Serie:
        result += (elem - mean)**2
    if Serie.size > 1:
        result = np.sqrt(result/(Serie.size - 1))
    else:
        result = 'NaN'
    return result

def ft_quartiles(Serie, p):
    Serie = Serie.sort_values(ignore_index=True)
    if Serie.size == 0: return 'NaN'
    k = (len(Serie) - 1) * (p / 100)
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return Serie[int(k)]
    d0 = Serie[int(f)] * (c - k)
    d1 = Serie[int(c)] * (k - f)
    return (d0 + d1)

def ft_unique(Serie):
	tmp = []
	for elem in Serie:
		if elem not in tmp:
			tmp.append(elem)
	result = len(tmp) if len(tmp) > 0 else 'NaN'
	return result

def	ft_top(Group):
	if not Group.size:
		result = 'NaN'
	result = Group.size().sort_values(ascending=False).index[0]
	return result

def	ft_freq(Group):
	if not Group.size:
		result = 'NaN'
	result = Group.size().sort_values(ascending=False).values[0]
	return result

def ft_describe(data, inc=''):
	if inc == 'all':
		index = { 0:'count', 1:'unique', 2:'top', 3:'freq', 4:'mean', 5:'std', 6:'min', 7:'25%', 8:'50%', 9:'75%', 10:'max'}
	else:
		index = { 0:'count', 1:'mean', 2:'std', 3:'min', 4:'25%', 5:'50%', 6:'75%', 7:'max'}
		data = data.select_dtypes('number')

	describe_result = pd.DataFrame()
	dict_columns = {}
	key_i = 0
	for value in data.columns:
		nan = 0
		tmp_column = []
		Serie = data[value][data[value].notna()]
		nan = 'NaN' if not is_numeric_dtype(Serie) else 'num'
		tmp_column.append(ft_count(Serie))
		if (inc == 'all'):
			tmp_column.append('NaN' if nan == 'num' else ft_unique(Serie))
			tmp_column.append(ft_top(data.groupby([value], sort=False)) if nan != 'num' else 'NaN')
			tmp_column.append(ft_freq(data.groupby([value], sort=False)) if nan != 'num' else 'NaN')
		if (data.select_dtypes(include='number')).size:
			tmp_column.append(nan if nan == 'NaN' else ft_mean(Serie))
			tmp_column.append(nan if nan == 'NaN' else ft_std(Serie))
			tmp_column.append(nan if nan == 'NaN' else ft_min(Serie))
			tmp_column.append(nan if nan == 'NaN' else ft_quartiles(Serie, 25))
			tmp_column.append(nan if nan == 'NaN' else ft_quartiles(Serie, 50))
			tmp_column.append(nan if nan == 'NaN' else ft_quartiles(Serie, 75))
			tmp_column.append(nan if nan == 'NaN' else ft_max(Serie))
		tmp_serie = pd.Series(tmp_column)
		describe_result = pd.concat([describe_result, tmp_serie], axis=1, ignore_index=True)
		dict_columns[key_i] = value
		key_i += 1
	describe_result = describe_result.rename(columns=dict_columns, index=index)
	return describe_result

# Main Function
if __name__ == "__main__":
    # ArgumentParser
	parser = argparse.ArgumentParser(description='DescribePandas @Paris 42 School - Made by @abbensid & @bhamdi')
	parser.add_argument("input", type=str, help="The file containing dataset")
	parser.add_argument("-a", "--all", action="store_true", help="describe(include='all')")
	args = parser.parse_args()
	try:
		data = pd.read_csv(args.input, index_col=False)
		if (args.all):
			print(ft_describe(data, 'all'))
		else:
			print(ft_describe(data))
	except (Exception, BaseException) as e:
		print(f"{Color.WARNING} {e} {Color.END}")
		sys.exit(1)