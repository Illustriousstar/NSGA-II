import pandas as pd
import numpy as np


def topsis(data, weight=None):
	# 归一化
	data_original = pd.DataFrame(data.copy())
	data = pd.DataFrame(data)
	data = data / np.sqrt((data ** 2).sum(axis=0))

	# 最优最劣方案
	Z = pd.DataFrame([data.max(axis=0), data.min(axis=0)], index=['负理想解', '正理想解'])

	# 距离
	weight = entropyWeight(data) if weight is None else np.array(weight)
	Result = data_original
	Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
	Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

	# 综合得分指数
	Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
	Result['排序'] = Result.rank(ascending=False)['综合得分指数']
	# 排序
	Result = Result.sort_values(by='排序')

	return Result, Z, weight

def entropyWeight(data):
	data = np.array(data)
	# 归一化
	P = data / data.sum(axis=0)

	# 计算熵值
	E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

	# 计算权系数
	return (1 - E) / (1 - E).sum()
