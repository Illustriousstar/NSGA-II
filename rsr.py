# RSR 模型的 Python 实现
# 来自：https://zhuanlan.zhihu.com/p/38209882

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


def rsr(data, weight=None, threshold=None, full_rank=True):
	Result = pd.DataFrame()
	n, m = data.shape

	# 对原始数据编秩
	if full_rank:
		for i, X in enumerate(data.columns):
			Result[f'X{str(i + 1)}: {X}'] = data.iloc[:, i]
			Result[f'R{str(i + 1)}: {X}'] = data.iloc[:, i].rank(method="dense", ascending=False)
	else:
		for i, X in enumerate(data.columns):
			Result[f'X{str(i + 1)}: {X}'] = data.iloc[:, i]
			Result[f'R{str(i + 1)}: {X}'] = 1 + (n - 1) * (data.iloc[:, i].max() - data.iloc[:, i]) / (data.iloc[:, i].max() - data.iloc[:, i].min())

	# 计算秩和比
	weight = 1 / m if weight is None else np.array(weight) / sum(weight)
	Result['RSR'] = (Result.iloc[:, 1::2] * weight).sum(axis=1) / n
	Result['RSR_Rank'] = Result['RSR'].rank(ascending=False)

	# 绘制 RSR 分布表
	RSR = Result['RSR']
	RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))
	Distribution = pd.DataFrame(index=sorted(RSR.unique()))
	Distribution['f'] = RSR.value_counts().sort_index()
	Distribution['Σ f'] = Distribution['f'].cumsum()
	Distribution[r'\bar{R} f'] = [RSR_RANK_DICT[i] for i in Distribution.index]
	Distribution[r'\bar{R}/n*100%'] = Distribution[r'\bar{R} f'] / n
	Distribution.iat[-1, -1] = 1 - 1 / (4 * n)
	Distribution['Probit'] = 5 - norm.isf(Distribution.iloc[:, -1])

	# 计算回归方差并进行回归分析
	r0 = np.polyfit(Distribution['Probit'], Distribution.index, deg=1)
	print(sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit'])).fit().summary())
	if r0[1] > 0:
		print(f"\n回归直线方程为: y = {r0[0]} Probit + {r0[1]}")
	else:
		print(f"\n回归直线方程为: y = {r0[0]} Probit - {abs(r0[1])}")

	# 代入回归方程并分档排序
	Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit'])
	Result['RSR Regression'] = np.polyval(r0, Result['Probit'])
	threshold = np.polyval(r0, [2, 4, 6, 8]) if threshold is None else np.polyval(r0, threshold)
	Result['Level'] = pd.cut(Result['RSR Regression'], threshold, labels=range(len(threshold) - 1, 0, -1))

	return Result, Distribution


def rsrAnalysis(data, file_name=None, **kwargs):
	Result, Distribution = rsr(data, **kwargs)
	file_name = 'RSR 分析结果报告.xlsx' if file_name is None else file_name + '.xlsx'
	Excel_Writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
	Result.to_excel(Excel_Writer, '综合评价结果')
	Result.sort_values(by='Level', ascending=False).to_excel(Excel_Writer, '分档排序结果')
	Distribution.to_excel(Excel_Writer, 'RSR分布表')
	# Excel_Writer.save()
	Excel_Writer.close()

	return Result, Distribution

