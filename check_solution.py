# 检查NSGA-II生成的方案
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import calculate_objective, non_dominated_sorting
from restrictions import restrict_solution, restrict_solution_violation
from topsis import topsis
from load_data import *


# 读取NSGA-II生成的方案, solution.csv
solution = np.loadtxt("小算力解集.csv" if mode=="小算力" else "大算力解集.csv", dtype=int, delimiter=',')
print(f"方案数量: {len(solution)}")
# 选择满足约束条件的方案
solution = np.unique(solution, axis=0)
solution = solution[restrict_solution(solution, airline_people_num, airline_building_max, hard_restriction=True)]
print(f"满足约束条件的方案数量: {len(solution)}")
# 非支配排序
violations = restrict_solution_violation(solution, airline_people_num, airline_building_max)
objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
pareto_layer = non_dominated_sorting(violations, objective)

layers = 1
solution = solution[pareto_layer < layers]
objective = objective[pareto_layer < layers]
pareto_layer = pareto_layer[pareto_layer < layers]

# 根据指标排序
sort_index = np.argsort(objective[:, 0])
solution = solution[sort_index]
objective = objective[sort_index]
pareto_layer = pareto_layer[sort_index]

# 绘制Pareto前沿
plt.figure()
plt.scatter(objective[:, 0], objective[:, 1], c=pareto_layer, cmap='rainbow')
plt.xlabel('people number')
plt.ylabel('union cross building number')
plt.title('Pareto Front')
plt.show()

# 计算TOPSIS得分
objective_norm = objective / np.linalg.norm(objective, axis=0)
# 把012方案转化为ABC
solution = np.array([['A', 'B', 'C'][i] for i in solution.flatten()]).reshape(solution.shape)
objective_under_weight = []
for weight in np.linspace(0, 1, 101):
    weights = np.array([weight, 1-weight])
    Result, Z, weights_ = topsis(objective, weights)
    objective_ = Result.iloc[0, :2].to_numpy().astype(int)
    objective_under_weight.append(objective_)
objective_under_weight = np.array(objective_under_weight)
weight = np.array([0.2, 0.8])
Result, Z, weight = topsis(objective, weight)
# 保存到excel
writer = pd.ExcelWriter(f"综合评价结果_{mode}.xlsx")
pd.DataFrame(objective, columns=['中转人数', '同航系跨楼数']).to_excel(writer, sheet_name="目标函数值",index=False)
pd.DataFrame(objective_norm, columns=['中转人数', '同航系跨楼数']).to_excel(writer, sheet_name="归一化目标函数值",index=False)
pd.DataFrame(np.concatenate([objective, solution], axis=1)).to_excel(writer, sheet_name="目标函数值与方案",index=False, header=['中转人数', '同航系跨楼数'] + airline_list)
pd.DataFrame(objective_under_weight, columns=['中转人数', '同航系跨楼数']).to_excel(writer, sheet_name="权重组合下目标函数值",index=False)
pd.DataFrame(Result).to_excel(writer, sheet_name="TOPSIS 0.2权重",index=False)
writer.close()