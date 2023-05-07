# 检查NSGA-II生成的方案
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import calculate_objective, non_dominated_sorting
from restrictions import restrict_solution, restrict_solution_violation
airline_transport_num=pd.read_csv('test.csv',header=None)
airline_people_raw = pd.read_excel("高峰小时旅客运输量.xlsx", header=None)
airline_union_raw=pd.read_excel('所属航系.xlsx')
airline_building_max=pd.read_excel('航站楼最大客流量.xlsx').values[:,2].squeeze().astype(int)
num_building = len(airline_building_max) # 航站楼数量
num_airline = len(airline_people_raw) # 航空公司数量

# 航空公司之间转运的人流量
airline2num=dict(zip(airline_transport_num[0],airline_transport_num[1]))

# 航系对应的航空公司
airline_union=airline_union_raw.values.transpose()
airline_union=[list(filter(lambda x: not pd.isnull(x) == True, i)) for i in airline_union]
num_airline_union = len(airline_union) # 航系数量
# 获取全部航空公司名，可根据顺序判断所属航系
airline_list = [airline for union in airline_union for airline in union]
# 根据航系的航空公司数量，按顺序分配航系id
airline_union_count = [len(union) for union in airline_union]
airline_union_id = np.zeros(num_airline, dtype=int)
cumsum = 0
for i, count in enumerate(airline_union_count):
    airline_union_id[cumsum:cumsum+count] = i
    cumsum += count
airline_union2d = (airline_union_id.reshape(-1, 1) == airline_union_id.reshape(1, -1)).astype(int)

airline_transport_num = []
for i in range(num_airline):
    airline_transport_num.append([])
    for j in range(num_airline):
        airline_transport_num[i].append(airline2num.get(airline_list[i]+'_'+airline_list[j],0))
airline_transport_num = np.array(airline_transport_num,dtype=int)

# 航空公司高峰人流量
airline_people_dict = dict(zip(airline_people_raw[0], airline_people_raw[1]))
airline_people_num = np.array([airline_people_dict[airline] for airline in airline_list])


# 读取NSGA-II生成的方案, solution.csv
solution = np.loadtxt('solution.csv', dtype=int, delimiter=',')
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
# 绘制Pareto前沿
solution = solution[pareto_layer < layers]
objective = objective[pareto_layer < layers]
pareto_layer = pareto_layer[pareto_layer < layers]
plt.figure()
plt.scatter(objective[:, 0], objective[:, 1], c=pareto_layer, cmap='rainbow')
plt.xlabel('people number')
plt.ylabel('union cross building number')
plt.title('Pareto Front')
plt.show()