#  这是Kalyanmoy Deb教授流行的NSGA-II算法的python实现

# 导入模块
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import partial

from restrictions import restrict_solution
from metrics import calculate_objective, calculate_crowding_distance
from reproduction import crossover, mutate, selection

#################
# 参数与数据设置
#################

iter_max = 150
num_population = 10000
prob_crossover = 0.9
prob_mutate = 0.1

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
        airline_transport_num[i].append(airline2num.get(airline_list[i]+'_'+airline_list[j],0) / 2.)
airline_transport_num = np.array(airline_transport_num,dtype=int)

# 航空公司高峰人流量
airline_people_dict = dict(zip(airline_people_raw[0], airline_people_raw[1]))
airline_people_num = np.array([airline_people_dict[airline] for airline in airline_list])

# 生成初始解，[]中的值为航站楼的编号
def init_solution(num_airline:int, num_building:int, num_population:int=1, restriction_func=None):
    # 创建空的解集
    solution = np.zeros((0, num_airline), dtype=int)
    # 生成初始解
    while solution.shape[0] < num_population:
        # 生成一批解
        proposal = np.random.randint(0, num_building, size=(num_population, num_airline))
        # 检查约束条件
        if restriction_func is not None:
            proposal = proposal[restriction_func(proposal)]
        # 将可行解加入解集
        solution = np.concatenate((solution, proposal), axis=0)
        # 去重
        solution = np.unique(solution, axis=0)
    # 取出指定数量的解
    solution = solution[:num_population]
    return solution
    
# 限制函数使用偏函数，将限制函数的参数固定
restrict_func_hard = partial(restrict_solution, airline_people_num=airline_people_num, airline_building_max=airline_building_max, hard_restriction=True)
restrict_func_soft = partial(restrict_solution, airline_people_num=airline_people_num, airline_building_max=airline_building_max, hard_restriction=False)

solution = init_solution(num_airline, num_building, num_population, restrict_func_hard)

graph_data=pd.DataFrame(columns=['people_num','airline'])
for iter_num in range(iter_max):
    # 产生下一代解
    solution_crossover = crossover(solution, prob_crossover, restrict_func_hard)
    solultion_mutate = mutate(solution, prob_mutate, num_building, restrict_func_soft)
    solution = np.concatenate((solution, solution_crossover, solultion_mutate), axis=0)
    # 评价解的目标值
    objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
    # 计算拥挤度
    crowding_distance = calculate_crowding_distance(objective)
    # 选择
    solution = selection(solution, objective, crowding_distance, num_population)

    # 计算最优评价值和平均评价值
    objective_min = np.min(objective, axis=0)
    objective_mean = np.mean(objective, axis=0)
    objective_max = np.max(objective, axis=0)
    graph_data.loc[len(graph_data)]=objective_min
    print('iter_num: %d, objective_min: %s, objective_mean: %s, objective_max: %s' % (iter_num, objective_min, objective_mean, objective_max))

# 迭代结束，输出最优解
# 按评价排序
objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
index = np.lexsort(objective.T)
solution = solution[index]
objective = objective[index]
# plt.con
plt.xlabel('people_num')
plt.ylabel('airline')
plt.scatter(graph_data['people_num'],graph_data['airline'])
plt.show()
# 保存结果
np.savetxt('solution.csv', solution, delimiter=',', fmt='%d')

