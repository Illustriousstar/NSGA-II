#  这是Kalyanmoy Deb教授流行的NSGA-II算法的python实现

# 导入模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

from restrictions import restrict_solution, restrict_solution_violation
from metrics import calculate_objective, calculate_crowding_distance, non_dominated_sorting
from reproduction import reprocude
from load_data import *

#################
# 参数与数据设置
#################

iter_max = 300 # 迭代次数
num_population = 2000 # 种群规模
prob_crossover = 0.9 # 交叉概率
prob_mutate = 0.1 # 变异概率


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
violation_func = partial(restrict_solution_violation, airline_people_num=airline_people_num, airline_building_max=airline_building_max, hard_restriction=True)
solution = init_solution(num_airline, num_building, num_population, restrict_func_hard)


for iter_num in range(iter_max):
    # 评价解的目标值
    objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
    # 产生下一代解
    solution_child = reprocude(solution, objective, 
                               prob_crossover, prob_mutate, 
                               num_building, 
                               restrict_func_hard, restrict_func_soft)
    # 将父代和子代合并
    solution = np.concatenate((solution, solution_child), axis=0)
    
    # 去重
    solution = np.unique(solution, axis=0)

    # 计算各项指标
    violation = violation_func(solution)
    objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
    crowding_distance = calculate_crowding_distance(objective)
    # 快速非支配排序
    pareto_layer = non_dominated_sorting(violation, objective)
    features = np.stack((-crowding_distance, pareto_layer), axis=1)
    solution_index = np.lexsort(features.transpose())
    # 按排序结果取出解
    solution = solution[solution_index[:num_population]]

    # 选择符合限制的解
    solution_ = solution[restrict_func_hard(solution)]
    objective_ = calculate_objective(solution_, airline_transport_num, airline_union2d, num_building, num_airline_union)
    # 计算最优评价值和平均评价值
    objective_min = np.min(objective_, axis=0)
    objective_mean = np.mean(objective_, axis=0)
    objective_max = np.max(objective_, axis=0)
    objective_data = np.stack((objective_min, objective_mean, objective_max))
    objective_data = np.transpose(objective_data).astype(int)
    print('iter_num: %d, num_legal_pop: %s, objectives:\n %s' % (iter_num, len(solution_), objective_data))

# 迭代结束，输出最优解
# 按评价排序
violation = violation_func(solution)
objective = calculate_objective(solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
pareto_layer = non_dominated_sorting(violation, objective)
solution_index = np.argsort(pareto_layer)
samples = solution_index[:300]
# 画评价函数的散点图
plt.scatter(objective[samples, 0], objective[samples, 1])
plt.show()
# 保存结果
np.savetxt("小算力解集.csv" if mode=="小算力" else "大算力解集.csv", solution, delimiter=',', fmt='%d')

