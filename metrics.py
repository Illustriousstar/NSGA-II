import numpy as np
from functools import cmp_to_key
# 计算优化目标：航空公司的跨航站楼，航系的跨航站楼
def calculate_objective(chromo:np.array, airline_transport_num:np.array, airline_union2d:np.array, num_building:int, num_airline_union:int):
    assert len(chromo.shape) == 2, 'chromo must be 2D array'
    num_chromo, num_airline = chromo.shape

    # 1.计算航空公司的跨航站楼
    # [解的数目, 航空公司数目, 航空公司数目]
    solution_airline_transport_state = (chromo.reshape(num_chromo, 1, num_airline) != chromo.reshape(num_chromo, num_airline, 1)).astype(int)
    solution_airline_transport_num = solution_airline_transport_state * airline_transport_num
    solution_airline_transport_num = solution_airline_transport_num.sum(axis=(1,2))

    # 2.计算航系内部的跨航站楼数
    # 计算任意两个航空公司的跨楼数目
    sloution_airline_transport_building_num = solution_airline_transport_state
    # 按照航系内部跨楼数
    sloution_airline_transport_building_within_union_num = sloution_airline_transport_building_num * airline_union2d
    sloution_airline_transport_building_within_union_num = sloution_airline_transport_building_within_union_num.sum(axis=(1,2))

    # 计算结果合并
    objective = np.stack((solution_airline_transport_num, sloution_airline_transport_building_within_union_num), axis=1)
    
    return objective

# 计算拥挤度
def calculate_crowding_distance(objective:np.array):
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_objective = objective.shape
    # 计算拥挤度
    crowding_distance = np.zeros((num_chromo, num_objective))
    for i in range(num_objective):
        # 对每个目标排序
        indices = np.argsort(objective[:, i])
        # 计算拥挤度
        crowding_distance[indices[0], i] = np.inf
        crowding_distance[indices[-1], i] = np.inf
        crowding_distance[indices[1:-1], i] = (objective[indices[2:], i] - objective[indices[:-2], i]) / (objective[indices[-1], i] - objective[indices[0], i])
    # 计算总拥挤度
    crowding_distance = crowding_distance.sum(axis=1)
    return crowding_distance



# 非支配排序比较函数，优先级为
# 1.符合限制或违约次数少的
# 2.Pareto层级低的
# 3.拥挤度高的
def non_dominated_comparator(a:np.array, b:np.array):
    # (a, b)是两个特征向量
    # [违约程度，拥挤度，[目标函数]]

    # 一个符合约束而另一个不符合时，取符合的那个
    # 两个都不符合时，取违约小的那个
    if a[0] != b[0]:
        if a[0] < b[0]:
            return 1
        else: 
            return -1
    # 同时符合约束，或违约程度相同
    # a[0] == b[0]
    else:
        # 检查目标函数
        objective_less = (a[2:] <= b[2:]).all()
        objective_equal = (a[2:] < b[2:]).any()
        crowding_distance_greater = a[1] > b[1]
        # 目标函数Pareto层级更低
        if objective_less:
            return 1
        elif objective_equal and crowding_distance_greater:
            return 1
        elif objective_equal and not crowding_distance_greater:
            return -1
        else:
            # objective_greater
            return -1

# 非支配排序，优先级为
# 1.符合限制或违约次数少的
# 2.Pareto层级低的
def non_dominated_sorting(violation:np.array, objective:np.array):
    assert len(violation.shape) == 1, 'violation must be 1D array'
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_objective = objective.shape

    # 自定义比较函数，依据优先级
    violation_less = (violation.reshape(num_chromo, 1) < violation.reshape(1, num_chromo))
    violation_equal = (violation.reshape(num_chromo, 1) == violation.reshape(1, num_chromo))
    objective_less = (objective.reshape(num_chromo, 1, num_objective) <= objective.reshape(1, num_chromo, num_objective)).all(axis=2)
    objective_same = (objective.reshape(num_chromo, 1, num_objective) == objective.reshape(1, num_chromo, num_objective)).all(axis=2)
    # 先置为优先级低
    compare_matrix = np.zeros((num_chromo, num_chromo), dtype=bool)
    # 违约程度低的优先级高
    compare_matrix[violation_less] = True
    # 违约程度相同，Pareto层级低的优先级高（包括自己，注意排除）
    compare_matrix[violation_equal & objective_less] = True
    # 违约程度相同，Pareto层级相同，目标函数相同，取消优先级
    compare_matrix[violation_equal & objective_same] = False
    # compare_matrix[violation_equal & objective_greater] = False
    # compare_matrix[violation_greater] = False
    # 创建输入索引和输出排序索引
    index_input = np.arange(num_chromo)
    pareto_layer = np.zeros(num_chromo, dtype=int)
    # 循环排序
    cur_pareto_layer = 0
    while index_input.size > 0:
        # 优先级不被别人支配的
        pareto_layer_map = (~compare_matrix[index_input][:, index_input]).all(axis=0)
        # 输出索引
        index_output = index_input[pareto_layer_map]
        # 从输入索引中删除
        index_input = index_input[~pareto_layer_map]
        # 指定Pareto层级
        pareto_layer[index_output] = cur_pareto_layer
        # 下一层级
        cur_pareto_layer += 1

    # 比较矩阵转换为排序索引
    # index = np.argsort(pareto_layer)
    return pareto_layer

# 非支配排序比较函数，只比较Pareto层级，不考虑拥挤度和违约程度
def non_dominated_comparator_pareto(a:np.array, b:np.array):
    # (a, b)是两个特征向量objective=
    objective_less = (a <= b).all()
    objective_equal = (a < b).any()

    # 目标函数Pareto层级更低
    if objective_less:
        return 1
    elif objective_equal:
        return 0
    else:
        return -1


# 非支配排序，只比较Pareto层级，不考虑拥挤度和违约程度
def non_dominated_sorting_pareto(objective:np.array):
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_objective = objective.shape
    features = objective
    # 自定义比较函数，依据优先级
    keys = np.apply_along_axis(cmp_to_key(non_dominated_comparator_pareto), axis=1, arr=features)
    index = np.argsort(keys)[::-1]
    return index
