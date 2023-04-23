import numpy as np
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
