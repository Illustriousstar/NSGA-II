# 描述对于可行解的限制条件
import numpy as np

#################
# 约束函数设置
#################
# 第一个约束函数，限制航站楼流量
def restriction1(chromo:np.array, airline_people_num:np.array, airline_building_max:np.array):
    assert len(chromo.shape) == 2, 'chromo must be 2D array'
    num_chromo, num_airline = chromo.shape
    num_building = len(airline_building_max)
    # 利用numpy广播批量处理
    # [解的数目, 航站楼数目, 航空公司数目]
    building_id = np.arange(num_building).reshape(1, num_building, 1)
    airline_in_building = (chromo.reshape(num_chromo, 1, num_airline) == building_id).astype(int)
    airline_people_num = airline_people_num.reshape(1, 1, num_airline)
    people_in_building = np.sum(airline_in_building * airline_people_num, axis=2)
    result = (people_in_building <= airline_building_max.reshape(1, num_building)).sum(axis=1)==3

    # 计算违反约束的误差
    violation = people_in_building - airline_building_max.reshape(1, num_building)
    violation = violation * (violation > 0)
    violation = violation.sum(axis=1)

    # 返回每一个解是否可行，可行为True，不可行为False
    # [解的数目]
    return result, violation

# 第二个约束函数，保证航空公司分配航站楼且唯一
# 不用做，天然就对
def restriction2(chromo:np.array):
    assert len(chromo.shape) == 2, 'chromo must be 2D array'
    # return true for each chromo
    return np.ones(chromo.shape[0], dtype=bool)

# 第三个约束函数，航站楼不为空
def restriction3(chromo:np.array, num_building:int):
    assert len(chromo.shape) == 2, 'chromo must be 2D array'
    num_chromo, num_airline = chromo.shape
    # 利用numpy广播批量处理
    # [解的数目, 航站楼数目, 航空公司数目]
    building_id = np.arange(num_building).reshape(1, num_building, 1)
    airline_in_building = (chromo.reshape(num_chromo, 1, num_airline) == building_id).astype(int)
    result = (airline_in_building.sum(axis=2) > 0).sum(axis=1)==3
    # 返回每一个解是否可行，可行为True，不可行为False
    # [解的数目]
    return result

# 第四个约束函数，变量之间的约束
# 不用做，天然就对
def restriction4(chromo:np.array):
    assert len(chromo.shape) == 2, 'chromo must be 2D array'
    # return true for each chromo
    return np.ones(chromo.shape[0], dtype=bool)

def restrict_solution(solution:np.array, airline_people_num:np.array, airline_building_max:np.array,
                      hard_restriction:bool=True):
    num_building = len(airline_building_max)
    if hard_restriction:
        result1, violation1 = restriction1(solution, airline_people_num, airline_building_max)
    else:
        result1 = np.ones(solution.shape[0], dtype=bool)
    result3 = restriction3(solution, num_building)
    result = result1 & result3
    return result

def restrict_solution_violation(solution:np.array, airline_people_num:np.array, airline_building_max:np.array,
                                hard_restriction:bool=True):
    num_building = len(airline_building_max)
    if hard_restriction:
        result1, violation1 = restriction1(solution, airline_people_num, airline_building_max)
    else:
        result1 = np.ones(solution.shape[0], dtype=bool)
        violation1 = np.zeros(solution.shape[0], dtype=int)
    result3 = restriction3(solution, num_building)
    result = result1 & result3
    return violation1

