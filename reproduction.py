# 产生下一代的操作：交叉、变异、选择
import numpy as np

# 交叉算子，两点交叉算子
def crossover(solution:np.array, prob_crossover:float, restriction_func=None, max_iter=100):
    assert len(solution.shape) == 2, 'solution must be 2D array'
    num_chromo, num_airline = solution.shape
    assert num_chromo >= 2, 'num_chromo must be greater than 2'
    
    solution_out = np.zeros((0, num_airline), dtype=int)

    # 复制解集，用于存储交叉后的解
    solution = solution.copy()
    # 打乱顺序
    np.random.shuffle(solution)
    # 选取偶数个解，用于交叉
    if num_chromo % 2 == 1:
        solution = solution[:-1, :]
    # [配对数, 2, 航空公司数]
    solution = solution.reshape(-1, 2, num_airline)

    # 按概率决定每对是否交叉
    chromo_to_crossover = np.random.rand(num_chromo // 2) < prob_crossover
    # 不交叉的直接加入输出解集
    solution_out = np.concatenate((solution_out, solution[~chromo_to_crossover, :, :].reshape(-1, num_airline)), axis=0)
    # 交叉的加入输入解集
    solution = solution[chromo_to_crossover, :, :]
    # 循环确保一定交叉
    iter = 0
    while iter < max_iter and solution.shape[0] != 0:
        # 随机选择点位
        index = np.random.randint(num_airline, size=(solution.shape[0], 2))
        # 保证点位递增
        index.sort(axis=1)
        # 区间有效才算成功交叉
        crossover_state = (index[:, 0] != index[:, 1])
        # 逐个基因决定是否交叉
        gene_id = np.arange(num_airline).reshape(1, -1).repeat(solution.shape[0], axis=0)
        gene_to_crossover = (gene_id >= index[:, 0].reshape(-1, 1)) & (gene_id < index[:, 1].reshape(-1, 1))
        # 交叉
        solution1 = np.where(gene_to_crossover, solution[:, 1, :], solution[:, 0, :])
        solution2 = np.where(gene_to_crossover, solution[:, 0, :], solution[:, 1, :])
        # 检查限制，一对都交叉成功才可以
        if restriction_func is not None:
            result1 = restriction_func(solution1)
            result2 = restriction_func(solution2)
            result = result1 & result2
            crossover_state = crossover_state & result
            solution1 = solution1[result, :]
            solution2 = solution2[result, :]
        # 将交叉后的解加入输出解集
        solution_out = np.concatenate((solution_out, solution1, solution2), axis=0)
        # 从输入解集中删除已经交叉的解
        solution = solution[~crossover_state, :, :]
        iter += 1
    return solution_out

# 变异算子，基本位变异算子
def mutate(solution:np.array, prob_mutation:float, num_building:int, restriction_func=None, max_iter=100):
    assert len(solution.shape) == 2, 'solution must be 2D array'
    num_chromo, num_airline = solution.shape
    assert num_chromo >= 2, 'num_chromo must be greater than 2'
    

    solution_out = np.zeros((0, num_airline), dtype=int)

    # 按概率决定每个染色体是否变异
    chromo_to_mutate = np.random.rand(num_chromo) < prob_mutation
    # 不变异的直接加入输出解集
    solution_out = np.concatenate((solution_out, solution[~chromo_to_mutate, :]), axis=0)
    # 变异的加入输入解集
    solution = solution[chromo_to_mutate, :]
    iter = 0
    while iter < max_iter and solution.shape[0] != 0:
        # 每个染色体随机选择一个基因变异
        gene_id = np.random.randint(num_airline, size=solution.shape[0])
        # 变异后不能与原基因相同
        gene_value = np.random.randint(num_building, size=solution.shape[0])
        gene_value = np.where(gene_value == solution[np.arange(solution.shape[0]), gene_id], (gene_value + 1)%num_building, gene_value)
        # 变异
        solution_mutate = solution.copy()
        solution_mutate[np.arange(solution.shape[0]), gene_id] = gene_value
        # 检查限制
        result = np.ones(solution.shape[0], dtype=bool)
        if restriction_func is not None:
            result = restriction_func(solution_mutate)
        solution_mutate = solution_mutate[result, :]
        # 将变异后的解加入输出解集
        solution_out = np.concatenate((solution_out, solution_mutate), axis=0)
        # 从输入解集中删除已经变异的解
        solution = solution[~result, :]
    return solution_out

# 选择种群，二元锦标赛选择算子
def binary_tournament_selection(solution:np.array, objective:np.array, num_chromo_out:int, restriction_func):
    assert len(solution.shape) == 2, 'solution must be 2D array'
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_airline = solution.shape
    assert num_chromo >= 2, 'num_chromo must be greater than 2'
    assert num_chromo >= num_chromo_out, 'num_chromo must be greater than num_chromo_out'

    # 选择种群，批量处理
    solution_out = np.zeros((0, num_airline), dtype=int)
    # 选择两批解
    indices = np.random.choice(num_chromo, size=(2, num_chromo), replace=True)
    # 比较两批解
    # 符合限制条件的解
    solution1 = solution[indices[0], :]
    solution2 = solution[indices[1], :]
    result1 = restriction_func(solution1).astype(int)
    result2 = restriction_func(solution2).astype(int)
    restriction_priority_high = result1 > result2
    restriction_priority_equal = result1 == result2
    # Pareto层级更优，所有目标都小于对方
    objective_less = (objective[indices[0]] < objective[indices[1]]).all(axis=1)
    # 选择较好的解
    index = np.where(restriction_priority_high, indices[0], indices[1])
    index = np.where(restriction_priority_equal & objective_less, indices[0], index)
    # 将较好的解加入解集
    solution_out = np.concatenate((solution_out, solution[index]), axis=0)

    return solution_out

# 选择种群，二元锦标赛选择算子
def selection(solution:np.array, objective:np.array, crowding_distance:np.array, num_chromo_out:int, restriction_func):
    assert len(solution.shape) == 2, 'solution must be 2D array'
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_airline = solution.shape
    assert num_chromo >= 2, 'num_chromo must be greater than 2'
    assert num_chromo >= num_chromo_out, 'num_chromo must be greater than num_chromo_out'

    # 选择种群，批量处理
    solution_out = np.zeros((0, num_airline), dtype=int)
    while solution_out.shape[0] < num_chromo_out:
        num_chromo = solution.shape[0]
        # 选择两批解
        indices = np.random.choice(num_chromo, size=(2, num_chromo // 2), replace=False)
        # 比较两批解
        # 符合限制条件的解
        solution1 = solution[indices[0], :]
        solution2 = solution[indices[1], :]
        result1 = restriction_func(solution1).astype(int)
        result2 = restriction_func(solution2).astype(int)
        restriction_priority_high = result1 > result2
        restriction_priority_equal = result1 == result2
        # 所有目标都小于对方
        objective_less = (objective[indices[0]] < objective[indices[1]]).all(axis=1)
        # 有目标小于对方
        objective_cross = (objective[indices[0]] < objective[indices[1]]).any(axis=1)
        crowding_distance_greater = crowding_distance[indices[0]] > crowding_distance[indices[1]]
        # 选择较好的解
        index = np.where(restriction_priority_high, indices[0], indices[1])
        index = np.where(restriction_priority_equal & objective_less, indices[0], index)
        index = np.where(restriction_priority_equal & objective_cross & crowding_distance_greater, indices[0], index)
        # 将较好的解加入解集
        solution_out = np.concatenate((solution_out, solution[index]), axis=0)
        # 去重
        # 实际上不用去重，因为每次只选择两批解，不会出现重复的解
        solution_out = np.unique(solution_out, axis=0)
        solution = np.delete(solution, index, axis=0)
    # 取出指定数量的解
    solution_out = solution_out[:num_chromo_out]
    return solution_out

def reprocude(solution:np.array, objective:np.array, 
              prob_crossover:float, prob_mutation:float,
              num_building:int,
              restriction_func_hard, restriction_func_soft):
    assert len(solution.shape) == 2, 'solution must be 2D array'
    assert len(objective.shape) == 2, 'objective must be 2D array'
    num_chromo, num_airline = solution.shape

    # 选择种群，产生子代
    solution = binary_tournament_selection(solution, objective, num_chromo, restriction_func_hard)
    # 交叉
    solution = crossover(solution, prob_crossover, restriction_func_hard)
    # 变异
    solution = mutate(solution, prob_mutation, num_building, restriction_func_soft)

    return solution