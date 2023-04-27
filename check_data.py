# 此文件用于检验现有分配方案的指标
# 导入模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

from restrictions import restrict_solution, restrict_solution_violation
from metrics import calculate_objective, calculate_crowding_distance, non_dominated_sorting
from reproduction import reprocude

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

# 读入旧计划方案，所属航系.xlsx第三个表
old_solution_raw = pd.read_excel("所属航系.xlsx", sheet_name=2, header=None)
old_solution = old_solution_raw.values.reshape(1, num_airline)
for idx, building_char in enumerate(['A', 'B', 'C']):
    old_solution[old_solution == building_char] = idx
old_solution = old_solution.astype(int)
old_solution_objective = calculate_objective(old_solution, airline_transport_num, airline_union2d, num_building, num_airline_union)


# 读入现行计划方案，原航站楼分配.xlsx第三个表
existing_solution_raw = pd.read_excel("原航站楼分配.xlsx", sheet_name=2, header=None)
existing_solution = np.ones((1, num_airline), dtype=int) * 2
# 转为list并删除空值
existing_solution_raw = existing_solution_raw.values.T.tolist()
existing_solution_raw = [list(filter(lambda x: not pd.isnull(x) == True, i)) for i in existing_solution_raw]
for building_idx, building_airline_list in enumerate(existing_solution_raw):
    existing_solution[0, [airline_list.index(airline) for airline in building_airline_list]] = building_idx
existing_solution_objective = calculate_objective(existing_solution, airline_transport_num, airline_union2d, num_building, num_airline_union)
# 按航系输出旧方案
for union_idx in range(num_airline_union):
    print("航系{}: {}".format(union_idx, airline_union[union_idx]))
    print("现方案: {}".format(existing_solution[0, airline_union_id == union_idx]))
    print()
print("\t人流量\t航系跨楼")
print("旧方案\t{}\t{}".format(old_solution_objective[0, 0], old_solution_objective[0, 1]))
print("现方案\t{}\t{}".format(existing_solution_objective[0, 0], existing_solution_objective[0, 1]))