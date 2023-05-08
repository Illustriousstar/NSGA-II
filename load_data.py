import pandas as pd
import numpy as np

# mode = "小算力"
mode = "大算力"

if mode == "小算力":
    airline_transport_num=pd.read_csv('airline_sum19.csv')
    airline_union_raw=pd.read_excel('所属航系.xlsx', sheet_name='小算力航系')
    num_building = 2
    num_airline = 19
else:
    airline_transport_num=pd.read_csv('airline_sum57.csv')
    airline_union_raw=pd.read_excel('所属航系.xlsx', sheet_name='大算力航系')
    num_building = 3
    num_airline = 57
airline_people_raw = pd.read_excel("高峰小时旅客运输量.xlsx", header=None)
airline_building_max=pd.read_excel('航站楼最大客流量.xlsx').values[:num_building,2].squeeze().astype(int)

# 航空公司之间转运的人流量
airline_transport_num["进港_出港"] = airline_transport_num["进港航司"] + "_" + airline_transport_num["出港航司"]
airline2num=dict(zip(airline_transport_num["进港_出港"],airline_transport_num["今年旅客人数"]))

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

# 按照（航系确定的）航空公司顺序计算转运人数
airline_transport_num = []
for i in range(num_airline):
    airline_transport_num.append([])
    for j in range(num_airline):
        airline_transport_num[i].append(airline2num.get(airline_list[i]+'_'+airline_list[j],0))
airline_transport_num = np.array(airline_transport_num,dtype=int)

# 航空公司高峰人流量
airline_people_dict = dict(zip(airline_people_raw[0], airline_people_raw[1]))
airline_people_num = np.array([airline_people_dict[airline] for airline in airline_list])