#!/usr/bin/env python
# coding: utf-8

# In[3]:


def check_normal_volume(p1_p,p1_v,p2_p,p2_v,p3_p,p3_v):
        volume = min(p1_v,p3_v)
        volume = min(p2_p/p1_p*p2_v,volume)
        print(volume)
        return volume*0.5


# In[10]:


def profit_calculation(array1,array2,array3):
    base_sale_p = array1[0] #ethusd
    base_sale_v = array1[1]
    base_buy_p = array1[2] #ethusd
    base_buy_v = array1[3]
    
    change_sale_p = array2[0] #btcusd
    change_sale_v = array2[1]
    change_buy_p = array2[2] #btcusd
    change_buy_v = array2[3]
    #自身也是可能会赔钱 - 本身是有波动的
    market_sale_p = array3[0] #ethbtc
    market_sale_v = array3[1]
    market_buy_p = array3[2] #ethbtc
    market_buy_v = array3[3]
    difference = (base_buy_p/change_sale_p - market_sale_p)/market_sale_p
    #正套利
    if(base_buy_p/change_sale_p - market_sale_p)/market_sale_p > 0.008:
        Q = check_normal_volume(base_buy_p,base_buy_v,change_sale_p,change_sale_v,market_sale_p,market_sale_v)
        base_account_receive = base_buy_p * Q * (1-0.003)
        market_account_used = market_sale_p * Q * (1-0.003)
        change_account_used = change_sale_p * market_account_used * (1-0.003)
        profit = base_account_receive-change_account_used
        
        return profit,Q,difference
    #逆套利    
    if(market_buy_p - base_sale_p/change_buy_p )/market_buy_p > 0.008:   
        Q = check_normal_volume(base_sale_p,base_sale_v,change_sale_p,change_sale_v,market_buy_p,market_buy_v)
        base_account_used = base_sale_p * Q * (1-0.003)
        market_account_receive = market_buy_p * Q * (1-0.003)
        change_account_receive = change_buy_p * market_account_receive * (1-0.003)
        profit = change_account_receive -  base_account_used
        
        return profit,Q,difference
    return  0,0,difference

        


# In[11]:


import pandas as pd
t = 0
base = []
change = []
market =[]
profit_get = []
total_get = []
total=0
volume_get = []
difference = []
point = []
with open('D:/DATA/LTCUSD.csv') as f:
    for i, line in enumerate(f): # 一行一行遍历，i 为遍历行数，从 0 开始，line 每行内容
        
        line_arr = line.split(',') # 通过 str.split 后返回 list 数据
        base.append(float(line_arr[2]))
        base.append(float(line_arr[3]))

with open('D:/DATA/BTCUSD.csv') as f:
    for i, line in enumerate(f): # 一行一行遍历，i 为遍历行数，从 0 开始，line 每行内容
        
        line_arr = line.split(',') # 通过 str.split 后返回 list 数据
        change.append(float(line_arr[2]))
        change.append(float(line_arr[3]))
with open('D:/DATA/LTCBTC.csv') as f:
    for i, line in enumerate(f): # 一行一行遍历，i 为遍历行数，从 0 开始，line 每行内容
        
        line_arr = line.split(',') # 通过 str.split 后返回 list 数据
        market.append(float(line_arr[2]))
        market.append(float(line_arr[3])) 
t = 0

print(len(base))
for i in range(29999):
    point.append(i)
    profit,volume,difference_get = profit_calculation(base[t:t+4],change[t:t+4],market[t:t+4])
    difference.append(difference_get)
    profit_get.append(profit)
    volume_get.append(volume)
    if(profit>0):
        total += profit
        total_get.append(total)
    t = t+4


# In[12]:


print(len(point),len(difference))
import matplotlib.pyplot as plt

#多大的单量，多大的冲击

plt.plot(total_get)
# plot函数作图


# show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
plt.show()  
print(total)
print(total_get)


# In[ ]:




