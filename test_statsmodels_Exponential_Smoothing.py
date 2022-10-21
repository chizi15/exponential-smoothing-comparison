"""
1. 每季度采用加权cross validation筛选模型，cross validation的间距可设为一个季度，权重按训练集长度由长至短递减，
最终加权后MASE(mean absolute scaled error)<1的模型则认为打败benchmark，在下一个季度可用；若MASE<1的模型不足五个，则按MASE升序取五个模型，
再计算各模型预测值的加权绝对偏差比(WADR)，权重按预测期由近及远递减，按WADR升序取五个模型，两次结果取并集作为每季度筛选出的模型。
在做加权cross validation时，若某轮MASE值过大，则当轮MASE值不进入加权，以免某种模型在某轮MASE值过大，直接被排除，以一次预测结果判定整体预测效果的情况出现。
benchmark可采用naive（或MV、dms等初级算法），此时benchmark的预测值采用单步预测模式，即折线模式，使按季度筛选的标准更严。

2. 在做季度筛选时，将各单品的周序列输入模型即可，预测4步，不使用日序列；或将周日序列均输入模型计算MASE值，但周序列的MASE占较大权重。
避免筛选条件过于严格导致通过筛选的模型数量太少，不利于后续步骤的进行。

3. 按每季度筛选出模型后，第一周使用所有模型的预测值，各模型的初始权重，以加权cross validation计算出的初始MASE为基础，
（各模型初始MASE通常大于下文中各自新MASE，0<新MASE<1），可按 1/MASE(i) 并作归一化得到，或按 1/log(1+MASE(i)) 并作归一化得到；
从第二周开始，每周进行新一轮预测时，先将筛选出的模型与benchmark计算一次新MASE，新MASE<1的模型才在当周使用其预测值；
假定此时有5个模型的当期新MASE<1，则这5个模型均采用，并有新MASE和上一轮MASE，计算( ln(e+新MASE(i))+ln(e+旧MASE(i)) ) / 2 = MASE(i)，
再按 1/MASE(i) 并作归一化得到新一轮权重，或按 1/log(1+MASE(i)) 并作归一化得到新一轮权重。此方法根据各模型的初始MASE及后续更新MASE的数值特征来滚动计算权重。
另1，每周预测时的benchmark须采用多步预测模式，即水平直线模式，使第二次的筛选标准符合实际预测时使用benchmark的情况。
另2，每周给出筛选后各模型的预测值后，仍需判断是否有极端离群的预测值或预测趋势出现，可用预测期的MASE做第一步筛选，绝对中位差做第二步筛选。

4. SimpleExpSmoothing和Holt(damped_trend=False)的目标函数是凸函数，调用局部优化方法可得最小值，与参数的初始位置无关（即可令use_brute=False）；
Holt(damped_trend=True)和ExponentialSmoothing的目标函数是非凸函数，调用全局优化方法才可得最小值（method='basinhopping'），此时与参数的初始位置无关（即可令use_brute=False）；
若调用局部优化方法（即use_basinhopping=False），则与参数的初始位置有关，此时令use_brute=True能更大概率接近最小值。

5. 通常采用全局优化方法会得到更好的拟合值和预测值，但耗时大大增加；
考虑到性价比，在汇总层级使用top-down时，再令use_basinhopping=True，当训练单品序列时，令use_basinhopping=False即可。

6. 当序列历史数据长度 ≥（一个周期+1）时，即可采用ExponentialSmoothing，此时s_t是由过去一个真实周期与过去一个虚拟周期对应点的y_t和s_t加权整合得到。
当序列历史数据长度 ≥（两个周期+1）时，ExponentialSmoothing中s_t是由过去两个真实周期、最初一个虚拟周期对应点的y_t和s_t加权整合得到。
通常历史周期数越多，带有季节项的ExponentialSmoothing的预测越稳健；当历史周期数越少，带有季节项的模型的拟合值和预测值间越容易出现跳跃，
在预测期容易出现离群值，因为季节项各点参考的历史值越、更少，受某一周期中数值大小的随机性影响更大。

7. 在Holt和ExponentialSmoothing中，当damped_trend=True时，预测值与历史观测值的偏离更小，各预测值的波动更小。
设置damping_trend在change-point、趋势改变处更保险，更不容易使预测值在趋势上出现较大偏离，对Holt尤其重要。

8. naive的多步预测模式（水平直线）和单步预测模式（折线），与SimpleExpSmoothing.fit(smoothing_level=1)的多步预测和单步预测模式均等价；
对于naive、SimpleExpSmoothing和Holt(damping_trend很小时)，因为预测值为水平直线，难以适用于预测期走势斜率较大的情况；
所以当使用Holt时，令damping_trend趋近1，use_brute=False更好。

9. 当训练集长度不足两个周期，或为两个周期左右，且预测期较长时，ExponentialSmoothing(trend='mul',seasonal='add')的预测值可能较快递增或递减；
因为在多步预测时，加法的季节项难以抑制乘法趋势项的指数式变化。

10. 当历史数据周期间带趋势时（不是指近期趋势trend项），对于ExponentialSmoothing，damped_trend=False更好，或指定damping_trend=1.

11. 对于Holt和ExponentialSmoothing，当预测值整体偏小时，应增大damping_trend；预测值整体偏大时，应减小damping_trend；
0<damping_trend≤1，防止出现离群预测值；damping_trend最好指定而不通过训练得到，一是减少模型训练时间，二是防止因训练出过小的phi值而使预测值趋近于水平直线。
因此当上一周预测偏小时，当周预测时可适当增大damping_trend；当上一周预测偏大时，当周预测时可适当减小damping_trend。

12. 多重季节性模型训练时间长，对于典型单一季节性时序，预测精度通常不如单一季节性模型高。

13. 如果在一定长度的验证集内预测值比真实值普遍偏小或偏大，则说明在训练集内有该模型没有捕捉到的模式；预测值与真实值的残差之和越趋近零越好。

14. 总结：从精度、开销、稳定性、全面性综合考虑，statsmodels中平滑类模型比生产系统和自定义平滑类模型更好。
statsmodels中可使用如下模型1：SimpleExpSmoothing().fit(optimized=True, use_brute=False)，
2：Holt(exponential=False, damped_trend=True).fit(damping_trend=趋近于1的小数, optimized=True, use_brute=False)，
3：Holt(exponential=True, damped_trend=True).fit(damping_trend=趋近于1的小数, optimized=True, use_brute=False)，
4：ExponentialSmoothing(trend='add', seasonal='add', damped_trend=True).fit(damping_trend=趋近于1的小数, use_boxcox=None, method='L-BFGS-B', use_brute=True)
5：ExponentialSmoothing(trend='add', seasonal='mul', damped_trend=True).fit(damping_trend=趋近于1的小数, use_boxcox=None, method='L-BFGS-B', use_brute=True)
6：ExponentialSmoothing(trend='add', seasonal='mul', damped_trend=True).fit(damping_trend=趋近于1的小数, use_boxcox=None, method='basinhopping', use_brute=False)
"""
"""
特别重要：
当序列长度为周期的n倍或n倍多几个点时，例如周期365，序列长度730，则从第730个点，即最后一个拟合点开始，包括其后数个预测点，会出现突变，
越远离第730个点，突变量越小，越接近越大；当序列长度比周期的倍数n越大时，这种突变影响越小；对于本例，即靠近730的最近的几个点突变量较大，
所以序列长度应比730多一些点如760。综上，序列长度应为周期的n倍多一些点。
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
# import test_smoothing_models
# import test2_smoothing_models
import wualgorithm
import numpy as np
import random
from warnings import filterwarnings
import seaborn as sns
sns.set_style('darkgrid')
plt.rc('font',size=10)
filterwarnings("ignore")

###########---------------set up and plot input data-----------------######################

up_limit = 20  # 设置level、trend、season项的取值上限
steps_day, steps_week = 28, 4
length = [730+steps_day, 365+steps_day, 104+steps_week, 52+steps_week]  # 代表每个序列的长度，分别为周、日序列的一年及两年。
y_level, y_trend, y_season, y_noise, y_input_add, y_input_mul = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)

weights = []
for i in range(-up_limit+1, 1):
    weights.append(0.6 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)

##########################################################

# 用正弦函数模拟加法多重季节性，并设置level，trend，noise分项
# y_season[0]是两年日序列的季节项，有两年、八个季度、24个月、104周共四个季节性分项
y_season[0] = 4 * (1/2*np.sin(np.linspace(0, 2*2*np.pi*(1+28/730), length[0])) \
              + 1/3*np.cos(np.linspace(0, 8*2*np.pi*(1+28/730), length[0])) \
              + 1/4*np.sin(np.linspace(0, 24*2*np.pi*(1+28/730), length[0])) \
              + 1/5*np.cos(np.linspace(0, 104*2*np.pi*(1+28/730), length[0])))
# y_season[1]是一年日序列的季节项，有一年、四个季度、12个月、52周共四个季节性分项
y_season[1] = 4 * (1/2*np.sin(np.linspace(0, 1*2*np.pi*(1+28/365), length[1])) \
              + 1/3*np.cos(np.linspace(0, 4*2*np.pi*(1+28/365), length[1])) \
              + 1/4*np.sin(np.linspace(0, 12*2*np.pi*(1+28/365), length[1])) \
              + 1/5*np.cos(np.linspace(0, 52*2*np.pi*(1+28/365), length[1])))
# y_season[2]是两年周序列的季节项，有两年、八个季度、24个月共三个季节性分项
y_season[2] = 3 * (np.sin(np.linspace(0, 2*2*np.pi*(1+4/104), length[2])) \
              + 1/2*np.cos(np.linspace(0, 8*2*np.pi*(1+4/104), length[2])) \
              + 1/3*np.sin(np.linspace(0, 24*2*np.pi*(1+4/104), length[2])))
# y_season[3]是一年周序列的季节项，有一年、四个季度、12个月共三个季节性分项
y_season[3] = 3 * (np.sin(np.linspace(0, 1*2*np.pi*(1+4/52), length[3])) \
              + 1/2*np.cos(np.linspace(0, 4*2*np.pi*(1+4/52), length[3])) \
              + 1/3*np.sin(np.linspace(0, 12*2*np.pi*(1+4/52), length[3])))
for i in range(0, len(length)):
    y_level[i] = np.array(random.choices(range(0, up_limit), weights=weights, k=length[i])) / 5 + 5  # 用指数权重分布随机数模拟基础项
    y_trend[i] = 2*max(y_season[i]) + np.log2(np.linspace(2, 2**(up_limit/8), num=length[i])) + (min(np.log2(np.linspace(2, 2**(up_limit/8), num=length[i]))) +
                 max(np.log2(np.linspace(2, 2**(up_limit/8), num=length[i])))) / length[i] * np.linspace(1, length[i], num=length[i]) # 用对数函数与线性函数的均值模拟趋势性
    y_noise[i] = np.random.normal(0, 1, length[i]) / 5  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈正态分布。
    y_input_add[i] = y_level[i] + y_trend[i] + y_season[i] + y_noise[i] # 假定各项以加法方式组成输入数据

    y_level[i] = pd.Series(y_level[i]).rename('y_level')
    y_trend[i] = pd.Series(y_trend[i]).rename('y_trend')
    y_season[i] = pd.Series(y_season[i]).rename('y_season')
    y_noise[i] = pd.Series(y_noise[i]).rename('y_noise')
    y_input_add[i] = pd.Series(y_input_add[i]).rename('y_input_add')
    y_input_add[i][y_input_add[i] < 0] = 0

# # 绘制四条加法季节性时间序列
# plt.figure('add: 730+28', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_add[0].plot(ax=ax1, legend=True)
# y_level[0].plot(ax=ax2, legend=True)
# y_trend[0].plot(ax=ax3, legend=True)
# y_season[0].plot(ax=ax4, legend=True)
# y_noise[0].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('add: 365+28', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_add[1].plot(ax=ax1, legend=True)
# y_level[1].plot(ax=ax2, legend=True)
# y_trend[1].plot(ax=ax3, legend=True)
# y_season[1].plot(ax=ax4, legend=True)
# y_noise[1].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('add: 104+4', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# y_input_add[2].plot(ax=ax1, legend=True)
# y_level[2].plot(ax=ax2, legend=True)
# y_trend[2].plot(ax=ax3, legend=True)
# y_season[2].plot(ax=ax4, legend=True)
# y_noise[2].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('add: 52+4', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[3])
# y_input_add[3].plot(ax=ax1, legend=True)
# y_level[3].plot(ax=ax2, legend=True)
# y_trend[3].plot(ax=ax3, legend=True)
# y_season[3].plot(ax=ax4, legend=True)
# y_noise[3].plot(ax=ax5, legend=True)
# plt.show()

##########################################################

# 用正弦函数模拟乘法多重季节性，并设置level，trend，noise分项
# y_season[0]是两年日序列的季节项，有两年、八个季度、24个月、104周共四个季节性分项
y_season[0] = 0.2 * (4+1/2+1/3+1/4+1/5 + 1/2*np.sin(np.linspace(0, 2*2*np.pi*(1+28/730), length[0])) \
              + 1/3*np.cos(np.linspace(0, 8*2*np.pi*(1+28/730), length[0])) \
              + 1/4*np.sin(np.linspace(0, 24*2*np.pi*(1+28/730), length[0])) \
              + 1/5*np.cos(np.linspace(0, 104*2*np.pi*(1+28/730), length[0])))
# y_season[1]是一年日序列的季节项，有一年、四个季度、12个月、52周共四个季节性分项
y_season[1] = 0.2 * (4+1/2+1/3+1/4+1/5 + 1/2*np.sin(np.linspace(0, 1*2*np.pi*(1+28/365), length[1])) \
              + 1/3*np.cos(np.linspace(0, 4*2*np.pi*(1+28/365), length[1])) \
              + 1/4*np.sin(np.linspace(0, 12*2*np.pi*(1+28/365), length[1])) \
              + 1/5*np.cos(np.linspace(0, 52*2*np.pi*(1+28/365), length[1])))
# y_season[2]是两年周序列的季节项，有两年、八个季度、24个月共三个季节性分项
y_season[2] = 0.2 * (3+1+1/2+1/3 + np.sin(np.linspace(0, 2*2*np.pi*(1+4/104), length[2])) \
              + 1/2*np.cos(np.linspace(0, 8*2*np.pi*(1+4/104), length[2])) \
              + 1/3*np.sin(np.linspace(0, 24*2*np.pi*(1+4/104), length[2])))
# y_season[3]是一年周序列的季节项，有一年、四个季度、12个月共三个季节性分项
y_season[3] = 0.2 * (3+1+1/2+1/3 + np.sin(np.linspace(0, 1*2*np.pi*(1+4/52), length[3])) \
              + 1/2*np.cos(np.linspace(0, 4*2*np.pi*(1+4/52), length[3])) \
              + 1/3*np.sin(np.linspace(0, 12*2*np.pi*(1+4/52), length[3])))
for i in range(0, len(length)):
    y_level[i] = np.array(random.choices(range(0, up_limit), weights=weights, k=length[i])) / 10 + 5  # 用指数权重分布随机数模拟基础项
    y_trend[i] = 2*max(y_season[i]) + np.log2(np.linspace(2, 2**(up_limit/8), num=length[i])) + (min(np.log2(np.linspace(2, 2**(up_limit/8), num=length[i]))) +
                 max(np.log2(np.linspace(2, 2**(up_limit/8), num=length[i])))) / length[i] * np.linspace(1, length[i], num=length[i]) # 用对数函数与线性函数的均值模拟趋势性
    # 假定数据处于理想状态，并使噪音以乘法方式进入模型，可构造外层函数是指数函数，内层函数是正态分布的复合函数，使噪音在1附近呈类正态分布。
    y_noise[i] = 1.1**(np.random.normal(0, 1, length[i])/5)
    y_input_mul[i] = (y_level[i] + y_trend[i]) * y_season[i] * y_noise[i] # 假定季节项以乘法方式组成输入数据

    y_level[i] = pd.Series(y_level[i]).rename('y_level')
    y_trend[i] = pd.Series(y_trend[i]).rename('y_trend')
    y_season[i] = pd.Series(y_season[i]).rename('y_season')
    y_noise[i] = pd.Series(y_noise[i]).rename('y_noise')
    y_input_mul[i] = pd.Series(y_input_mul[i]).rename('y_input_mul')
    y_input_mul[i][y_input_mul[i] < 0] = 0

# # 绘制四条乘法季节性时间序列
# plt.figure('mul: 730+28', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_mul[0].plot(ax=ax1, legend=True)
# y_level[0].plot(ax=ax2, legend=True)
# y_trend[0].plot(ax=ax3, legend=True)
# y_season[0].plot(ax=ax4, legend=True)
# y_noise[0].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('mul: 365+28', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_mul[1].plot(ax=ax1, legend=True)
# y_level[1].plot(ax=ax2, legend=True)
# y_trend[1].plot(ax=ax3, legend=True)
# y_season[1].plot(ax=ax4, legend=True)
# y_noise[1].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('mul: 104+4', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[2]-1)
# y_input_mul[2].plot(ax=ax1, legend=True)
# y_level[2].plot(ax=ax2, legend=True)
# y_trend[2].plot(ax=ax3, legend=True)
# y_season[2].plot(ax=ax4, legend=True)
# y_noise[2].plot(ax=ax5, legend=True)
# plt.show()
#
# plt.figure('mul: 52+4', figsize=(20,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[3]-1)
# y_input_mul[3].plot(ax=ax1, legend=True)
# y_level[3].plot(ax=ax2, legend=True)
# y_trend[3].plot(ax=ax3, legend=True)
# y_season[3].plot(ax=ax4, legend=True)
# y_noise[3].plot(ax=ax5, legend=True)
# plt.show()

#############################--------------above, set up and plot input data-----------------#############################

# ######################################------SES y_input_add------------######################################
#
# # SES中initial_level所用权重
# weights = []
# for i in range(1, len(y_input_add[0][0:730]) + 1):
#     weights.append(i / len(y_input_add[0][0:730]))
# weights = np.array(weights) / sum(weights)
#
# # 各模型拟合历史数据
# fit_SES = SimpleExpSmoothing(y_input_add[0][0:730], initialization_method='known',
#                              initial_level=np.average(y_input_add[0][0:730], weights=weights)).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_add[0][0:730], initialization_method='known',
#                              initial_level=np.average(y_input_add[0][0:730], weights=weights)).fit(optimized=True, use_brute=False)
#
# # 打印模型参数
# print()
# print('SimpleExpSmoothing_add_730:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# # 对各模型拟合值及预测值绘图，并作比较
# plt.figure('730+28+SES_add', figsize=(20,10))
# ax_SES = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='b')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='r')
# fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='r', legend=True)
# plt.show()
#
# # 计算各模型MASE值，小于1启用，否则弃用；各点偏差比的权重按由近及远递减；此处采用naive作为benchmark。
# MASE_SES = sum(abs((fit_SES.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_add_730，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_add_730，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_add_730，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_add_730，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# #----------------------------------------------------------------------------------------------------------------------
#
# # SES中initial_level所用权重
# weights = []
# for i in range(1, len(y_input_add[1][0:365]) + 1):
#     weights.append(i / len(y_input_add[1][0:365]))
# weights = np.array(weights) / sum(weights)
#
# # 各模型拟合历史数据
# fit_SES = SimpleExpSmoothing(y_input_add[1][0:365], initialization_method='known',
#                              initial_level=np.average(y_input_add[1][0:365], weights=weights)).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_add[1][0:365], initialization_method='known',
#                              initial_level=np.average(y_input_add[1][0:365], weights=weights)).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_add_365:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('365+28+SES_add', figsize=(20,10))
# ax_SES = y_input_add[1].rename('y_input_add[1]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_day).values - y_input_add[1][365:365+steps_day].values) / (y_input_add[1][365-1:365-1+steps_day].values - y_input_add[1][365:365+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_add_365，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_add_365，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES.forecast(steps_day).values - y_input_add[1][365:365+steps_day].values) / (y_input_add[1][365-1:365-1+steps_day].values - y_input_add[1][365:365+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_add_365，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_add_365，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ##############################--------------------------------------------------------------------------
#
# # SES中initial_level所用权重
# weights = []
# for i in range(1, len(y_input_add[2][0:104]) + 1):
#     weights.append(i / len(y_input_add[2][0:104]))
# weights = np.array(weights) / sum(weights)
#
# # 各模型拟合历史数据
# fit_SES = SimpleExpSmoothing(y_input_add[2][0:104], initialization_method='known',
#                              initial_level=np.average(y_input_add[2][0:104], weights=weights)).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_add[2][0:104], initialization_method='known',
#                              initial_level=np.average(y_input_add[2][0:104], weights=weights)).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_add_104:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SimpleExpSmoothing"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('104+4+SES_add', figsize=(20,10))
# ax_SES = y_input_add[2].rename('y_input_add[2]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_add_104，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_add_104，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_add_104，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_add_104，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ##############################------------------------------------------------------------------------------
#
# # SES中initial_level所用权重
# weights = []
# for i in range(1, len(y_input_add[3][0:52]) + 1):
#     weights.append(i / len(y_input_add[3][0:52]))
# weights = np.array(weights) / sum(weights)
#
# # 各模型拟合历史数据
# fit_SES = SimpleExpSmoothing(y_input_add[3][0:52], initialization_method='known',
#                              initial_level=np.average(y_input_add[3][0:52], weights=weights)).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_add[3][0:52], initialization_method='known',
#                              initial_level=np.average(y_input_add[3][0:52], weights=weights)).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_add_52:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('52+4+SES_add', figsize=(20,10))
# ax_SES = y_input_add[3].rename('y_input_add[3]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_week).values - y_input_add[3][52:52+steps_week].values) / (y_input_add[3][52-1:52-1+steps_week].values - y_input_add[3][52:52+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_add_52，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_add_52，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_week).values - y_input_add[3][52:52+steps_week].values) / (y_input_add[3][52-1:52-1+steps_week].values - y_input_add[3][52:52+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_add_52，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_add_52，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ############################----------SimpleExpSmoothing y_input_mul, below---------------#############################
#
# fit_SES = SimpleExpSmoothing(y_input_mul[0][0:730]).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_mul[0][0:730]).fit(optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha更好
#
# print()
# print('SimpleExpSmoothing_mul_730:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('730+28+SES_mul', figsize=(20,10))
# ax_SES = y_input_mul[0].rename('y_input_mul[0]').plot(color='k', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='b')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='r')
# fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='r', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_mul_730，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_mul_730，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_mul_730，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_mul_730，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ##############################-----------------------------------------------------------------------------------------
#
# fit_SES = SimpleExpSmoothing(y_input_mul[1][0:365]).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_mul[1][0:365]).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_365_mul:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('365+28+SES_mul', figsize=(20,10))
# ax_SES = y_input_mul[1].rename('y_input_mul[1]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_day).values - y_input_mul[1][365:365+steps_day].values) / (y_input_mul[1][365-1:365-1+steps_day].values - y_input_mul[1][365:365+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_mul_365，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_mul_365，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_day).values - y_input_mul[1][365:365+steps_day].values) / (y_input_mul[1][365-1:365-1+steps_day].values - y_input_mul[1][365:365+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_mul_365，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_mul_365，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ##############################-------------------------------------------------------------------------------------
#
# fit_SES = SimpleExpSmoothing(y_input_mul[2][0:104]).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_mul[2][0:104]).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_mul_104:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('104+4+SES_mul', figsize=(20,10))
# ax_SES = y_input_mul[2].rename('y_input_mul[2]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_mul_104，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_mul_104，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_mul_104，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_mul_104，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ##############################---------------------------------------------------------------------------------------
#
# fit_SES = SimpleExpSmoothing(y_input_mul[3][0:52]).fit(smoothing_level=0.2)
# fit_SES_train = SimpleExpSmoothing(y_input_mul[3][0:52]).fit(optimized=True, use_brute=False)
#
# print()
# print('SimpleExpSmoothing_mul_52:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# plt.figure('52+4+SES_mul', figsize=(20,10))
# ax_SES = y_input_mul[3].rename('y_input_mul[3]').plot(color='black', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# fit_SES.fittedvalues.plot(ax=ax_SES, color='blue')
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='red')
# fit_SES.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES.model.params['smoothing_level']).plot(ax=ax_SES, color='b', legend=True)
# fit_SES_train.forecast(steps_week).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='red', legend=True)
# plt.show()
#
# MASE_SES = sum(abs((fit_SES.forecast(steps_week).values - y_input_mul[3][52:52+steps_week].values) / (y_input_mul[3][52-1:52-1+steps_week].values - y_input_mul[3][52:52+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES < 1:
#     print('SimpleExpSmoothing_mul_52，fit_SES可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES))
#     print(fit_SES.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_mul_52，fit_SES不可用，其MASE值为：{:.2f}'.format(MASE_SES))
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_week).values - y_input_mul[3][52:52+steps_week].values) / (y_input_mul[3][52-1:52-1+steps_week].values - y_input_mul[3][52:52+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_mul_52，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_week))
# else:
#     print('SimpleExpSmoothing_mul_52，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# #######################################----------SES, above------------################################################
#
#
# ######################----------------------Holt y_input_mul, below---------------------------#######################
#
# # fit models; "exponential=True" means Holt multiplicative model
# Holt_add_dam = Holt(y_input_add[0][0:730], exponential=False, damped_trend=True).fit(smoothing_level=0.1, smoothing_trend=0.2, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_add[0][0:730], exponential=False, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_add[0][0:730], exponential=True, damped_trend=True).fit(smoothing_level=0.1, smoothing_trend=0.2, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_add[0][0:730], exponential=True, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# # print parameters
# print()
# print('Hlot_730 y_input_add:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# # print figures
# plt.figure('730+28+Holt_dam y_input_add', figsize=(20,10))
# ax_Holt = y_input_add[0].rename('y_input_add[0]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################---------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_add[1][0:365], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.15, smoothing_trend=0.25, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_add[1][0:365], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False,)
# Holt_mul_dam = Holt(y_input_add[1][0:365], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.15, smoothing_trend=0.25, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_add[1][0:365], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_365 y_input_add:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('365+28+Holt_dam y_input_add', figsize=(20,10))
# ax_Holt = y_input_add[1].rename('y_input_add[1]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################----------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_add[2][0:104], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.2, smoothing_trend=0.3, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_add[2][0:104], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_add[2][0:104], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.2, smoothing_trend=0.3, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_add[2][0:104], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_104 y_input_add:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('104+4+Holt_dam y_input_add', figsize=(20,10))
# ax_Holt = y_input_add[2].rename('y_input_add[2]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################--------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_add[3][0:52], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.25, smoothing_trend=0.35, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_add[3][0:52], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_add[3][0:52], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.25, smoothing_trend=0.35, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_add[3][0:52], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_52 y_input_add:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('52+4+Holt_dam y_input_add', figsize=(20,10))
# ax_Holt = y_input_add[3].rename('y_input_add[3]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ########################-----------------------Holt y_input_mul, below-----------------------#########################
#
# Holt_add_dam = Holt(y_input_mul[0][0:730], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.1, smoothing_trend=0.2, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_mul[0][0:730], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_mul[0][0:730], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.1, smoothing_trend=0.2, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_mul[0][0:730], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_730 y_input_mul:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('730+28+Holt_dam y_input_mul', figsize=(20,10))
# ax_Holt = y_input_mul[0].rename('y_input_mul[0]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_mul[1][0:365], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.15, smoothing_trend=0.25, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_mul[1][0:365], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_mul[1][0:365], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.15, smoothing_trend=0.25, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_mul[1][0:365], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_365 y_input_mul:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('365+28+Holt_dam y_input_mul', figsize=(20,10))
# ax_Holt = y_input_mul[1].rename('y_input_mul[1]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_day).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_mul[2][0:104], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.2, smoothing_trend=0.3, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_mul[2][0:104], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_mul[2][0:104], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.2, smoothing_trend=0.3, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_mul[2][0:104], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
#
# print()
# print('Hlot_104 y_input_mul:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('104+4+Holt_dam y_input_mul', figsize=(20,10))
# ax_Holt = y_input_mul[2].rename('y_input_mul[2]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# ##############################-------------------------------------------------------------------------------------
#
# Holt_add_dam = Holt(y_input_mul[3][0:52], exponential=False, damped_trend=True).\
#     fit(smoothing_level=0.25, smoothing_trend=0.35, damping_trend=0.99, optimized=False)
# Holt_add_dam_train = Holt(y_input_mul[3][0:52], exponential=False, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam = Holt(y_input_mul[3][0:52], exponential=True, damped_trend=True).\
#     fit(smoothing_level=0.25, smoothing_trend=0.35, damping_trend=0.99, optimized=False)
# Holt_mul_dam_train = Holt(y_input_mul[3][0:52], exponential=True, damped_trend=True).\
#     fit(damping_trend=0.99, optimized=True, use_brute=False) # start_params取上一轮训练得到的alpha/beta/phi更好
#
# print()
# print('Hlot_52 y_input_mul:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
# results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam.params[p] for p in params] + [Holt_mul_dam.sse]
# results["Holt_mul_dam_train"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# plt.figure('52+4+Holt_dam y_input_mul', figsize=(20,10))
# ax_Holt = y_input_mul[3].rename('y_input_mul[3]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("week")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_dam.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_dam.forecast(steps_week).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_week).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_dam.forecast(steps_week).rename('Holt_mul_dam').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_week).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# #################################---------------Holt, above------------################################
#
#
# #######################------------HoltWinters y_input_add, below------------#########################
#
# # fit models
# HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# # print figures
# plt.figure('730+28+HoltWinters y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print parameters
# print()
# print('HoltWinters_730 y_input_add:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# # print internals
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# # print the models to select
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# #######################------------global optimization, method optional------------#########################
#
# # fit models
# HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
#
# # print figures
# plt.figure('730+28+HoltWinters y_input_add global optimization', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print parameters
# print()
# print('HoltWinters_730 y_input_add global optimization:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# # print internals
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# # print the models to select
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_730 y_input_add global optimization，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add global optimization，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_730 y_input_add global optimization，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add global optimization，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_730 y_input_add global optimization，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add global optimization，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_730 y_input_add global optimization，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add global optimization，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# #######################################################################------------------------------------------
#
# # ExponentialSmoothing中设置initial_level, initial_trend, initial_seasonal时所用权重
# weights = []
# for i in range(1, len(y_input_add[1][0:365+1]) + 1):
#     weights.append(i / len(y_input_add[1][0:365+1]))
# weights = np.array(weights) / sum(weights)
#
# HW_add_add_dam = ExponentialSmoothing(y_input_add[1][0:365+1], seasonal_periods=365, trend='add', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level=np.average(y_input_add[1][0:365+1], weights=weights),
#         initial_trend=np.array((sum(y_input_add[1][0:365+1][int(np.ceil(len(y_input_add[1][0:365+1]) / 2)):]) - sum(y_input_add[1][0:365+1][:int(np.floor(len(y_input_add[1][0:365+1]) / 2))])) / (np.floor(len(y_input_add[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[1][0:365+1][:len(y_input_add[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_add[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[1][0:365+1], seasonal_periods=365, trend='add', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level=np.average(y_input_add[1][0:365+1], weights=weights),
#         initial_trend=np.array((sum(y_input_add[1][0:365+1][int(np.ceil(len(y_input_add[1][0:365+1]) / 2)):]) - sum(y_input_add[1][0:365+1][:int(np.floor(len(y_input_add[1][0:365+1]) / 2))])) / (np.floor(len(y_input_add[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[1][0:365+1][:len(y_input_add[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_add[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level=np.average(y_input_add[1][0:365+1], weights=weights),
#         initial_trend=np.array((sum(y_input_add[1][0:365+1][int(np.ceil(len(y_input_add[1][0:365+1]) / 2)):]) - sum(y_input_add[1][0:365+1][:int(np.floor(len(y_input_add[1][0:365+1]) / 2))])) / (np.floor(len(y_input_add[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[1][0:365+1][:len(y_input_add[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_add[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level=np.average(y_input_add[1][0:365+1], weights=weights),
#         initial_trend=np.array((sum(y_input_add[1][0:365+1][int(np.ceil(len(y_input_add[1][0:365+1]) / 2)):]) - sum(y_input_add[1][0:365+1][:int(np.floor(len(y_input_add[1][0:365+1]) / 2))])) / (np.floor(len(y_input_add[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[1][0:365+1][:len(y_input_add[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_add[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('366+27+HoltWinters y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[1].rename('y_input_add[1]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_366 y_input_add:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[1][0:365+1], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day-1) - y_input_add[1][365+1:]) / y_input_add[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[1][0:365+1], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day-1) - y_input_add[1][365+1:]) / y_input_add[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[1][0:365+1], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day-1) - y_input_add[1][365+1:]) / y_input_add[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[1][0:365+1], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day-1) - y_input_add[1][365+1:]) / y_input_add[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day-1).values - y_input_add[1][365+1:365+1+steps_day-1].values) / (y_input_add[1][365:365+steps_day-1].values - y_input_add[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day-1).values - y_input_add[1][365+1:365+1+steps_day-1].values) / (y_input_add[1][365:365+steps_day-1].values - y_input_add[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day-1).values - y_input_add[1][365+1:365+1+steps_day-1].values) / (y_input_add[1][365:365+steps_day-1].values - y_input_add[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day-1).values - y_input_add[1][365+1:365+1+steps_day-1].values) / (y_input_add[1][365:365+steps_day-1].values - y_input_add[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_366 y_input_add，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_add，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_366 y_input_add，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_add，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_366 y_input_add，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_add，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_366 y_input_add，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_add，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ##############################-------------------------------------------------------------------------------
#
# HW_add_add_dam = ExponentialSmoothing(y_input_add[2][0:104], seasonal_periods=52, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[2][0:104], seasonal_periods=52, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[2][0:104], seasonal_periods=52, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[2][0:104], seasonal_periods=52, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('104+4+HoltWinters y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[2].rename('y_input_add[2]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_week).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_week).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_week).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_week).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_104 y_input_add:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[2][:104], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[2][:104].index)
# print('internal items of HoltWinters_104 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_week) - y_input_add[2][104:]) / y_input_add[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[2][:104], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[2][:104].index)
# print('internal items of HoltWinters_104 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_week) - y_input_add[2][104:]) / y_input_add[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[2][:104], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[2][:104].index)
# print('internal items of HoltWinters_104 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_week) - y_input_add[2][104:]) / y_input_add[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[2][:104], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[2][:104].index)
# print('internal items of HoltWinters_104 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_week) - y_input_add[2][104:]) / y_input_add[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_week).values - y_input_add[2][104:104+steps_week].values) / (y_input_add[2][104-1:104-1+steps_week].values - y_input_add[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_104 y_input_add，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_add，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_104 y_input_add，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_add，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_104 y_input_add，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_add，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_104 y_input_add，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_add，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ##############################------------------------------------------------------------------------------
#
# # ExponentialSmoothing中设置initial_level, initial_trend, initial_seasonal时所用权重
# weights = []
# for i in range(1, len(y_input_add[3][0:52+1]) + 1):
#     weights.append(i / len(y_input_add[3][0:52+1]))
# weights = np.array(weights) / sum(weights)
#
# HW_add_add_dam = ExponentialSmoothing(y_input_add[3][0:52+1], seasonal_periods=52, trend='add', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_add[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_add[3][0:52+1][int(np.ceil(len(y_input_add[3][0:52+1]) / 2)):]) - sum(y_input_add[3][0:52+1][:int(np.floor(len(y_input_add[3][0:52+1]) / 2))])) / (np.floor(len(y_input_add[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[3][0:52+1][:len(y_input_add[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_add[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[3][0:52+1], seasonal_periods=52, trend='add', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_add[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_add[3][0:52+1][int(np.ceil(len(y_input_add[3][0:52+1]) / 2)):]) - sum(y_input_add[3][0:52+1][:int(np.floor(len(y_input_add[3][0:52+1]) / 2))])) / (np.floor(len(y_input_add[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[3][0:52+1][:len(y_input_add[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_add[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_add[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_add[3][0:52+1][int(np.ceil(len(y_input_add[3][0:52+1]) / 2)):]) - sum(y_input_add[3][0:52+1][:int(np.floor(len(y_input_add[3][0:52+1]) / 2))])) / (np.floor(len(y_input_add[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[3][0:52+1][:len(y_input_add[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_add[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_add[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_add[3][0:52+1][int(np.ceil(len(y_input_add[3][0:52+1]) / 2)):]) - sum(y_input_add[3][0:52+1][:int(np.floor(len(y_input_add[3][0:52+1]) / 2))])) / (np.floor(len(y_input_add[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_add[3][0:52+1][:len(y_input_add[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_add[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('53+3+HoltWinters y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[3].rename('y_input_add[3]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_week-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_week-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_week-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_week-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_53 y_input_add:')
# results = pd.DataFrame(index=['alpha','beta','phi','gamma','l_0','b_0','SSE'])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[3][:52+1], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_week-1) - y_input_add[3][52+1:]) / y_input_add[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[3][:52+1], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_week-1) - y_input_add[3][52+1:]) / y_input_add[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[3][:52+1], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_week-1) - y_input_add[3][52+1:]) / y_input_add[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[3][:52+1], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_week-1) - y_input_add[3][52+1:]) / y_input_add[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_week-1).values - y_input_add[3][52+1:52+1+steps_week-1].values) / (y_input_add[3][52:52+steps_week-1].values - y_input_add[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_week-1).values - y_input_add[3][52+1:52+1+steps_week-1].values) / (y_input_add[3][52:52+steps_week-1].values - y_input_add[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_week-1).values - y_input_add[3][52+1:52+1+steps_week-1].values) / (y_input_add[3][52:52+steps_week-1].values - y_input_add[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_week-1).values - y_input_add[3][52+1:52+1+steps_week-1].values) / (y_input_add[3][52:52+steps_week-1].values - y_input_add[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_53 y_input_add，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_add，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_53 y_input_add，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_add，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_53 y_input_add，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_add，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_53 y_input_add，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_add，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# #####################--------------------HoltWinters y_input_mul, below----------------------#########################
#
# HW_add_add_dam = ExponentialSmoothing(y_input_mul[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_mul[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_mul[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('730+28+HoltWinters y_input_mul', figsize=(20,10))
# ax_HoltWinters = y_input_mul[0].rename('y_input_mul[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_730 y_input_mul:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_mul[0][:730], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day) - y_input_mul[0][730:]) / y_input_mul[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_mul[0][:730], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day) - y_input_mul[0][730:]) / y_input_mul[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_mul[0][:730], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day) - y_input_mul[0][730:]) / y_input_mul[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_mul[0][:730], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day) - y_input_mul[0][730:]) / y_input_mul[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day).values - y_input_mul[0][730:730+steps_day].values) / (y_input_mul[0][730-1:730-1+steps_day].values - y_input_mul[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_730 y_input_mul，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_mul，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_730 y_input_mul，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_mul，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_730 y_input_mul，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_mul，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_730 y_input_mul，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_mul，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ##############################-----------------------------------------------------------------------------------------
#
# # ExponentialSmoothing中设置initial_level, initial_trend, initial_seasonal时所用权重
# weights = []
# for i in range(1, len(y_input_mul[1][0:365+1]) + 1):
#     weights.append(i / len(y_input_mul[1][0:365+1]))
# weights = np.array(weights) / sum(weights)
#
# HW_add_add_dam = ExponentialSmoothing(y_input_mul[1][0:365+1], seasonal_periods=365, trend='add', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[1][0:365+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[1][0:365+1][int(np.ceil(len(y_input_mul[1][0:365+1]) / 2)):]) - sum(y_input_mul[1][0:365+1][:int(np.floor(len(y_input_mul[1][0:365+1]) / 2))])) / (np.floor(len(y_input_mul[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[1][0:365+1][:len(y_input_mul[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_mul[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_mul[1][0:365+1], seasonal_periods=365, trend='add', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[1][0:365+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[1][0:365+1][int(np.ceil(len(y_input_mul[1][0:365+1]) / 2)):]) - sum(y_input_mul[1][0:365+1][:int(np.floor(len(y_input_mul[1][0:365+1]) / 2))])) / (np.floor(len(y_input_mul[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[1][0:365+1][:len(y_input_mul[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_mul[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_mul[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[1][0:365+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[1][0:365+1][int(np.ceil(len(y_input_mul[1][0:365+1]) / 2)):]) - sum(y_input_mul[1][0:365+1][:int(np.floor(len(y_input_mul[1][0:365+1]) / 2))])) / (np.floor(len(y_input_mul[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[1][0:365+1][:len(y_input_mul[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_mul[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_mul[1][0:365+1], seasonal_periods=365, trend='mul', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[1][0:365+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[1][0:365+1][int(np.ceil(len(y_input_mul[1][0:365+1]) / 2)):]) - sum(y_input_mul[1][0:365+1][:int(np.floor(len(y_input_mul[1][0:365+1]) / 2))])) / (np.floor(len(y_input_mul[1][0:365+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[1][0:365+1][:len(y_input_mul[1][0:365+1]) // 365 * 365]).reshape(-1, 365).mean(axis=0) - np.average(y_input_mul[1][0:365+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('366+27+HoltWinters y_input_mul', figsize=(20,10))
# ax_HoltWinters = y_input_mul[1].rename('y_input_mul[1]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[1]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_366 y_input_mul:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_mul[1][0:365+1], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day-1) - y_input_mul[1][365+1:]) / y_input_mul[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_mul[1][0:365+1], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day-1) - y_input_mul[1][365+1:]) / y_input_mul[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_mul[1][0:365+1], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day-1) - y_input_mul[1][365+1:]) / y_input_mul[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_mul[1][0:365+1], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[1][0:365+1].index)
# print('internal items of HoltWinters_366 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day-1) - y_input_mul[1][365+1:]) / y_input_mul[1][365+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_366 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day-1).values - y_input_mul[1][365+1:365+1+steps_day-1].values) / (y_input_mul[1][365:365+steps_day-1].values - y_input_mul[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[1][365+1:365+1+steps_day-1].values) / (y_input_mul[1][365:365+steps_day-1].values - y_input_mul[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day-1).values - y_input_mul[1][365+1:365+1+steps_day-1].values) / (y_input_mul[1][365:365+steps_day-1].values - y_input_mul[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day-1).values - y_input_mul[1][365+1:365+1+steps_day-1].values) / (y_input_mul[1][365:365+steps_day-1].values - y_input_mul[1][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day-1+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_366 y_input_mul，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_mul，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_366 y_input_mul，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_mul，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_366 y_input_mul，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_mul，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_366 y_input_mul，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day-1))
# else:
#     print('HoltWinters_366 y_input_mul，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ##############################-----------------------------------------------------------------------------------------
#
# HW_add_add_dam = ExponentialSmoothing(y_input_mul[2][0:104], seasonal_periods=52, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_mul[2][0:104], seasonal_periods=52, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_mul[2][0:104], seasonal_periods=52, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_mul[2][0:104], seasonal_periods=52, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# plt.figure('104+4+HoltWinters y_input_mul', figsize=(20,10))
# ax_HoltWinters = y_input_mul[2].rename('y_input_mul[2]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[2]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_week).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_week).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_week).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_week).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# print()
# print('HoltWinters_104 y_input_mul:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_mul[2][:104], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[2][:104].index)
# print('internal items of HoltWinters_104 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_week) - y_input_mul[2][104:]) / y_input_mul[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_mul[2][:104], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[2][:104].index)
# print('internal items of HoltWinters_104 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_week) - y_input_mul[2][104:]) / y_input_mul[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_mul[2][:104], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[2][:104].index)
# print('internal items of HoltWinters_104 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_week) - y_input_mul[2][104:]) / y_input_mul[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_mul[2][:104], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[2][:104].index)
# print('internal items of HoltWinters_104 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_week) - y_input_mul[2][104:]) / y_input_mul[2][104:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_104 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_week).values - y_input_mul[2][104:104+steps_week].values) / (y_input_mul[2][104-1:104-1+steps_week].values - y_input_mul[2][104:104+steps_week].values))
#     * (np.array(range(steps_week, 0, -1)) / sum(np.array(range(1, steps_week+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_104 y_input_mul，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_mul，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_104 y_input_mul，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_mul，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_104 y_input_mul，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_mul，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_104 y_input_mul，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_week))
# else:
#     print('HoltWinters_104 y_input_mul，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ##############################-------------------------------------------------------------------------------------
#
# # ExponentialSmoothing中设置initial_level, initial_trend, initial_seasonal时所用权重
# weights = []
# for i in range(1, len(y_input_mul[3][0:52+1]) + 1):
#     weights.append(i / len(y_input_mul[3][0:52+1]))
# weights = np.array(weights) / sum(weights)
#
# # fit models
# HW_add_add_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='add', seasonal='add',
#                                       damped_trend=False, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='add', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# # plot time series
# plt.figure('53+3+HoltWinters y_input_mul', figsize=(20,10))
# ax_HoltWinters = y_input_mul[3].rename('y_input_mul[3]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_week-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_week-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_week-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_week-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print model parameters
# print()
# print('HoltWinters_53 y_input_mul:')
# results = pd.DataFrame(index=['alpha','beta','phi','gamma','l_0','b_0','SSE'])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# # print model internals
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# # select models through benchmark
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# #######################------------global optimization, method optional, below------------#########################
#
# # ExponentialSmoothing中设置initial_level, initial_trend, initial_seasonal时所用权重
# weights = []
# for i in range(1, len(y_input_mul[3][0:52+1]) + 1):
#     weights.append(i / len(y_input_mul[3][0:52+1]))
# weights = np.array(weights) / sum(weights)
#
# # fit models
# HW_add_add_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='add', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_add_mul_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='add', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_mul_add_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='add',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_mul[3][0:52+1], seasonal_periods=52, trend='mul', seasonal='mul',
#                                       damped_trend=True, initialization_method='known',
#         initial_level = np.average(y_input_mul[3][0:52+1], weights=weights),
#         initial_trend = np.array((sum(y_input_mul[3][0:52+1][int(np.ceil(len(y_input_mul[3][0:52+1]) / 2)):]) - sum(y_input_mul[3][0:52+1][:int(np.floor(len(y_input_mul[3][0:52+1]) / 2))])) / (np.floor(len(y_input_mul[3][0:52+1]) / 2)) ** 2),
#         initial_seasonal=np.array(y_input_mul[3][0:52+1][:len(y_input_mul[3][0:52+1]) // 52 * 52]).reshape(-1, 52).mean(axis=0) - np.average(y_input_mul[3][0:52+1], weights=weights)).\
#     fit(damping_trend=0.99, use_boxcox=None, method='basinhopping', use_brute=False)
#
# # plot time series
# plt.figure('53+3+HoltWinters y_input_mul global optimization', figsize=(20,10))
# ax_HoltWinters = y_input_mul[3].rename('y_input_mul[3]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[3]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_week-1).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_week-1).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_week-1).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_week-1).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print model parameters
# print()
# print('HoltWinters_53 y_input_mul global optimization:')
# results = pd.DataFrame(index=['alpha','beta','phi','gamma','l_0','b_0','SSE'])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# # print model internals
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_mul[3][:52+1], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_mul[3][:52+1].index)
# print('internal items of HoltWinters_53 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_week-1) - y_input_mul[3][52+1:]) / y_input_mul[3][52+1:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_53 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# # select models through benchmark
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_week-1).values - y_input_mul[3][52+1:52+1+steps_week-1].values) / (y_input_mul[3][52:52+steps_week-1].values - y_input_mul[3][52+1:52+1+steps_week-1].values))
#     * (np.array(range(steps_week-1, 0, -1)) / sum(np.array(range(1, steps_week-1+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_53 y_input_mul，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_week-1))
# else:
#     print('HoltWinters_53 y_input_mul，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))
#
# ######################--------------------------HoltWinters, above-------------------------#########################
#
#
# #####################################---分析、对比statsmodels中以下13个平滑类模型---########################################
#
# ###############################################---SES_train, below---###############################################
# # fit model
# fit_SES_train = SimpleExpSmoothing(y_input_add[0][0:730]).fit(optimized=True, use_brute=False)
#
# # 打印模型参数
# print()
# print('SimpleExpSmoothing_add_730:')
# results=pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
# params = ['smoothing_level', 'initial_level']
# results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
# print(results)
#
# # 对各模型拟合值及预测值绘图，并作比较
# plt.figure('730+28+SimpleExpSmoothing_add', figsize=(20,10))
# ax_SES = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend=True)
# ax_SES.set_ylabel("amount")
# ax_SES.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# fit_SES_train.fittedvalues.plot(ax=ax_SES, color='r')
# fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$'%fit_SES_train.model.params['smoothing_level']).plot(ax=ax_SES, color='r', legend=True)
# plt.show()
#
# # print forecast deviation ratio
# fit_SES_train_residual = (fit_SES_train.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of fit_SES_train_730 is:')
# print(fit_SES_train_residual)
#
# # 计算各模型MASE值，小于1启用，否则弃用；各点偏差比的权重按由近及远递减；此处采用naive作为benchmark。
# MASE_SES_train = sum(abs((fit_SES_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_SES_train < 1:
#     print('SimpleExpSmoothing_add_730，fit_SES_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_SES_train))
#     print(fit_SES_train.forecast(steps_day))
# else:
#     print('SimpleExpSmoothing_add_730，fit_SES_train不可用，其MASE值为：{:.2f}'.format(MASE_SES_train))
#
# ########################################---Holt_train, bwlow---######################################################
#
# # fit models
# Holt_add_train = Holt(y_input_add[0][0:730], exponential=False, damped_trend=False).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_add_dam_train = Holt(y_input_add[0][0:730], exponential=False, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_train = Holt(y_input_add[0][0:730], exponential=True, damped_trend=False).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_dam_train = Holt(y_input_add[0][0:730], exponential=True, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False)
#
# # print model parameters
# print()
# print('Hlot_730 y_input_add:')
# results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
# results["Holt_add"] = [Holt_add_train.params[p] for p in params] + [Holt_add_train.sse]
# results["Holt_add_dam"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
# results["Holt_mul"] = [Holt_mul_train.params[p] for p in params] + [Holt_mul_train.sse]
# results["Holt_mul_dam"] = [Holt_mul_dam_train.params[p] for p in params] + [Holt_mul_dam_train.sse]
# print(results)
#
# # print figures
# plt.figure('730+28+Holt_train y_input_add', figsize=(20,10))
# ax_Holt = y_input_add[0].rename('y_input_add[0]').plot(color='black', legend=True)
# ax_Holt.set_ylabel("amount")
# ax_Holt.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# Holt_add_train.fittedvalues.plot(ax=ax_Holt, color='blue')
# Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
# Holt_mul_train.fittedvalues.plot(ax=ax_Holt, color='g')
# Holt_mul_dam_train.fittedvalues.plot(ax=ax_Holt, color='y')
# Holt_add_train.forecast(steps_day).rename('Holt_add_train').plot(ax=ax_Holt, color='b', legend=True)
# Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)
# Holt_mul_train.forecast(steps_day).rename('Holt_mul_train').plot(ax=ax_Holt, color='g', legend=True)
# Holt_mul_dam_train.forecast(steps_day).rename('Holt_mul_dam_train').plot(ax=ax_Holt, color='y', legend=True)
# plt.show()
#
# # print deviation ratio
# Holt_add_train_residual = (Holt_add_train.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of Holt_add_train_730 is:')
# print(Holt_add_train_residual)
# Holt_add_dam_train_residual = (Holt_add_dam_train.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of Holt_add_dam_train_730 is:')
# print(Holt_add_dam_train_residual)
# Holt_mul_train_residual = (Holt_mul_train.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of Holt_mul_train_730 is:')
# print(Holt_mul_train_residual)
# Holt_mul_dam_train_residual = (Holt_mul_dam_train.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of Holt_mul_dam_train_730 is:')
# print(Holt_mul_dam_train_residual)
#
# # print the models to select
# MASE_Holt_add_train = sum(abs((Holt_add_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_Holt_add_dam_train = sum(abs((Holt_add_dam_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_Holt_mul_train = sum(abs((Holt_mul_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_Holt_mul_dam_train = sum(abs((Holt_mul_dam_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_Holt_add_train < 1:
#     print('Holt_730 y_input_add，Holt_add_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_Holt_add_train))
#     print(Holt_add_train.forecast(steps_day))
# else:
#     print('Holt_730 y_input_add，Holt_add_train不可用，其MASE值为：{:.2f}'.format(MASE_Holt_add_train))
# if MASE_Holt_add_dam_train < 1:
#     print('Holt_730 y_input_add，Holt_add_dam_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_Holt_add_dam_train))
#     print(Holt_add_dam_train.forecast(steps_day))
# else:
#     print('Holt_730 y_input_add，Holt_add_dam_train不可用，其MASE值为：{:.2f}'.format(MASE_Holt_add_dam_train))
# if MASE_Holt_mul_train < 1:
#     print('Holt_730 y_input_add，Holt_mul_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_Holt_mul_train))
#     print(Holt_mul_train.forecast(steps_day))
# else:
#     print('Holt_730 y_input_add，Holt_mul_train不可用，其MASE值为：{:.2f}'.format(MASE_Holt_mul_train))
# if MASE_Holt_mul_dam_train < 1:
#     print('Holt_730 y_input_add，Holt_mul_dam_train可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_Holt_mul_dam_train))
#     print(Holt_mul_dam_train.forecast(steps_day))
# else:
#     print('Holt_730 y_input_add，Holt_mul_dam_train不可用，其MASE值为：{:.2f}'.format(MASE_Holt_mul_dam_train))
#
# ##################################-------ExponentialSmoothing, below-------############################################
#
# # fit models
# HW_add_add = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped_trend=False).\
#     fit(use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped_trend=False).\
#     fit(use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped_trend=False).\
#     fit(use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped_trend=False).\
#     fit(use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# # print figures
# plt.figure('730+28+ExponentialSmoothing y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# HW_add_add.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add.forecast(steps_day).rename('HW_add_add').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul.forecast(steps_day).rename('HW_add_mul').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add.forecast(steps_day).rename('HW_mul_add').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul.forecast(steps_day).rename('HW_mul_mul').plot(ax=ax_HoltWinters, color='y', legend=True)
#
# # print parameters
# print()
# print('HoltWinters_730 y_input_add:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add"] = [HW_add_add.params[p] for p in params] + [HW_add_add.sse]
# results["HW_add_mul"] = [HW_add_mul.params[p] for p in params] + [HW_add_mul.sse]
# results["HW_mul_add"] = [HW_mul_add.params[p] for p in params] + [HW_mul_add.sse]
# results["HW_mul_mul"] = [HW_mul_mul.params[p] for p in params] + [HW_mul_mul.sse]
# print(results)
#
# # print internals
# df_HW_add_add = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_add.level, HW_add_add.trend, HW_add_add.season, HW_add_add.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_add are:')
# print(df_HW_add_add)
# HW_add_add_residual = (HW_add_add.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_add is:')
# print(HW_add_add_residual)
#
# df_HW_add_mul = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_mul.level, HW_add_mul.trend, HW_add_mul.season, HW_add_mul.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_mul are:')
# print(df_HW_add_mul)
# HW_add_mul_residual = (HW_add_mul.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_mul is:')
# print(HW_add_mul_residual)
#
# df_HW_mul_add = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_add.level, HW_mul_add.trend, HW_mul_add.season, HW_mul_add.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_add are:')
# print(df_HW_mul_add)
# HW_mul_add_residual = (HW_mul_add.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_add is:')
# print(HW_mul_add_residual)
#
# df_HW_mul_mul = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_mul.level, HW_mul_mul.trend, HW_mul_mul.season, HW_mul_mul.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_mul are:')
# print(df_HW_mul_mul)
# HW_mul_mul_residual = (HW_mul_mul.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_mul is:')
# print(HW_mul_mul_residual)
#
# # print the models to select
# MASE_HW_add_add = sum(abs((HW_add_add.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_add_mul = sum(abs((HW_add_mul.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_add = sum(abs((HW_mul_add.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_mul = sum(abs((HW_mul_mul.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_HW_add_add < 1:
#     print('HoltWinters_730 y_input_add，HW_add_add可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add))
#     print(HW_add_add.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_add不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add))
# if MASE_HW_add_mul < 1:
#     print('HoltWinters_730 y_input_add，HW_add_mul可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul))
#     print(HW_add_mul.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_mul不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul))
# if MASE_HW_mul_add < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_add可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add))
#     print(HW_mul_add.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_add不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add))
# if MASE_HW_mul_mul < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_mul可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul))
#     print(HW_mul_mul.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_mul不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul))
#
# ####################################-------ExponentialSmoothing_dam, below-----########################################
#
# # fit models
# HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_add_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='add', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='add', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
# HW_mul_mul_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=365, trend='mul', seasonal='mul', damped_trend=True).\
#     fit(damping_trend=0.99, use_boxcox=None, method='L-BFGS-B', use_brute=True)
#
# # print figures
# plt.figure('730+28+ExponentialSmoothing_dam y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='red')
# HW_add_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
# HW_mul_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='g')
# HW_mul_mul_dam.fittedvalues.plot(ax=ax_HoltWinters, color='y')
# HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# HW_add_mul_dam.forecast(steps_day).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
# HW_mul_add_dam.forecast(steps_day).rename('HW_mul_add_dam').plot(ax=ax_HoltWinters, color='g', legend=True)
# HW_mul_mul_dam.forecast(steps_day).rename('HW_mul_mul_dam').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print parameters
# print()
# print('HoltWinters_730 y_input_add:')
# results = pd.DataFrame(index=["alpha","beta","phi","gamma","l_0","b_0","SSE"])
# params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
# results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
# results["HW_add_mul_dam"] = [HW_add_mul_dam.params[p] for p in params] + [HW_add_mul_dam.sse]
# results["HW_mul_add_dam"] = [HW_mul_add_dam.params[p] for p in params] + [HW_mul_add_dam.sse]
# results["HW_mul_mul_dam"] = [HW_mul_mul_dam.params[p] for p in params] + [HW_mul_mul_dam.sse]
# print(results)
#
# # print internals
# df_HW_add_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_add_dam.level, HW_add_add_dam.trend, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_add_dam are:')
# print(df_HW_add_add_dam)
# HW_add_add_dam_residual = (HW_add_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_add_dam is:')
# print(HW_add_add_dam_residual)
#
# df_HW_add_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_add_mul_dam.level, HW_add_mul_dam.trend, HW_add_mul_dam.season, HW_add_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_add_mul_dam are:')
# print(df_HW_add_mul_dam)
# HW_add_mul_dam_residual = (HW_add_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_add_mul_dam is:')
# print(HW_add_mul_dam_residual)
#
# df_HW_mul_add_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_add_dam.level, HW_mul_add_dam.trend, HW_mul_add_dam.season, HW_mul_add_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_add_dam are:')
# print(df_HW_mul_add_dam)
# HW_mul_add_dam_residual = (HW_mul_add_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_add_dam is:')
# print(HW_mul_add_dam_residual)
#
# df_HW_mul_mul_dam = pd.DataFrame(np.c_[y_input_add[0][:730], HW_mul_mul_dam.level, HW_mul_mul_dam.trend, HW_mul_mul_dam.season, HW_mul_mul_dam.fittedvalues],
#                                 columns=['y_t','l_t','b_t','s_t','yhat_t'], index=y_input_add[0][:730].index)
# print('internal items of HoltWinters_730 HW_mul_mul_dam are:')
# print(df_HW_mul_mul_dam)
# HW_mul_mul_dam_residual = (HW_mul_mul_dam.forecast(steps_day) - y_input_add[0][730:]) / y_input_add[0][730:] * 100
# print('forecast and actual deviation ratio(%) of HoltWinters_730 HW_mul_mul_dam is:')
# print(HW_mul_mul_dam_residual)
#
# # print the models to select
# MASE_HW_add_add_dam = sum(abs((HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_add_mul_dam = sum(abs((HW_add_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_add_dam = sum(abs((HW_mul_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# MASE_HW_mul_mul_dam = sum(abs((HW_mul_mul_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values) / (y_input_add[0][730-1:730-1+steps_day].values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))
# if MASE_HW_add_add_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_add_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_add_dam))
#     print(HW_add_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_add_dam))
# if MASE_HW_add_mul_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_add_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_add_mul_dam))
#     print(HW_add_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_add_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_add_mul_dam))
# if MASE_HW_mul_add_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_add_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_add_dam))
#     print(HW_mul_add_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_add_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_add_dam))
# if MASE_HW_mul_mul_dam < 1:
#     print('HoltWinters_730 y_input_add，HW_mul_mul_dam可用，其MASE值为：{:.2f}，其预测值为：'.format(MASE_HW_mul_mul_dam))
#     print(HW_mul_mul_dam.forecast(steps_day))
# else:
#     print('HoltWinters_730 y_input_add，HW_mul_mul_dam不可用，其MASE值为：{:.2f}'.format(MASE_HW_mul_mul_dam))


# #########################################---与自定义平滑类模型对比---###########################################
# print('生产系统、自定义、statsmodels中平滑类模型对比：')
#
# ###################################---simple exponential smoothing compare, below---##################################
# print('\n','生产系统、自定义、statsmodels中simple exponential smoothing对比：')
# # fit models
# fit_SES_train = SimpleExpSmoothing(y_input_add[0][0:730]).fit(optimized=True, use_brute=False)
# SES_SM = test_smoothing_models.simple(list(y_input_add[0][0:730]), steps_day)
# SES_WU = wualgorithm.simple(list(y_input_add[0][0:730]), steps_day)
#
# # print figures
# plt.figure('730+28+compared SES y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# pd.concat([fit_SES_train.fittedvalues, fit_SES_train.forecast(steps_day)], ignore_index=True).rename('fit_SES_train')\
#     .plot(ax=ax_HoltWinters, color='r', legend=True)
# pd.concat([pd.Series(SES_SM['fittedvalues']), pd.Series(SES_SM['predict'])], ignore_index=True).rename('SES_SM')\
#     .plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(SES_WU['fittedvalues']), pd.Series(SES_WU['pred'])], ignore_index=True).rename('SES_WU')\
#     .plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print statistics data
# print('在训练集上，自定义简单平滑模型的RMSE与SES的RMSE之比为：{:.2f}%'.format(SES_SM['rmse'] / np.sqrt(fit_SES_train.sse/730) * 100))
# print('在训练集上，生产系统简单平滑模型的RMSE与SES的RMSE之比为：{:.2f}%'.format(SES_WU['rmse'] / np.sqrt(fit_SES_train.sse/730) * 100), '\n')
# print('在验证集上，自定义简单平滑模型与SES的加权MASE值为：{:.2f}'.format(sum(abs((np.array(SES_SM['predict']) - y_input_add[0][730:730+steps_day].values) / (fit_SES_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))))
# print('在验证集上，生产系统简单平滑模型与SES的加权MASE值为：{:.2f}'.format(sum(abs((np.array(SES_WU['pred']) - y_input_add[0][730:730+steps_day].values) / (fit_SES_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))))
# #########################################------------------------------------------------------------------------
#
# ######################################------Holt additive compare, below---------######################################
# print('\n','生产系统、自定义、statsmodels中Holt additive对比：')
# # fit models
# Holt_add_dam_train = Holt(y_input_add[0][0:730], exponential=False, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_add_SM = test_smoothing_models.linear(list(y_input_add[0][0:730]), steps_day)
# Holt_add_WU = wualgorithm.linear(list(y_input_add[0][0:730]), steps_day)
#
# # print figures
# plt.figure('730+28+compared Holt_add y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# pd.concat([Holt_add_dam_train.fittedvalues, Holt_add_dam_train.forecast(steps_day)], ignore_index=True).rename('Holt_add_dam_train').plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(Holt_add_SM['fittedvalues']), pd.Series(Holt_add_SM['predict'])], ignore_index=True).rename('Holt_add_SM').plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(Holt_add_WU['fittedvalues']), pd.Series(Holt_add_WU['pred'])], ignore_index=True).rename('Holt_add_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
# plt.show()
#
# # print statistics data
# print('在训练集上，自定义霍尔特加法模型的RMSE与Holt_add_dam的RMSE之比为：{:.2f}%'.format(Holt_add_SM['rmse'] / np.sqrt(Holt_add_dam_train.sse/730) * 100))
# print('在训练集上，生产系统霍尔特加法模型的RMSE与Holt_add_dam的RMSE之比为：{:.2f}%'.format(Holt_add_WU['rmse'] / np.sqrt(Holt_add_dam_train.sse/730) * 100), '\n')
# print('在验证集上，自定义霍尔特加法模型与Holt_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(Holt_add_SM['predict']) - y_input_add[0][730:730+steps_day].values) / (Holt_add_dam_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))))
# print('在验证集上，生产系统霍尔特加法模型与Holt_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(Holt_add_WU['pred']) - y_input_add[0][730:730+steps_day].values) / (Holt_add_dam_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))))
# #########################################--------------------------------------------------------------------
#
# ##########################################---Holt multiplicative compare, below---##########################################
# print('\n','生产系统、自定义、statsmodels中Holt multiplicative对比：')
# # fit models
# Holt_mul_dam_train = Holt(y_input_add[0][0:730], exponential=True, damped_trend=True).fit(damping_trend=0.99, optimized=True, use_brute=False)
# Holt_mul_SM = test_smoothing_models.double_mul(list(y_input_add[0][0:730]), steps_day)
#
# # print figures
# plt.figure('730+28+compared Holt_mul y_input_add', figsize=(20,10))
# ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
# ax_HoltWinters.set_ylabel("amount")
# ax_HoltWinters.set_xlabel("day")
# xlim = plt.gca().set_xlim(0, length[0]-1)
# pd.concat([Holt_mul_dam_train.fittedvalues, Holt_mul_dam_train.forecast(steps_day)], ignore_index=True).rename('Holt_mul_dam_train').plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(Holt_mul_SM['fittedvalues']), pd.Series(Holt_mul_SM['predict'])], ignore_index=True).rename('Holt_mul_SM').plot(ax=ax_HoltWinters, color='b', legend=True)
# plt.show()
#
# # print statistics data
# print('在训练集上，自定义霍尔特乘法模型的RMSE与Holt_mul_dam的RMSE之比为：{:.2f}%'.format(Holt_mul_SM['rmse'] / np.sqrt(Holt_mul_dam_train.sse/730) * 100), '\n')
# print('在验证集上，自定义霍尔特乘法模型与Holt_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(Holt_mul_SM['predict']) - y_input_add[0][730:730+steps_day].values) / (Holt_mul_dam_train.forecast(steps_day).values - y_input_add[0][730:730+steps_day].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(1, steps_day+1)))))))
# #########################################----------------------------------------------------------------------

##########################################---HoltWinters additive compare, below---##########################################
print('\n','生产系统、自定义、statsmodels中HoltWinters additive 730 对比：')
weights = []
for i in range(1, len(y_input_add[0][0:730]) + 1):
    weights.append(i / len(y_input_add[0][0:730]))
weights = np.array(weights) / sum(weights)

# fit models
period = 360
HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:730], seasonal_periods=period, trend='add', seasonal='add',
                                      damped_trend=True, initialization_method='known',
        initial_level=np.average(y_input_add[0][0:730], weights=weights),
        initial_trend=np.array((sum(y_input_add[0][0:730][int(np.ceil(len(y_input_add[0][0:730]) / 2)):]) - sum(y_input_add[0][0:730][:int(np.floor(len(y_input_add[0][0:730]) / 2))])) / (np.floor(len(y_input_add[0][0:730]) / 2)) ** 2),
        initial_seasonal=np.array(y_input_add[0][0:730][:len(y_input_add[0][0:730]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:730], weights=weights)
                                      ).\
    fit(damping_trend=0.98, use_boxcox=None, method='ls', use_brute=False)
# HWA_SM = test_smoothing_models.additive(list(y_input_add[0][0:730]), period, steps_day)
# HWA_SM2 = test2_smoothing_models.additive(list(y_input_add[0][0:730]), period, steps_day)
# HWMA_SM = test_smoothing_models.double_seasonal_add_add_day(list(y_input_add[0][0:730]), m_year=period, m_week=7, fc=steps_day)
HWA_WU = wualgorithm.additive(list(y_input_add[0][0:730]), period, steps_day)

# print figures
plt.figure(f'{len(y_input_add[0][0:730])}+{steps_day}+compared HW y_input_add[0]', figsize=(20,10))
ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
xlim = plt.gca().set_xlim(0, length[0]-1)
pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(steps_day)], ignore_index=True).rename('HW_add_add_dam').\
    plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(HWA_SM['fittedvalues']), pd.Series(HWA_SM['predict'])], ignore_index=True).rename('HWA_SM').\
#     plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(HWMA_SM['fittedvalues']), pd.Series(HWMA_SM['predict'])], ignore_index=True).rename('HWMA_SM').\
#     plot(ax=ax_HoltWinters, color='g', legend=True)
pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU').\
    plot(ax=ax_HoltWinters, color='y', legend=True)
plt.show()

# print statistics data
# print('在训练集上，自定义霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_SM['rmse'] / np.sqrt(HW_add_add_dam.sse/(730)) * 100))
# print('在训练集上，自定义霍尔特温特斯多重季节性加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWMA_SM['rmse'] / np.sqrt(HW_add_add_dam.sse/(730)) * 100))
print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(730)) * 100), '\n')
# print('在验证集上，自定义霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_SM['predict']) - y_input_add[0][730:730+steps_day+1].values) / (HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day+1].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(steps_day, 0, -1)))))))
# print('在验证集上，自定义霍尔特温特斯多重季节性加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWMA_SM['predict']) - y_input_add[0][730:730+steps_day+1].values) / (HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day+1].values))
#     * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(steps_day, 0, -1)))))))
print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - y_input_add[0][730:730+steps_day+1].values) / (HW_add_add_dam.forecast(steps_day).values - y_input_add[0][730:730+steps_day+1].values))
    * (np.array(range(steps_day, 0, -1)) / sum(np.array(range(steps_day, 0, -1)))))))

#########----------------------------------------------------------------------------------------------------------
print('\n','生产系统、自定义、statsmodels中HoltWinters additive 365 对比：')
weights = []
for i in range(1, len(y_input_add[0][0:365+1]) + 1):
    weights.append(i / len(y_input_add[0][0:365+1]))
weights = np.array(weights) / sum(weights)

# fit models
period = 360
HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:365+1], seasonal_periods=period, trend='add', seasonal='add',
                                      damped_trend=True, initialization_method='known',
        initial_level=np.average(y_input_add[0][0:365+1], weights=weights),
        initial_trend=np.array((sum(y_input_add[0][0:365+1][int(np.ceil(len(y_input_add[0][0:365+1]) / 2)):]) - sum(y_input_add[0][0:365+1][:int(np.floor(len(y_input_add[0][0:365+1]) / 2))])) / (np.floor(len(y_input_add[0][0:365+1]) / 2)) ** 2),
        initial_seasonal=np.array(y_input_add[0][0:365+1][:len(y_input_add[0][0:365+1]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:365+1], weights=weights)
                                      ).fit(damping_trend=0.98, use_boxcox=None, method='ls', use_brute=False)
# HWA_SM = test_smoothing_models.additive(list(y_input_add[0][0:365+1]), period, steps_day-1)
# HWA_SM2 = test2_smoothing_models.additive(list(y_input_add[0][0:365+1]), period, steps_day-1)
# HWMA_SM = test_smoothing_models.double_seasonal_add_add_day(list(y_input_add[0][0:365+1]), m_year=period, m_week=7, fc=28-1)
HWA_WU = wualgorithm.additive(list(y_input_add[0][0:365+1]), period, steps_day-1)

# print figures
plt.figure(f'{len(y_input_add[0][0:365+1])}+{steps_day-1}+compared HW y_input_add', figsize=(20,10))
ax_HoltWinters = y_input_add[0][:366+steps_day].rename('y_input_add[0][:366+steps_day]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
xlim = plt.gca().set_xlim(0, length[1]-1)
pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(steps_day-1)], ignore_index=True).rename('HW_add_add_dam')\
    .plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(HWA_SM['fittedvalues']), pd.Series(HWA_SM['predict'])], ignore_index=True).rename('HWA_SM')\
#     .plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(HWMA_SM['fittedvalues']), pd.Series(HWMA_SM['predict'][:-1])], ignore_index=True).rename('HWMA_SM')\
#     .plot(ax=ax_HoltWinters, color='g', legend=True)
pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU')\
    .plot(ax=ax_HoltWinters, color='y', legend=True)
plt.show()

# print statistics data
# print('在训练集上，自定义霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_SM['rmse'] / np.sqrt(HW_add_add_dam.sse/(365+1)) * 100))
# print('在训练集上，自定义霍尔特温特斯多重季节性加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWMA_SM['rmse'] / np.sqrt(HW_add_add_dam.sse/(365+1)) * 100))
print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(365+1)) * 100), '\n')
# print('在验证集上，自定义霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_SM['predict']) - y_input_add[0][365+1:365+1+steps_day-1].values) / (HW_add_add_dam.forecast(steps_day-1).values - y_input_add[0][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
# print('在验证集上，自定义霍尔特温特斯多重季节性加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWMA_SM['predict'][:-1]) - y_input_add[0][365+1:365+1+steps_day-1].values) / (HW_add_add_dam.forecast(steps_day-1).values - y_input_add[0][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - y_input_add[0][365+1:365+1+steps_day-1].values) / (HW_add_add_dam.forecast(steps_day-1).values - y_input_add[0][365+1:365+1+steps_day-1].values))
    * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
#########################################----------------------------------------------------------------------------

#####################################---HoltWinters multiplicative compare, below---###################################
print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
weights = []
for i in range(1, len(y_input_mul[0][0:730+1]) + 1):
    weights.append(i / len(y_input_mul[0][0:730+1]))
weights = np.array(weights) / sum(weights)

# fit models
HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:730+1], seasonal_periods=350, trend='add', seasonal='mul',
                                      damped_trend=True, initialization_method='known',
        initial_level=np.average(y_input_mul[0][0:730+1], weights=weights),
        initial_trend=np.array((sum(y_input_mul[0][0:730+1][int(np.ceil(len(y_input_mul[0][0:730+1]) / 2)):]) - sum(y_input_mul[0][0:730+1][:int(np.floor(len(y_input_mul[0][0:730+1]) / 2))])) / (np.floor(len(y_input_mul[0][0:730+1]) / 2)) ** 2),
        initial_seasonal=np.array(y_input_mul[0][0:730+1][:len(y_input_mul[0][0:730+1]) // 350 * 350]).reshape(-1, 350).mean(axis=0) - np.average(y_input_mul[0][0:730+1], weights=weights)
                                      ).\
    fit(damping_trend=0.98, use_boxcox=None, method='ls', use_brute=False)
# HWM_SM = test_smoothing_models.multiplicative(list(y_input_mul[0][0:730+1]), 350, steps_day-1)
# HWMM_SM = test_smoothing_models.multiseasonal_mul_day(list(y_input_mul[0][0:730+1]))
HWM_WU = wualgorithm.multiplicative(list(y_input_mul[0][0:730+1]), 350, steps_day-1)

# print figures
plt.figure(f'{len(y_input_mul[0][0:730+1])}+{steps_day-1}+compared HW y_input_mul', figsize=(20,10))
ax_HoltWinters = y_input_mul[0].rename('y_input_mul[0]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
xlim = plt.gca().set_xlim(0, length[0]-1)
pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(steps_day-1)], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(HWM_SM['fittedvalues']), pd.Series(HWM_SM['predict'])], ignore_index=True).rename('HWM_SM').plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(HWMM_SM['fittedvalues']), pd.Series(HWMM_SM['predict'][:-1])], ignore_index=True).rename('HWMM_SM').plot(ax=ax_HoltWinters, color='g', legend=True)
pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
plt.show()

# print statistics data
# print('在训练集上，自定义霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_SM['rmse'] / np.sqrt(HW_add_mul_dam.sse/(730+1)) * 100))
# print('在训练集上，自定义霍尔特温特斯多重季节性乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWMM_SM['rmse'] / np.sqrt(HW_add_mul_dam.sse/(730+1)) * 100))
print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(730+1)) * 100), '\n')
# print('在验证集上，自定义霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_SM['predict']) - y_input_mul[0][730+1:730+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][730+1:730+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
# print('在验证集上，自定义霍尔特温特斯多重季节性乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWMM_SM['predict'][:-1]) - y_input_mul[0][730+1:730+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][730+1:730+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_WU['pred']) - y_input_mul[0][730+1:730+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][730+1:730+1+steps_day-1].values))
    * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
#########################################---------------------------------------------------------------------------

print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
weights = []
for i in range(1, len(y_input_mul[0][0:365+1]) + 1):
    weights.append(i / len(y_input_mul[0][0:365+1]))
weights = np.array(weights) / sum(weights)

# fit models
HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:365+1], seasonal_periods=330, trend='add', seasonal='mul',
                                      damped_trend=True, initialization_method='known',
        initial_level=np.average(y_input_mul[0][0:365+1], weights=weights),
        initial_trend=np.array((sum(y_input_mul[0][0:365+1][int(np.ceil(len(y_input_mul[0][0:365+1]) / 2)):]) - sum(y_input_mul[0][0:365+1][:int(np.floor(len(y_input_mul[0][0:365+1]) / 2))])) / (np.floor(len(y_input_mul[0][0:365+1]) / 2)) ** 2),
        initial_seasonal=np.array(y_input_mul[0][0:365+1][:len(y_input_mul[0][0:365+1]) // 330 * 330]).reshape(-1, 330).mean(axis=0) - np.average(y_input_mul[0][0:365+1], weights=weights)
                                      ).fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
# HWM_SM = test_smoothing_models.multiplicative(list(y_input_mul[0][0:365+1]), 330, steps_day-1)
# HWMM_SM = test_smoothing_models.multiseasonal_mul_day(list(y_input_mul[0][0:365+1]))
HWM_WU = wualgorithm.multiplicative(list(y_input_mul[0][0:365+1]), 330, steps_day-1)

# print figures
plt.figure(f'{len(y_input_mul[0][0:365+1])}+{steps_day-1}+compared HW y_input_mul', figsize=(20,10))
ax_HoltWinters = y_input_mul[0][:366+steps_day].rename('y_input_mul[0][:366+steps_day]').plot(color='k', legend='True')
ax_HoltWinters.set_ylabel("amount")
ax_HoltWinters.set_xlabel("day")
xlim = plt.gca().set_xlim(0, length[1]-1)
pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(steps_day-1)], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
# pd.concat([pd.Series(HWM_SM['fittedvalues']), pd.Series(HWM_SM['predict'])], ignore_index=True).rename('HWM_SM').plot(ax=ax_HoltWinters, color='b', legend=True)
# pd.concat([pd.Series(HWMM_SM['fittedvalues']), pd.Series(HWMM_SM['predict'][:-1])], ignore_index=True).rename('HWMM_SM').plot(ax=ax_HoltWinters, color='g', legend=True)
pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
plt.show()

# print statistics data
# print('在训练集上，自定义霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_SM['rmse'] / np.sqrt(HW_add_mul_dam.sse/(365+1)) * 100))
# print('在训练集上，自定义霍尔特温特斯多重季节性乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWMM_SM['rmse'] / np.sqrt(HW_add_mul_dam.sse/(365+1)) * 100))
print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(365+1)) * 100), '\n')
# print('在验证集上，自定义霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_SM['predict']) - y_input_mul[0][365+1:365+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
# print('在验证集上，自定义霍尔特温特斯多重季节性乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWMM_SM['predict'][:-1]) - y_input_mul[0][365+1:365+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][365+1:365+1+steps_day-1].values))
#     * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_WU['pred']) - y_input_mul[0][365+1:365+1+steps_day-1].values) / (HW_add_mul_dam.forecast(steps_day-1).values - y_input_mul[0][365+1:365+1+steps_day-1].values))
    * (np.array(range(steps_day-1, 0, -1)) / sum(np.array(range(1, steps_day)))))))
