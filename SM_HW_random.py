# #!/usr/bin/python
# -*- coding: utf-8 -*-
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

14. 乘法模型比加法模型稍慢，而且可能训练不收敛：SVD did not converge

15. 总结：从精度、开销、稳定性、全面性综合考虑，statsmodels中平滑类模型比生产系统和自定义平滑类模型更好。
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
import sys
sys.path.append("D:\Work info\Repositories\exponential-smoothing-using-statsmodels-master\exponential-smoothing-using-statsmodels-master\wualgorithm.py")
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
##################################################################################################################


#################################---HoltWinters additive compare, below---##########################################
moving_holiday = False
category = 1  # 1：日序>182，非节假日，m=7；2：周序>52+shift，非节假日，m=52；
# 3：固定假日，先用日序测试，序列长度>365+shift，m=365；再加周序测试，序列长度>52+shift，m=52，观察周序对节假日偏移的影响有多大
period = 365
# 当序列长度在一至两个周期内，序列长度至少要给一个周期+shift，shift越大越好，特别是对于乘法模型；
# 更稳妥的做法是，不管序列长度是几倍周期，不管是哪种季节性模型，都+shift，即使序列长度>整数倍周期，而不是刚好等于整数倍周期。
shift = 7
if moving_holiday == False:
    if category == 1:  # 日序，非节假日
        alpha, beta, gamma = (0, 1/10), (0, 1/10), (0, 1/2)
    elif category == 2:  # 周序，非节假日
        alpha, beta, gamma = (0, 1/5), (0, 1/5), (0, 2/3)
    elif category == 3:  # 固定假日，暂时只用日序
        alpha, beta, gamma = (0, 1), (0, 1), (0, 1)
    else:
        raise ValueError('类型category只有三种取值：1,2,3')
    n = 0  # 记录使用least square做目标函数和求解成功的次数
    print('\n',f'生产系统、自定义、statsmodels中HoltWinters additive {period*2} 对比：')
    weights = []
    for i in range(1, len(y_input_add[0][0:period*2+shift]) + 1):
        weights.append(i / len(y_input_add[0][0:period*2+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:period*2+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                bounds={'smoothing_level': alpha, 'smoothing_trend': beta, 'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_add[0][0:period*2+shift], weights=weights),
                initial_trend=np.array((sum(y_input_add[0][0:period*2+shift][int(np.ceil(len(y_input_add[0][0:period*2+shift]) / 2)):]) - sum(y_input_add[0][0:period*2+shift][:int(np.floor(len(y_input_add[0][0:period*2+shift]) / 2))])) / (np.floor(len(y_input_add[0][0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_add[0][0:period*2+shift][:len(y_input_add[0][0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:period*2+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                bounds={'smoothing_level': alpha, 'smoothing_trend': beta, 'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_add[0][0:period*2+shift], weights=weights),
                initial_trend=np.array((sum(y_input_add[0][0:period*2+shift][int(np.ceil(len(y_input_add[0][0:period*2+shift]) / 2)):]) - sum(y_input_add[0][0:period*2+shift][:int(np.floor(len(y_input_add[0][0:period*2+shift]) / 2))])) / (np.floor(len(y_input_add[0][0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_add[0][0:period*2+shift][:len(y_input_add[0][0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_add_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_add_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_add_dam.params["smoothing_seasonal"], 3)}')
    HWA_WU = wualgorithm.additive(list(y_input_add[0][0:period*2+shift]), period, steps_day-shift)

    # print figures
    plt.figure(f'{len(y_input_add[0][0:period*2+shift])}+{steps_day-shift}+compared HW y_input_add[0]', figsize=(20,10))
    ax_HoltWinters = y_input_add[0].rename('y_input_add[0]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[0]-1)
    pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(steps_day-shift)], ignore_index=True).rename('HW_add_add_dam').\
        plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU').\
        plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(period*2)) * 100))
    print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - y_input_add[0][period*2:period*2+steps_day-shift].values) / (HW_add_add_dam.forecast(steps_day-shift).values - y_input_add[0][period*2:period*2+steps_day-shift].values))
        * (np.array(range(steps_day-shift, 0, -1)) / sum(np.array(range(steps_day-shift, 0, -1)))))))

    #########----------------------------------------------------------------------------------------------------------
    print('\n',f'生产系统、自定义、statsmodels中HoltWinters additive {period} 对比：')
    weights = []
    for i in range(1, len(y_input_add[0][0:period+shift]) + 1):
        weights.append(i / len(y_input_add[0][0:period+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:period+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_add[0][0:period+shift], weights=weights),
                initial_trend=np.array((sum(y_input_add[0][0:period+shift][int(np.ceil(len(y_input_add[0][0:period+shift]) / 2)):]) - sum(y_input_add[0][0:period+shift][:int(np.floor(len(y_input_add[0][0:period+shift]) / 2))])) / (np.floor(len(y_input_add[0][0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_add[0][0:period+shift][:len(y_input_add[0][0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_add_dam = ExponentialSmoothing(y_input_add[0][0:period+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_add[0][0:period+shift], weights=weights),
                initial_trend=np.array((sum(y_input_add[0][0:period+shift][int(np.ceil(len(y_input_add[0][0:period+shift]) / 2)):]) - sum(y_input_add[0][0:period+shift][:int(np.floor(len(y_input_add[0][0:period+shift]) / 2))])) / (np.floor(len(y_input_add[0][0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_add[0][0:period+shift][:len(y_input_add[0][0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_add[0][0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_add_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_add_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_add_dam.params["smoothing_seasonal"], 3)}')
    HWA_WU = wualgorithm.additive(list(y_input_add[0][0:period+shift]), period, steps_day-shift)

    # print figures
    plt.figure(f'{len(y_input_add[0][0:period+shift])}+{steps_day-shift}+compared HW y_input_add', figsize=(20,10))
    ax_HoltWinters = y_input_add[0][:366+steps_day].rename('y_input_add[0][:366+steps_day]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[1]-1)
    pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(steps_day-shift)], ignore_index=True).rename('HW_add_add_dam')\
        .plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU')\
        .plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(period)) * 100))
    print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - y_input_add[0][period:period+steps_day-shift].values) / (HW_add_add_dam.forecast(steps_day-shift).values - y_input_add[0][period:period+steps_day-shift].values))
        * (np.array(range(steps_day-shift, 0, -1)) / sum(np.array(range(1, steps_day-shift)))))))
    #########################################----------------------------------------------------------------------------

    #####################################---HoltWinters multiplicative compare, below---###################################
    print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
    weights = []
    for i in range(1, len(y_input_mul[0][0:period*2+shift]) + 1):
        weights.append(i / len(y_input_mul[0][0:period*2+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:period*2+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_mul[0][0:period*2+shift], weights=weights),
                initial_trend=np.array((sum(y_input_mul[0][0:period*2+shift][int(np.ceil(len(y_input_mul[0][0:period*2+shift]) / 2)):]) - sum(y_input_mul[0][0:period*2+shift][:int(np.floor(len(y_input_mul[0][0:period*2+shift]) / 2))])) / (np.floor(len(y_input_mul[0][0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_mul[0][0:period*2+shift][:len(y_input_mul[0][0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_mul[0][0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=0.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:period*2+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_mul[0][0:period*2+shift], weights=weights),
                initial_trend=np.array((sum(y_input_mul[0][0:period*2+shift][int(np.ceil(len(y_input_mul[0][0:period*2+shift]) / 2)):]) - sum(y_input_mul[0][0:period*2+shift][:int(np.floor(len(y_input_mul[0][0:period*2+shift]) / 2))])) / (np.floor(len(y_input_mul[0][0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_mul[0][0:period*2+shift][:len(y_input_mul[0][0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_mul[0][0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=0.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_mul_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_mul_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_mul_dam.params["smoothing_seasonal"], 3)}')
    HWM_WU = wualgorithm.multiplicative(list(y_input_mul[0][0:period*2+shift]), period, steps_day-shift)

    # print figures
    plt.figure(f'{len(y_input_mul[0][0:period*2+shift])}+{steps_day-shift}+compared HW y_input_mul', figsize=(20,10))
    ax_HoltWinters = y_input_mul[0].rename('y_input_mul[0]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[0]-1)
    pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(steps_day-shift)], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(period*2)) * 100))
    print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_WU['pred']) - y_input_mul[0][period*2:period*2+steps_day-shift].values) / (HW_add_mul_dam.forecast(steps_day-shift).values - y_input_mul[0][period*2:period*2+steps_day-shift].values))
        * (np.array(range(steps_day-shift, 0, -1)) / sum(np.array(range(1, steps_day-shift)))))))
    #########################################---------------------------------------------------------------------------

    print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
    weights = []
    for i in range(1, len(y_input_mul[0][0:period+shift]) + 1):
        weights.append(i / len(y_input_mul[0][0:period+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:period+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_mul[0][0:period+shift], weights=weights),
                initial_trend=np.array((sum(y_input_mul[0][0:period+shift][int(np.ceil(len(y_input_mul[0][0:period+shift]) / 2)):]) - sum(y_input_mul[0][0:period+shift][:int(np.floor(len(y_input_mul[0][0:period+shift]) / 2))])) / (np.floor(len(y_input_mul[0][0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_mul[0][0:period+shift][:len(y_input_mul[0][0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_mul[0][0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_mul_dam = ExponentialSmoothing(y_input_mul[0][0:period+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(y_input_mul[0][0:period+shift], weights=weights),
                initial_trend=np.array((sum(y_input_mul[0][0:period+shift][int(np.ceil(len(y_input_mul[0][0:period+shift]) / 2)):]) - sum(y_input_mul[0][0:period+shift][:int(np.floor(len(y_input_mul[0][0:period+shift]) / 2))])) / (np.floor(len(y_input_mul[0][0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(y_input_mul[0][0:period+shift][:len(y_input_mul[0][0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(y_input_mul[0][0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_mul_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_mul_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_mul_dam.params["smoothing_seasonal"], 3)}')
    HWM_WU = wualgorithm.multiplicative(list(y_input_mul[0][0:period+shift]), period, steps_day-shift)

    # print figures
    plt.figure(f'{len(y_input_mul[0][0:period+shift])}+{steps_day-shift}+compared HW y_input_mul', figsize=(20,10))
    ax_HoltWinters = y_input_mul[0][:366+steps_day].rename('y_input_mul[0][:366+steps_day]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[1]-1)
    pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(steps_day-shift)], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(period)) * 100))
    print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_WU['pred']) - y_input_mul[0][period:period+steps_day-shift].values) / (HW_add_mul_dam.forecast(steps_day-shift).values - y_input_mul[0][period:period+steps_day-shift].values))
        * (np.array(range(steps_day-shift, 0, -1)) / sum(np.array(range(1, steps_day-shift)))))))

    print(f'\n使用ls的次数为：{n}')
else:
    # 移动假日不训练，只使用日序，配置参数和周期m
    pass
