'''
1. statsmodels中的Seasonal-Trend decomposition 无法直接预测时序，但可以通过将分解出的trend和/或seasonal分项继续分解，
得到规律性强、容易预测的子trend分项和/或子seasonal分项，对各子分项单独预测，再将各子分项的预测结果相加，得到最终的预测值。
需要注意的是，如果分解出的子分项太多，则残差项也会更多，则各子分项预测值相加得到的最终预测值通常会有更大的残差。
所以需要控制分解出的层级数目，找到提高子分项规律性与残差和增大的平衡点。
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from warnings import filterwarnings
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import seaborn as sns
sns.set_style('darkgrid')
plt.rc('figure',figsize=(19, 9))
plt.rc('font',size=10)
filterwarnings("ignore")

###########---------------set up and plot input data-----------------######################

up_limit = 20 # 设置level、trend、season项的取值上限
steps_day, steps_week = 28, 4
length = [730+steps_day, 365+steps_day, 104+steps_week, 52+steps_week] # 代表每个序列的长度，分别为周、日序列的一年及两年。
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
    y_input_add[i] = y_level[i] + y_trend[i] + y_season[i] + y_noise[i]  # 假定各项以加法方式组成输入数据

    y_level[i] = pd.Series(y_level[i]).rename('y_level')
    y_trend[i] = pd.Series(y_trend[i]).rename('y_trend')
    y_season[i] = pd.Series(y_season[i]).rename('y_season')
    y_noise[i] = pd.Series(y_noise[i]).rename('y_noise')
    y_input_add[i] = pd.Series(y_input_add[i]).rename('y_input_add')
    y_input_add[i][y_input_add[i] < 0] = 0

# 绘制四条加法季节性时间序列
plt.figure('add: 730+28')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_add[0].plot(ax=ax1, legend=True)
y_level[0].plot(ax=ax2, legend=True)
y_trend[0].plot(ax=ax3, legend=True)
y_season[0].plot(ax=ax4, legend=True)
y_noise[0].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_add[0], period=365, seasonal=7, trend=(730+steps_day) + (((730+steps_day) % 2) == 0)).fit().plot()
plt.show()

# Seasonal decomposition using moving averages
# 先分解小周期，则seasonal项具有强规律性，即可对trend项进行再分解
sdma_1 = seasonal_decompose(y_input_add[2][0:104], model='add', period=4, two_sided=False)
sdma_1.plot()
plt.show()

sdma_trend_1 = sdma_1.trend.dropna().reset_index(drop=True)
sdma_2 = seasonal_decompose(sdma_trend_1, model='add', period=13, two_sided=False)
sdma_2.plot()
plt.show()

sdma_trend_2 = sdma_2.trend.dropna().reset_index(drop=True)

forecast = sdma_1.seasonal[:steps_week].values + sdma_2.seasonal[:steps_week].values + ExponentialSmoothing(sdma_trend_2, trend='add', seasonal='add', seasonal_periods=26).fit().forecast(steps_week).values

# 先分解大周期，则trend项具有强规律性，即可对seasonal项进行再分解
sdma_1 = seasonal_decompose(y_input_add[2][0:104], model='add', period=52, two_sided=False)
sdma_1.plot()
plt.show()

sdma_seasonal_1 = sdma_1.seasonal.dropna().reset_index(drop=True)
sdma_2 = seasonal_decompose(sdma_seasonal_1, model='add', period=13, two_sided=False)
sdma_2.plot()
plt.show()

sdma_trend_2 = sdma_2.seasonal.dropna().reset_index(drop=True)


plt.figure('add: 365+28')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_add[1].plot(ax=ax1, legend=True)
y_level[1].plot(ax=ax2, legend=True)
y_trend[1].plot(ax=ax3, legend=True)
y_season[1].plot(ax=ax4, legend=True)
y_noise[1].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_add[1], period=365, seasonal=7, trend=(365+28) + (((365+28) % 2) == 0)).fit().plot()
plt.show()

plt.figure('add: 104+4')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[2]-1)
y_input_add[2].plot(ax=ax1, legend=True)
y_level[2].plot(ax=ax2, legend=True)
y_trend[2].plot(ax=ax3, legend=True)
y_season[2].plot(ax=ax4, legend=True)
y_noise[2].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_add[2], period=52, seasonal=int(52/4), trend=(104+4) + (((104+4) % 2) == 0)).fit().plot()
plt.show()

plt.figure('add: 52+4')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[3])
y_input_add[3].plot(ax=ax1, legend=True)
y_level[3].plot(ax=ax2, legend=True)
y_trend[3].plot(ax=ax3, legend=True)
y_season[3].plot(ax=ax4, legend=True)
y_noise[3].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_add[3], period=52, seasonal=int(52/4), trend=(52+4) + (((52+4) % 2) == 0)).fit().plot()
plt.show()

######################################################################

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
    y_input_mul[i] = (y_level[i] + y_trend[i]) * y_season[i] * y_noise[i]  # 假定季节项以乘法方式组成输入数据

    y_level[i] = pd.Series(y_level[i]).rename('y_level')
    y_trend[i] = pd.Series(y_trend[i]).rename('y_trend')
    y_season[i] = pd.Series(y_season[i]).rename('y_season')
    y_noise[i] = pd.Series(y_noise[i]).rename('y_noise')
    y_input_mul[i] = pd.Series(y_input_mul[i]).rename('y_input_mul')
    y_input_mul[i][y_input_mul[i] < 0] = 0

# 绘制四条乘法季节性时间序列
plt.figure('mul: 730+28')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_mul[0].plot(ax=ax1, legend=True)
y_level[0].plot(ax=ax2, legend=True)
y_trend[0].plot(ax=ax3, legend=True)
y_season[0].plot(ax=ax4, legend=True)
y_noise[0].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_mul[0], period=365, seasonal=7, trend=(730+28) + (((730+28) % 2) == 0)).fit().plot()
plt.show()

plt.figure('mul: 365+28')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_mul[1].plot(ax=ax1, legend=True)
y_level[1].plot(ax=ax2, legend=True)
y_trend[1].plot(ax=ax3, legend=True)
y_season[1].plot(ax=ax4, legend=True)
y_noise[1].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_mul[1], period=365, seasonal=7, trend=(365+28) + (((365+28) % 2) == 0)).fit().plot()
plt.show()

plt.figure('mul: 104+4')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[2]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[2]-1)
y_input_mul[2].plot(ax=ax1, legend=True)
y_level[2].plot(ax=ax2, legend=True)
y_trend[2].plot(ax=ax3, legend=True)
y_season[2].plot(ax=ax4, legend=True)
y_noise[2].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_mul[2], period=52, seasonal=int(52/4), trend=(104+4) + (((104+4) % 2) == 0)).fit().plot()
plt.show()

plt.figure('mul: 52+4')
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[3]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[3]-1)
y_input_mul[3].plot(ax=ax1, legend=True)
y_level[3].plot(ax=ax2, legend=True)
y_trend[3].plot(ax=ax3, legend=True)
y_season[3].plot(ax=ax4, legend=True)
y_noise[3].plot(ax=ax5, legend=True)
plt.show()

# 绘制STL分解序列；period, seasonal, trend must be odd.
STL(y_input_mul[3], period=52, seasonal=int(52/4), trend=(52+4) + (((52+4) % 2) == 0)).fit().plot()
plt.show()
