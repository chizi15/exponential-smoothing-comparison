# #!/usr/bin/python
# -*- coding: utf-8 -*-

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
import chinese_calendar as calendar
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 20)


# read and summerize data
account = pd.read_csv("D:\Work info\WestUnion\data\origin\DH\\C-acct-revised.csv")
account['busdate'] = pd.to_datetime(account['busdate'], infer_datetime_format=True)
account['code'] = account['code'].astype('str')
acct_grup = account.groupby(["organ", "code"])
print(f'\naccount\n\nshape: {account.shape}\n\ndtypes:\n{account.dtypes}\n\nisnull-columns:\n{account.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(account.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_grup)}\n')
pred = pd.read_csv("D:\Work info\WestUnion\data\origin\DH\prediction.csv")
pred['busdate'] = pd.to_datetime(pred['busdate'])
pred['code'] = pred['code'].astype('str')
pred_grup = pred.groupby(["organ", "code"])
print(f'\npred\n\nshape: {pred.shape}\n\ndtypes:\n{pred.dtypes}\n\nisnull-columns:\n{pred.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(pred.isnull().T.any())}\n\nnumber of commodities:\n{len(pred_grup)}\n')
commodity = pd.read_csv("D:\Work info\WestUnion\data\origin\DH\commodity.csv")
commodity[['class', 'sort']] = commodity[['class', 'sort']].astype('str')
comodt_grup = commodity.groupby(['code'])
print(f'\ncommodity\n\nshape: {commodity.shape}\n\ndtypes:\n{commodity.dtypes}\n\nisnull-columns:\n{commodity.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(commodity.isnull().T.any())}\n\nnumber of commodities:\n{len(comodt_grup)}\n')
stock = pd.read_csv("D:\Work info\WestUnion\data\origin\DH\stock.csv")
stock['busdate'] = pd.to_datetime(stock['busdate'])
stock['code'] = stock['code'].astype('str')
stock_grup = stock.groupby(["organ", "code"])
print(f'\nstock\n\nshape: {stock.shape}\n\ndtypes:\n{stock.dtypes}\n\nisnull-columns:\n{stock.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(stock.isnull().T.any())}\n\nnumber of commodities:\n{len(stock_grup)}\n')
# running = pd.read_csv("D:\Work info\WestUnion\data\origin\DH\\running.csv",
#                       parse_dates=['selldate'], dtype={'code': str})
# running['selltime'] = running['selltime'].apply(lambda x: x[:8])  # 截取出时分秒
# running['selltime'] = pd.to_datetime(running['selltime'], format='%H:%M:%S')
# running['selltime'] = running['selltime'].dt.time  # 去掉to_datetime自动生成的年月日
# run_grup = running.groupby(["organ", "code"])
# print(f'\nrunning\n\nshape: {running.shape}\n\ndtypes:\n{running.dtypes}\n\nisnull-columns:\n{running.isnull().any()}'
#       f'\n\nisnull-rows:\n{sum(running.isnull().T.any())}\n\nnumber of commodities:\n{len(run_grup)}\n')

# merge data and generate others
account['weekday'] = account['busdate'].apply(lambda x: x.weekday() + 1)  # the label of Monday is 0, so +1
df = pd.DataFrame(list(account['busdate'].apply(lambda x: calendar.get_holiday_detail(x))),
                  columns=['is_holiday', 'hol_type'])  # (True, None) is weekend, (False, 某节日)是指当天因某日调休而补班
print(f'\ndf\n\nshape: {df.shape}\n\ndtypes:\n{df.dtypes}\n\nisnull-columns:\n{df.isnull().any()}'
      f'\n\nisnull-rows, i.e. the number of rows of non-holiday:\n{sum(df.isnull().T.any())}\n')
if sum(df.isnull().T.any()) > 0:
    df.loc[df.isnull().T.any(), 'hol_type'] = '0'  # 将非节假日标为0
    print(f'\ndf\n\nshape: {df.shape}\n\ndtypes:\n{df.dtypes}\n\nisnull-columns:\n{df.isnull().any()}'
          f'\n\nisnull-rows, i.e. the number of rows of non-holiday:\n{sum(df.isnull().T.any())}\n')
account = pd.concat([account, df], axis=1)

acct_pred = pd.merge(account, pred, how='left', on=['organ', 'code', 'busdate'])
acct_pred_grup = acct_pred.groupby(["organ", "code"], as_index=False)
print(f'\nacct_pred\n\nshape: {acct_pred.shape}\n\ndtypes:\n{acct_pred.dtypes}\n\nisnull-columns:\n{acct_pred.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_grup)}\n')
acct_pred_com = pd.merge(acct_pred, commodity, how='inner', on=['code'])
acct_pred_com_grup = acct_pred_com.groupby(["organ", "code"], as_index=False)
print(f'\nacct_pred_com\n\nshape: {acct_pred_com.shape}\n\ndtypes:\n{acct_pred_com.dtypes}\n\nisnull-columns:\n{acct_pred_com.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred_com.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_com_grup)}\n')
acct_pred_com_stk = pd.merge(acct_pred_com, stock, how='left', on=['organ', 'code', 'busdate'])
acct_pred_com_stk_grup = acct_pred_com_stk.groupby(["organ", "code", 'class', 'sort'], as_index=False)
print(f'\nacct_pred_com_stk\n\nshape: {acct_pred_com_stk.shape}\n\ndtypes:\n{acct_pred_com_stk.dtypes}\n\nisnull-columns:\n{acct_pred_com_stk.isnull().any()}'
      f'\n\nisnull-rows:\n{sum(acct_pred_com_stk.isnull().T.any())}\n\nnumber of commodities:\n{len(acct_pred_com_stk_grup)}\n')
# running_grup = running.groupby(['organ', 'code', 'selldate'], as_index=False).sum()
# running_grup.rename(columns={'selldate': 'busdate', 'amount': 'amount_run', 'sum_sell': 'sum_price_run'}, inplace=True)


code_fitter = acct_pred_com_stk_grup.median()[acct_pred_com_stk_grup.median()['fix_amount'] > 0]['code']
acct_pred_com_stk_fitter = acct_pred_com_stk[acct_pred_com_stk['code'].isin(code_fitter)]
acct_pred_com_stk_fitter.groupby(["organ", "code", 'class', 'sort']).groups
acct_pred_com_stk_fitter.groupby(["organ", "code", 'class', 'sort']).get_group()


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
