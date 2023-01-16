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


#################################---HoltWinters additive compare, below---##########################################
seg_day = '2022-10-1'
moving_holiday = False
category = 3  # 1：日序>182，非节假日，m=7；2：周序>52+shift，非节假日，m=52；
# 3：固定假日，先用日序测试，序列长度>365+shift，m=365；再加周序测试，序列长度>52+shift，m=52，观察周序对节假日偏移的影响有多大
period = 365
# 当序列长度在一至两个周期内，序列长度至少要给一个周期+shift，shift越大越好，特别是对于乘法模型；
# 更稳妥的做法是，不管序列长度是几倍周期，不管是哪种季节性模型，都+shift，即使序列长度>整数倍周期，而不是刚好等于整数倍周期。
shift = 70
steps_day = shift + 4*7
steps_week = shift + 4*1
length = [period*2 + steps_day, period + steps_day, period*2 + steps_week, period + steps_week]  # 代表每个序列的长度，分别为周、日序列的一年及两年。


acct_pred_com_stk.sort_values(by=['organ', 'class', 'sort', 'code', 'busdate'], ascending=True, inplace=True)
code_fitter = acct_pred_com_stk_grup.median()[acct_pred_com_stk_grup.median()['fix_amount'] > 0]['code']
acct_pred_com_stk_fitter = acct_pred_com_stk[acct_pred_com_stk['code'].isin(code_fitter)]


single_item = acct_pred_com_stk_fitter[acct_pred_com_stk_fitter['code'] == code_fitter.values[0]].reset_index()
sin_itm_amt = single_item['fix_amount']
train = single_item[single_item['busdate'] < seg_day][-(period*2+shift):]['fix_amount']
valid = single_item[~(single_item['busdate'] < seg_day)]['fix_amount']


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
    print('\n',f'生产系统、自定义、statsmodels中HoltWinters additive {period*2+shift} 对比：')
    weights = []
    for i in range(1, len(train) + 1):
        weights.append(i / len(train))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_add_dam = ExponentialSmoothing(train, seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                bounds={'smoothing_level': alpha, 'smoothing_trend': beta, 'smoothing_seasonal': gamma},
                initial_level=np.average(train, weights=weights),
                initial_trend=np.array((sum(train[int(np.ceil(len(train) / 2)):]) - sum(train[:int(np.floor(len(train) / 2))])) / (np.floor(len(train) / 2)) ** 2),
                initial_seasonal=np.array(train[:len(train) // period * period]).reshape(-1, period).mean(axis=0) - np.average(train, weights=weights)
                                              ).\
            fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_add_dam = ExponentialSmoothing(train, seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                bounds={'smoothing_level': alpha, 'smoothing_trend': beta, 'smoothing_seasonal': gamma},
                initial_level=np.average(train, weights=weights),
                initial_trend=np.array((sum(train[int(np.ceil(len(train) / 2)):]) - sum(train[:int(np.floor(len(train) / 2))])) / (np.floor(len(train) / 2)) ** 2),
                initial_seasonal=np.array(train[:len(train) // period * period]).reshape(-1, period).mean(axis=0) - np.average(train, weights=weights)
                                              ).\
            fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_add_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_add_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_add_dam.params["smoothing_seasonal"], 3)}')
    HWA_WU = wualgorithm.additive(list(train), period, len(valid))

    # print figures
    plt.figure(f'{len(train)}+{len(valid)}+compared HW train', figsize=(20,10))
    ax_HoltWinters = train.rename('train').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[0]-1)
    pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(len(valid))], ignore_index=True).rename('HW_add_add_dam').\
        plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU').\
        plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(period*2)) * 100))
    print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - train[period*2:period*2+len(valid)].values) / (HW_add_add_dam.forecast(len(valid)).values - train[period*2:period*2+len(valid)].values))
        * (np.array(range(len(valid), 0, -1)) / sum(np.array(range(len(valid), 0, -1)))))))

    #########----------------------------------------------------------------------------------------------------------
    print('\n',f'生产系统、自定义、statsmodels中HoltWinters additive {period} 对比：')
    weights = []
    for i in range(1, len(train[0:period+shift]) + 1):
        weights.append(i / len(train[0:period+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_add_dam = ExponentialSmoothing(train[0:period+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(train[0:period+shift], weights=weights),
                initial_trend=np.array((sum(train[0:period+shift][int(np.ceil(len(train[0:period+shift]) / 2)):]) - sum(train[0:period+shift][:int(np.floor(len(train[0:period+shift]) / 2))])) / (np.floor(len(train[0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(train[0:period+shift][:len(train[0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(train[0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_add_dam = ExponentialSmoothing(train[0:period+shift], seasonal_periods=period, trend='add', seasonal='add',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average(train[0:period+shift], weights=weights),
                initial_trend=np.array((sum(train[0:period+shift][int(np.ceil(len(train[0:period+shift]) / 2)):]) - sum(train[0:period+shift][:int(np.floor(len(train[0:period+shift]) / 2))])) / (np.floor(len(train[0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array(train[0:period+shift][:len(train[0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average(train[0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_add_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_add_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_add_dam.params["smoothing_seasonal"], 3)}')
    HWA_WU = wualgorithm.additive(list(train[0:period+shift]), period, len(valid))

    # print figures
    plt.figure(f'{len(train[0:period+shift])}+{len(valid)}+compared HW y_input_add', figsize=(20,10))
    ax_HoltWinters = train[:366+steps_day].rename('train[:366+steps_day]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[1]-1)
    pd.concat([HW_add_add_dam.fittedvalues, HW_add_add_dam.forecast(len(valid))], ignore_index=True).rename('HW_add_add_dam')\
        .plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWA_WU['fittedvalues']), pd.Series(HWA_WU['pred'])], ignore_index=True).rename('HWA_WU')\
        .plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯加法模型的RMSE与HW_add_add_dam的RMSE之比为：{:.2f}%'.format(HWA_WU['rmse'] / np.sqrt(HW_add_add_dam.sse/(period)) * 100))
    print('在验证集上，生产系统霍尔特温特斯加法模型与HW_add_add_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWA_WU['pred']) - sin_itm_amt[period:period+len(valid)].values) / (HW_add_add_dam.forecast(len(valid)).values - sin_itm_amt[period:period+len(valid)].values))
        * (np.array(range(len(valid), 0, -1)) / sum(np.array(range(1, len(valid))))))))
    #########################################----------------------------------------------------------------------------

    #####################################---HoltWinters multiplicative compare, below---###################################
    print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
    weights = []
    for i in range(1, len((train+1)[0:period*2+shift]) + 1):
        weights.append(i / len((train+1)[0:period*2+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_mul_dam = ExponentialSmoothing((train+1)[0:period*2+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average((train+1)[0:period*2+shift], weights=weights),
                initial_trend=np.array((sum((train+1)[0:period*2+shift][int(np.ceil(len((train+1)[0:period*2+shift]) / 2)):]) - sum((train+1)[0:period*2+shift][:int(np.floor(len((train+1)[0:period*2+shift]) / 2))])) / (np.floor(len((train+1)[0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array((train+1)[0:period*2+shift][:len((train+1)[0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average((train+1)[0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=0.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_mul_dam = ExponentialSmoothing((train+1)[0:period*2+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average((train+1)[0:period*2+shift], weights=weights),
                initial_trend=np.array((sum((train+1)[0:period*2+shift][int(np.ceil(len((train+1)[0:period*2+shift]) / 2)):]) - sum((train+1)[0:period*2+shift][:int(np.floor(len((train+1)[0:period*2+shift]) / 2))])) / (np.floor(len((train+1)[0:period*2+shift]) / 2)) ** 2),
                initial_seasonal=np.array((train+1)[0:period*2+shift][:len((train+1)[0:period*2+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average((train+1)[0:period*2+shift], weights=weights)
                                              ).\
            fit(damping_trend=0.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_mul_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_mul_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_mul_dam.params["smoothing_seasonal"], 3)}')
    HWM_WU = wualgorithm.multiplicative(list((train+1)[0:period*2+shift]), period, len(valid))

    # print figures
    plt.figure(f'{len((train+1)[0:period*2+shift])}+{len(valid)}+compared HW y_input_mul', figsize=(20,10))
    ax_HoltWinters = (train+1).rename('(train+1)').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[0]-1)
    pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(len(valid))], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.
          format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(period*2+shift)) * 100))
    print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.
          format(sum(abs((np.array(HWM_WU['pred']) - (sin_itm_amt+1)[period*2:period*2+len(valid)].values) /
                         (HW_add_mul_dam.forecast(len(valid)).values - (sin_itm_amt+1)[period*2:period*2+len(valid)].values))
        * (np.array(range(len(valid), 0, -1)) / sum(np.array(range(1, len(valid))))))))
    #########################################---------------------------------------------------------------------------

    print('\n','生产系统、自定义、statsmodels中HoltWinters multiplicative对比：')
    weights = []
    for i in range(1, len((train+1)[0:period+shift]) + 1):
        weights.append(i / len((train+1)[0:period+shift]))
    weights = np.array(weights) / sum(weights)

    # fit models
    try:
        HW_add_mul_dam = ExponentialSmoothing((train+1)[0:period+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average((train+1)[0:period+shift], weights=weights),
                initial_trend=np.array((sum((train+1)[0:period+shift][int(np.ceil(len((train+1)[0:period+shift]) / 2)):]) - sum((train+1)[0:period+shift][:int(np.floor(len((train+1)[0:period+shift]) / 2))])) / (np.floor(len((train+1)[0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array((train+1)[0:period+shift][:len((train+1)[0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average((train+1)[0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='ls', use_brute=False)
        n += 1
    except Exception as e:
        HW_add_mul_dam = ExponentialSmoothing((train+1)[0:period+shift], seasonal_periods=period, trend='add', seasonal='mul',
                                              damped_trend=True, initialization_method='known',
                                              bounds={'smoothing_level': alpha, 'smoothing_trend': beta,
                                                      'smoothing_seasonal': gamma},
                initial_level=np.average((train+1)[0:period+shift], weights=weights),
                initial_trend=np.array((sum((train+1)[0:period+shift][int(np.ceil(len((train+1)[0:period+shift]) / 2)):]) - sum((train+1)[0:period+shift][:int(np.floor(len((train+1)[0:period+shift]) / 2))])) / (np.floor(len((train+1)[0:period+shift]) / 2)) ** 2),
                initial_seasonal=np.array((train+1)[0:period+shift][:len((train+1)[0:period+shift]) // period * period]).reshape(-1, period).mean(axis=0) - np.average((train+1)[0:period+shift], weights=weights)
                                              ).fit(damping_trend=.98, use_boxcox=None, method='L-BFGS-B', use_brute=False)
        print(f'ls: {e}, 改用L-BFGS-B做梯度下降求解参数，此时目标函数为极大似然估计的形式')
    print(f'parameters (statsmodels): alpha {round(HW_add_mul_dam.params["smoothing_level"], 3)}, '
          f'beta {round(HW_add_mul_dam.params["smoothing_trend"], 3)}, '
          f'gamma {round(HW_add_mul_dam.params["smoothing_seasonal"], 3)}')
    HWM_WU = wualgorithm.multiplicative(list((train+1)[0:period+shift]), period, len(valid))

    # print figures
    plt.figure(f'{len((train+1)[0:period+shift])}+{len(valid)}+compared HW y_input_mul', figsize=(20,10))
    ax_HoltWinters = (train+1)[:366+steps_day].rename('(train+1)[:366+steps_day]').plot(color='k', legend='True')
    ax_HoltWinters.set_ylabel("amount")
    ax_HoltWinters.set_xlabel("day")
    xlim = plt.gca().set_xlim(0, length[1]-1)
    pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(len(valid))], ignore_index=True).rename('HW_add_mul_dam').plot(ax=ax_HoltWinters, color='red', legend=True)
    pd.concat([pd.Series(HWM_WU['fittedvalues']), pd.Series(HWM_WU['pred'])], ignore_index=True).rename('HWM_WU').plot(ax=ax_HoltWinters, color='y', legend=True)
    plt.show()

    # print statistics data
    print('在训练集上，生产系统霍尔特温特斯乘法模型的RMSE与HW_add_mul_dam的RMSE之比为：{:.2f}%'.format(HWM_WU['rmse'] / np.sqrt(HW_add_mul_dam.sse/(period+shift)) * 100))
    print('在验证集上，生产系统霍尔特温特斯乘法模型与HW_add_mul_dam的加权MASE值为：{:.2f}'.format(sum(abs((np.array(HWM_WU['pred']) - (sin_itm_amt+1)[period:period+len(valid)].values) / (HW_add_mul_dam.forecast(len(valid)).values - (sin_itm_amt+1)[period:period+len(valid)].values))
        * (np.array(range(len(valid), 0, -1)) / sum(np.array(range(1, len(valid))))))))

    print(f'\n使用ls的次数为：{n}\n占总训练次数的百分比：{round(n/4*100, 2)}')
else:
    # 移动假日不训练，只使用日序，配置参数和周期m
    pass


pd.concat([HW_add_mul_dam.fittedvalues, HW_add_mul_dam.forecast(len(valid))], ignore_index=True).rename('HW_add_mul_dam')
