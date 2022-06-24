# coding:utf-8

from __future__ import division
import numpy as np
import pandas as pd
from sys import exit
from math import log, exp
from numpy import array
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

from errorcalculation import rmse as Rmse
from errorcalculation import aic as Aic
from errorcalculation import ds as Ds
from errorcalculation import stde as Stde

"""
def outlier_dealing(timeseries):
    '''
        异常处理函数
        parameters:
            --------
            timeseries:时间序列数据或者普通列表数据

        return:
            --------
            timeseries.values:处理后序列值
    '''
    timeseries = timeseries.copy()
    if isinstance(timeseries, pd.Series):
        timeseries.index = range(len(timeseries))  # 改变索引为从0开始的数值索引
    else:
        timeseries = pd.Series(timeseries, index=range(len(timeseries)))

    flag = True
    loop = 0
    # while flag:
    while loop < 4:
        ts_diff = timeseries.diff()  # 差分序列
        ts_diff.dropna(inplace=True)
        ts_diff.index = range(len(ts_diff))

        ts_diff_mean = ts_diff.mean()  # 差分均值
        ts_diff_std = ts_diff.std()   # 标准差

        # 上下区间
        st_up = ts_diff_mean + 1.96 * ts_diff_std
        st_down = ts_diff_mean - 1.96 * ts_diff_std
        # print('均值及标准差:',ts_diff_mean, ts_diff_std)
        # print('上下区间:',st_up,st_down)

        flag = False
        for idx in range(len(ts_diff) - 1):

            if idx == 0:
                if ts_diff.loc[idx] > st_up or ts_diff.loc[idx] < st_down:
                    # 第一个差分点出判定区，对应原序列的第一个点就是异常点
                    timeseries.loc[idx] = np.mean(timeseries.values[1:]) # 后面N周的均值

            if ts_diff.loc[idx] > st_up or ts_diff.loc[idx] < st_down:
                if ts_diff.loc[idx + 1] > st_up or ts_diff.loc[idx + 1] < st_down:
                    # 连续两个点出判定区间，则第一个点是异常点
                    # idx为差分异常点索引,对应原序列索引加1
                    timeseries.loc[idx + 1] = np.nan  # 原序列异常点用空值代替
                    flag = True  # 改变标记值

        timeseries.interpolate(inplace=True)  # 插值填充原序列
        loop += 1
        # print('第 {0} 次循环,可能还要再次循环...'.format(loop))
    print('循环结束了,总共进行了 {0} 次循环..'.format(loop))
    return list(timeseries.values)
"""

def outlier_dealing(timeseries):
    '''
        异常处理函数
        parameters:
            --------
            timeseries:时间序列数据或者普通列表数据

        return:
            --------
            timeseries.values:处理后序列值
    '''
    timeseries = timeseries.copy()
    if isinstance(timeseries, pd.Series):
        timeseries.index = range(len(timeseries))  # 改变索引为从0开始的数值索引
    else:
        timeseries = pd.Series(timeseries, index=range(len(timeseries)))

    flag = True
    loop = 0
    # while flag:
    while loop < 4:
        ts_diff = timeseries.diff()  # 差分序列
        ts_diff.dropna(inplace=True)
        ts_diff.index = range(len(ts_diff))

        ts_diff_mean = ts_diff.mean()  # 差分均值
        ts_diff_std = ts_diff.std()   # 标准差

        # 上下区间
        st_up = ts_diff_mean + 1.96 * ts_diff_std
        st_down = ts_diff_mean - 1.96 * ts_diff_std
        # print('均值及标准差:',ts_diff_mean, ts_diff_std)
        # print('上下区间:',st_up,st_down)

        # 判断差分序列是否出判定区间 # 连续两个点出判定区间，则第一个点是异常点
        diff_out_range = ts_diff.apply(lambda x: True if (x > st_up) or (x < st_down) else False)
        # 上偏移一位
        diff_out_range_shift = diff_out_range.shift(-1)
        # 两series 横向concat, columns 为[0,1]
        concat_df = pd.concat([diff_out_range, diff_out_range_shift], axis=1)
        # 筛选出差分序列异常点索引, 两列同时为True则为异常点,对应原序列索引加1
        outlier_sr = concat_df.apply(lambda x: True if x[0] and x[1] else False, axis=1)
        outlier_idx = outlier_sr[outlier_sr == True].index
        # 原序列异常点用空值代替
        timeseries.loc[[(li+1) for li in outlier_idx]] = np.nan

        # 差分序列第一个点为异常点,但第二个点不是异常点,那么原始序列第一个点也就是异常点
        # 用原始序列除开第一个点的均值代替
        if diff_out_range[0] and not diff_out_range[1]:
            timeseries.loc[0] = np.nanmean(timeseries.values[1:])

        timeseries.interpolate(inplace=True)  # 插值填充原序列
        loop += 1
        # print('第 {0} 次循环,可能还要再次循环...'.format(loop))
    print('循环结束了,总共进行了 {0} 次循环..'.format(loop))
    return list(timeseries.values)


# holtwinters模型
# 计算均方根误差,即观测值与预测值之差的平方,求和再平均,再开方
def RMSE(params, *args):
    '''
        计算均方根误差
        parameters:
            --------
            params:包含alpha,beta,gamma三者或其中之一的初始值元组
            args:包含原始计算序列与type的元组

        return:
            --------
            rmse:均方根误差
    '''

    Y = args[0]
    type = args[1]
    rmse = 0

    if type == 'simple':
        alpha = params[0]
        a = [Y[0]]
        y = [a[0]]

        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i])
            y.append(a[i + 1])

    elif type == 'linear':

        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]

        for i in range(len(Y)):

            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])

    else:

        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

        if type == 'additive':

            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]

            for i in range(len(Y)):

                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])

        elif type == 'multiplicative':

            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]

            for i in range(len(Y)):

                a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y.append((a[i + 1] + b[i + 1]) * s[i + 1])

        else:

            exit('Type must be either linear, additive or multiplicative')

    # rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    rmse = Rmse(Y, y[:-1])

    return rmse


def simple(x, fc, alpha=None):
    '''
        单参数简单指数平滑预测模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha:alpha参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            rmse
            aic
            ds
            prestd:残差标准差
    '''

    Y = x[:]
    if alpha is None:
        initial_values = array([0.1])
        boundaries = [(0, 1)]
        type = 'simple'

        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type), bounds=boundaries, approx_grad=True)
        alpha = parameters[0][0]

    print(alpha)
    a = [Y[0]]  # 求和
    y = [a[0]]  # 预测列
    rmse = 0

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1])

        a.append(alpha * Y[i] + (1 - alpha) * a[i])
        y.append(a[i + 1])

    # rmse = sqrt(sum([(m - n) ** 2 for m,n in zip(Y[:-fc],y[:-fc-1])]) / len(Y[:-fc]))
    rmse = Rmse(Y[:-fc], y[:-fc - 1])
    aic = Aic(Y[:-fc], y[:-fc - 1], 1)
    ds = Ds(Y[:-fc], y[:-fc - 1])
    prestd = Stde([(m - n) for m, n in zip(Y[:-fc], y[:-fc - 1])])

    return {'pred': Y[-fc:], 'alpha': alpha, 'rmse': rmse, 'aic':aic, 'ds':ds, 'prestd':prestd, 'fittedvalues': y[:-fc-1]}


def linear(x, fc, alpha=None, beta=None):
    '''
        双参数简单指数平滑预测模型
        parameters:
            --------
            x:时间序列数据列表
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始默认值为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            rmse
            aic
            ds
            prestd:残差标准差
    '''

    Y = x[:]

    if (alpha is None or beta is None):

        initial_values = array([0.3, 0.1])
        boundaries = [(0, 1), (0, 1)]
        type = 'linear'

        # fmin_l_bfgs_b为迭代函数,接受一个函数参数,返回使得该函数的结果最优的数据
        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type), bounds=boundaries, approx_grad=True)
        alpha, beta = parameters[0]

    print(alpha, beta)
    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] + b[-1])

        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])

    # rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    rmse = Rmse(Y[:-fc], y[:-fc - 1])
    aic = Aic(Y[:-fc], y[:-fc - 1], 2)
    ds = Ds(Y[:-fc], y[:-fc - 1])
    prestd = Stde([(m - n) for m, n in zip(Y[:-fc], y[:-fc - 1])])

    return {'pred': Y[-fc:], 'alpha': alpha, 'beta': beta, 'rmse': rmse, 'aic':aic, 'ds':ds, 'prestd':prestd, 'fittedvalues': y[:-fc-1]}


def additive(x, m, fc, alpha=None, beta=None, gamma=None):
    '''
        HW指数平滑加法预测模型
        parameters:
            --------
            x:时间序列数据列表
            m:序列周期数(eg.序列为月份数据,m=12)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            ds
            prestd:残差标准差
    '''

    Y = x[:]

    if (alpha is None or beta is None or gamma is None):

        initial_values = array([0.3, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'additive'

        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type, m), bounds=boundaries, approx_grad=True)
        alpha, beta, gamma = parameters[0]

    print(alpha, beta, gamma)
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] - a[0] for i in range(m)]
    y = [a[0] + b[0] + s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(round(a[-1] + b[-1] + s[-m], 4))

        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])

    # rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    rmse = Rmse(Y[:-fc], y[:-fc - 1])
    aic = Aic(Y[:-fc], y[:-fc - 1], 3)
    ds = Ds(Y[:-fc], y[:-fc - 1])
    prestd = Stde([(m - n) for m, n in zip(Y[:-fc], y[:-fc - 1])])

    return {'pred': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse, 'aic':aic, 'ds':ds,
            'prestd':prestd, 'fittedvalues': y[:-fc-1]}


def multiplicative(x, m, fc, alpha=None, beta=None, gamma=None):
    '''
        HW指数平滑乘法预测模型
        parameters:
            --------
            x:时间序列数据列表
            m:序列周期数(eg.序列为月份数据,m=12)
            fc:预测期数
            alpha:alpha参数,初始值默认为空
            beta:beta参数,初始值默认为空
            gamma:gamma参数,初始值默认为空

        return:
            --------
            pred:预测值
            alpha:最优alpha值
            beta:最优beta值
            gamma:最优gamma值
            rmse
            aic
            ds
            prestd:残差标准差
    '''

    Y = x[:]

    if (alpha is None or beta is None or gamma is None):

        initial_values = array([0.0, 1.0, 0.0])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'multiplicative'

        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type, m), bounds=boundaries, approx_grad=True)
        alpha, beta, gamma = parameters[0]

    print(alpha, beta, gamma)
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] / a[0] for i in range(m)]
    y = [(a[0] + b[0]) * s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s[-m])

        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])

    # rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    rmse = Rmse(Y[:-fc], y[:-fc - 1])
    aic = Aic(Y[:-fc], y[:-fc - 1], 3)
    ds = Ds(Y[:-fc], y[:-fc - 1])
    prestd = Stde([(m - n) for m, n in zip(Y[:-fc], y[:-fc - 1])])

    return {'pred': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse, 'aic':aic, 'ds':ds,
            'prestd':prestd, 'fittedvalues': y[:-fc-1]}


def multiplicative_seasonal(x, m, fc, alpha = None, beta = None, gamma = None):
    """
    同multiplicative,返回数据作调整
    """
    Y = x[:]

    if (alpha == None or beta == None or gamma == None):

        initial_values = array([0.0, 1.0, 0.0])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'multiplicative'

        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma = parameters[0]

    print(alpha, beta, gamma)
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] / a[0] for i in range(m)]
    y = [(a[0] + b[0]) * s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append((a[-1] + b[-1]) * s[-m])

        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])

    # rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    rmse = Rmse(Y[:-fc], y[:-fc - 1])

    return {'pred': Y[-fc:], 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'rmse': rmse, 'seasonal': s, 'fittedvalues': y[:-fc-1]}


def compare_func(x, m, fc):
    '''
        根据rmse值比较4种基本预测模型的预测效果
        parameters:
            --------
            x:时间序列数据列表
            m:序列周期数
            fc:预测期数

        return:
            --------
            pred
            alpha(beta,gamma)
            rmse
            aic
            ds
    '''

    try:
        cp1 = simple(x, fc)  # 单参数
        cp2 = linear(x, fc)  # 简单
        cp3 = additive(x, m, fc)  # 加法
        cp4 = multiplicative(x, m, fc)  # 乘法
    except:
        cp1 = simple(x, fc)  # 单参数
        cp2 = linear(x, fc)  # 简单
        cp3 = additive(x, m, fc)  # 加法
        cp4 = {'rmse': 1000000, 'aic':1000000, 'ds':-100}

    min_rmse = min(cp1['rmse'], cp2['rmse'], cp3['rmse'], cp4['rmse'])  # 比较最小RMSE
    # max_rmse = max(cp1['rmse'], cp2['rmse'], cp3['rmse'], cp4['rmse'])  # 比较最大RMSE
    min_aic = min(cp1['aic'], cp2['aic'], cp3['aic'], cp4['aic'])  # 比较最小AIC
    max_aic = max(cp1['aic'], cp2['aic'], cp3['aic'], cp4['aic'])  # 比较最大AIC
    min_ds = min(cp1['ds'], cp2['ds'], cp3['ds'], cp4['ds'])  # 比较最小DS
    max_ds = max(cp1['ds'], cp2['ds'], cp3['ds'], cp4['ds'])  # 比较最大DS

    if min_aic != max_aic:
        # AIC 不相等,取最小AIC
        print('根据AIC选择最优')

        if min_aic == cp1['aic']:
            print('simple')
            return cp1
        elif min_aic == cp2['aic']:
            print('linear')
            return cp2
        elif min_aic == cp3['aic']:
            print('additive')
            return cp3
        elif min_aic == cp4['aic']:
            print('multiplicative')
            return cp4
        else:
            return 'Compare ERROR'

    elif min_ds != max_ds:
        # AIC 相等 DS 不相等,取最大DS
        print('根据DS选择最优')

        if max_ds == cp1['ds']:
            print('simple')
            return cp1
        elif max_ds == cp2['ds']:
            print('linear')
            return cp2
        elif max_ds == cp3['ds']:
            print('additive')
            return cp3
        elif max_ds == cp4['ds']:
            print('multiplicative')
            return cp4
        else:
            return 'Compare ERROR'

    else:
        # AIC 与 DS 均各自相等,取最小RMSE
        print('根据RMSE选择最优')

        if min_rmse == cp1['rmse']:
            print('simple')
            return cp1
        elif min_rmse == cp2['rmse']:
            print('linear')
            return cp2
        elif min_rmse == cp3['rmse']:
            print('additive')
            return cp3
        elif min_rmse == cp4['rmse']:
            print('multiplicative')
            return cp4
        else:
            return 'Compare ERROR'


# Function for Fitting our data to Linear model
def linear_model_main(X_paras, Y_paras, myfunc, m, fc, Fvalue, x_predict):
    '''
        线性回归预测模型
        parameters:
            --------
            X_paras:回归模型自变量,列表数据
            Y_paras:回归模型因变量,列表数据
            myfunc:参与时间序列预测的的函数名称,(linear,compare_func)之一
            m:序列周期数
            fc:预测期数
            Fvalue:F检验判断值
            x_predict:回归模型自变量未来值

        return:
            --------
            pred:不含自变量预测值
            price_pred:含自变量预测值
            F_value:回归模型F检验值
    '''

    X_paras = [[i] for i in X_paras]  # 自变量转为单一列表格式
    logsale = [round(log(x + 1), 3) for x in Y_paras]
    x_predict = [float(x_predict)] * fc
#     x_predict = [float(7.8)]*fc
    print(x_predict)
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_paras, logsale)
    # predict_outcome = regr.predict(predict_value) # 预测值
    R2 = regr.score(X_paras, logsale)  # 可决系数R平方
    F_value = R2 / (1 - R2) * (len(X_paras) - 2)  # F检验
    a = regr.intercept_  # 截距
    b = regr.coef_  # 回归系数
    print(a, b)
    # 判断显著性检验情况
    print('F:', F_value)
    if F_value > Fvalue:
        # F0.05(1,40)=4.08
        print('回归方程显著')
#         presale = [exp(a + b*p)-1 for p in X_paras] # 销量回归预测值
        presale = [round(exp(a + b * p) - 1, 4) for p in X_paras]  # 销量回归预测值
        print('回归：', presale)
#         ss = [i/j for i,j in zip(Y_paras,presale)] # 埃普西龙：实际销量与回归销量之比，消除价格影响因素
        ss = [float("%.4f" % (i / j)) for i, j in zip(Y_paras, presale)]
        print('非价格因素长度:', len(ss))
        print('ss:', ss)
        if myfunc == linear:
            print('linear')
            xxx = myfunc(ss, fc)
            print(xxx)
        else:
            xxx = myfunc(ss, m, fc)
            print(xxx)
        price_factor = [(exp(a + b * g) - 1) for g in x_predict]  # 价格因素
        # xxx['pred'] = [(exp(a + b*g)-1)*p for g,p in zip(x_predict,xxx['pred'])] # 恢复价格因素
        price_xxx = [p * q for p, q in zip(xxx['pred'], price_factor)]  # 恢复价格因素
        print('恢复价格：', price_xxx)
    else:
        print('回归方程不明显')
        print('序列长度:', len(X_paras))
        if myfunc == linear:
            print('linear')
            xxx = myfunc(logsale, fc)
        else:
            xxx = myfunc(logsale, m, fc)
        xxx['pred'] = [exp(i) - 1 for i in xxx['pred']]
        price_xxx = xxx['pred']
        print(xxx)
    return {'pred': xxx, 'price_pred': price_xxx, 'F_value': F_value}


def rolling(seq, wind, fc):
    '''
        含有rmse和预测值的移动平均
        parameters:
            --------
            seq:计算移动平均的时间序列
            wind:移动期数
            fc:预测期数

        return:
            --------
            pred:预测值
            rmse
            aic
            ds
            prestd:残差标准差
    '''

    mavg = seq.rolling(window=wind,center=False).mean() # 移动平均,wind表示移动期数
    mavg.dropna(inplace=True)  # 原地删除空值
    man_value = list(mavg.values)  # 移动平均序列值
    old_value = list(seq.values)[wind - 1:]  # 排除移动期原始序列值
    # rmse = sqrt(sum((m-n)**2 for m,n in zip(man_value,old_value)) / len(man_value))
    rmse = Rmse(old_value, man_value)
    aic = Aic(old_value, man_value, 0)
    ds = Ds(old_value, man_value)
    prestd = Stde([(m - n) for m, n in zip(old_value, man_value)])
    predata = [man_value[-1]] * fc * 7  # 预测值

    return {'pred': predata, 'rmse': rmse, 'aic':aic, 'ds':ds, 'prestd':prestd}
